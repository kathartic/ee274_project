import numpy as np
from compressors.arithmetic_coding import AECParams, ArithmeticDecoder, ArithmeticEncoder
from compressors.elias_delta_uint_coder import EliasDeltaUintDecoder, EliasDeltaUintEncoder
from compressors.lz77 import LZ77Decoder, LZ77Encoder, LZ77Sequence
from compressors.probability_models import AdaptiveOrderKFreqModel
from core.data_block import DataBlock
from core.data_encoder_decoder import DataDecoder, DataEncoder
from typing import List, Tuple
from utils.bitarray_utils import BitArray, bitarray_to_uint, uint_to_bitarray


class LzArithmeticEncoder(DataEncoder):
    """Same as `LZ77Encoder`, but uses arithmetic encoding at the end."""

    def __init__(self):
        self.lz_encoder = LZ77Encoder()

    def encode_block(self, data_block: DataBlock) -> BitArray:
        sequences, literals = self.lz_encoder.lz77_parse_and_generate_sequences(
            data_block)

        # Encode sequences and literals with arithmetic.
        sequence_encoding = self._encode_sequences_arithmetic(sequences)
        literals_encoding = self._encode_literals_arithmetic(literals)
        return sequence_encoding + literals_encoding

    def _encode_sequences_arithmetic(self,
                                     sequences: LZ77Sequence) -> BitArray:
        """Encode LZ77 sequences with arithmetic.

        Same as `encode_lz77_sequences()` method of LZ77Encoder, but using
        arithmetic encoding.
        """
        min_match_length = self.lz_encoder.min_match_length
        literal_counts = [l.literal_count for l in sequences]
        match_lengths_processed = [
            l.match_length - min_match_length for l in sequences
        ]
        match_offsets_processed = [l.match_offset - 1 for l in sequences]
        combined = ([min_match_length] + literal_counts +
                    match_lengths_processed + match_offsets_processed)

        combined_block = DataBlock(combined)
        combined_list = list(combined_block.get_alphabet())
        encoder = ArithmeticEncoder(AECParams(), (combined_list, 0),
                                    AdaptiveOrderKFreqModel)
        combined_encoding = encoder.encode_block(combined_block)

        # HEADS UP! This is different from the original LZ77: we need to also
        # transmit the alphabet, so the decoder knows how to construct the
        # arithmetic decoder. Let's transmit the alphabet using Elias Delta
        # (make things easier).
        alphabet_encoding = EliasDeltaUintEncoder().encode_block(
            DataBlock(combined_list))

        # Return the encoded alphabet and the encoded sequences.
        # TODO(kathuan): think about the # of bits you need to encode the
        # alphabet a bit more...
        return (uint_to_bitarray(len(alphabet_encoding), 64) +
                alphabet_encoding +
                uint_to_bitarray(len(combined_encoding), 32) +
                combined_encoding)

    def _encode_literals_arithmetic(self, literals: List) -> BitArray:
        """Encode LZ77 literals with arithmetic.

        Same as `encode_literals()` method of LZ77Encoder, but using arithmetic
        encoding.
        """
        literal_block = DataBlock(literals)
        counts = literal_block.get_counts()
        if not len(counts):
            return uint_to_bitarray(0, 32)

        # Since we'll be encoding indexed values only, know that this ranges
        # from 0 to 255.  This will make things easier on the decoding side.
        encoder_alphabet = [i for i in range(256)]
        encoder = ArithmeticEncoder(AECParams(), (encoder_alphabet, 1),
                                    AdaptiveOrderKFreqModel)
        literals_encoding = encoder.encode_block(literal_block)
        return (uint_to_bitarray(len(literals_encoding), 32) +
                literals_encoding)


class LzArithmeticDecoder(DataDecoder):
    """See `LzArithmeticEncoder` for more details."""

    def __init__(self):
        self.lz_decoder = LZ77Decoder()

    def _decode_lz77_arithmetic_sequences(
            self, bitarray: BitArray) -> Tuple[List[LZ77Sequence], int]:
        """Entropy decodes LZ77 sequences.

        Args:
            bitarray: encoded bit array
        Returns:
            tuple of decoded sequences, number of bits consumed
        """

        # 1. Get encoded alphabet bits
        # ------------------------------------------------
        encoded_alphabet_bitarray = bitarray[:64]
        encoded_alphabet_size = bitarray_to_uint(encoded_alphabet_bitarray)

        # 2. Get encoded sequence bits
        # ------------------------------------------------
        # `encoded_block_size_offset` marks the end (exclusive) of the encoded
        # alphabet, and marks the beginning (inclusive) of the encoded
        # sequences' size definition.
        encoded_block_size_offset = 64 + encoded_alphabet_size
        encoded_alphabet_bits = bitarray[64:encoded_block_size_offset]

        # `sequence_offset` is where the size of the sequence stops being
        # defined (exclusive) and where the actual sequence data begins
        # (inclusive).
        sequence_offset = encoded_block_size_offset + 32

        # These bits define how large the sequence data is.
        sequence_block_size_bitarray = bitarray[
            encoded_block_size_offset:sequence_offset]
        sequence_block_size = bitarray_to_uint(sequence_block_size_bitarray)

        # This is the actual sequence data:
        num_bits_consumed = sequence_offset + sequence_block_size
        sequence_bitarray = bitarray[sequence_offset:num_bits_consumed]

        # 3. Decode alphabet
        # ------------------------------------------------
        alphabet_block, num_alpha_bits = EliasDeltaUintDecoder().decode_block(
            encoded_alphabet_bits)

        # Sanity check.
        assert num_alpha_bits == encoded_alphabet_size, "Expected to consume %d bits for alphabet decoding but consumed %d" % (
            encoded_alphabet_size, num_alpha_bits)
        num_bits_consumed += encoded_block_size_offset

        # 4. Decode sequences
        # ------------------------------------------------
        decoder = ArithmeticDecoder(AECParams(), (alphabet_block.data_list, 0),
                                    AdaptiveOrderKFreqModel)
        combined_decoded, combined_bits = decoder.decode_block(
            sequence_bitarray)
        combined_decoded = combined_decoded.data_list

        # Sanity checks.
        assert sequence_block_size == combined_bits
        assert (len(combined_decoded) - 1) % 3 == 0

        # From here on out, just copy code from existing LZ77 implementation.
        num_sequences = (len(combined_decoded) - 1) // 3
        min_match_length = combined_decoded[0]
        literal_counts = combined_decoded[1:1 + num_sequences]
        match_lengths = [
            l + min_match_length
            for l in combined_decoded[1 + num_sequences:1 + 2 * num_sequences]
        ]
        match_offsets = [
            l + 1 for l in combined_decoded[1 + 2 * num_sequences:1 +
                                            3 * num_sequences]
        ]
        lz77_sequences = [
            LZ77Sequence(l[0], l[1], l[2])
            for l in zip(literal_counts, match_lengths, match_offsets)
        ]
        return lz77_sequences, num_bits_consumed

    def _decode_literals(self, bitarray: BitArray) -> Tuple[list, int]:
        """Entropy decodes literals.

        Almost identical to `decode_literals` from LZ77Decoder, but this one
        uses arithmetic coding.

        Args:
            bitarray: encoded bitarray
        Returns:
            tuple of literals and # of bits consumed.
        """
        literal_size = bitarray_to_uint(bitarray[:32])
        num_bits_consumed = 32
        if literal_size == 0:
            return [], num_bits_consumed

        # Construct arithmetic decoder.
        decoder = ArithmeticDecoder(AECParams(), ([i for i in range(256)], 1),
                                    AdaptiveOrderKFreqModel)
        literals_decoded, num_literal_bits = decoder.decode_block(
            bitarray[num_bits_consumed:(num_bits_consumed + literal_size)])

        # Sanity check.
        assert num_literal_bits == literal_size

        num_bits_consumed += num_literal_bits
        return literals_decoded.data_list, num_bits_consumed

    def decode_block(self, bitarray: BitArray) -> Tuple[DataBlock, int]:
        sequences, num_bits_consumed_sequences = self._decode_lz77_arithmetic_sequences(
            bitarray)
        remaining_bitarray = bitarray[num_bits_consumed_sequences:]
        literals, num_bits_consumed_literals = self._decode_literals(
            remaining_bitarray)
        num_bits_consumed = num_bits_consumed_sequences + num_bits_consumed_literals
        decoded_block = DataBlock(
            self.lz_decoder.execute_lz77_sequences(literals, lz77_sequences))
        return decoded_block, num_bits_consumed


################################ TESTS ###################################


# Just make sure nothing explodes.
def test_constructs():
    encoder = LzArithmeticEncoder()
    decoder = LzArithmeticDecoder()
    data_list = np.array([
        [1, 2, 3, 4, 2, 2, 2, 2, 2, 2],  # R-values
        [255, 254, 254, 254, 254, 254, 254, 254, 253, 252],  # G-values
        [67, 189, 53, 90, 67, 18, 40, 63, 12, 46],  # B- values
        [39, 82, 102, 85, 2, 2, 2, 2, 2, 2],  # A-values
    ]).flatten().tolist()
    data_block = DataBlock(data_list)

    encoded = encoder.encode_block(data_block)
    assert encoded is not None

    decoded, num_bits_consumed = decoder.decode_block(encoded)
    assert decoded is not None
    assert num_bits_consumed > 0
    
