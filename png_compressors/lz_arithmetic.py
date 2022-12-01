import numpy as np
from compressors.arithmetic_coding import AECParams, ArithmeticEncoder
from compressors.elias_delta_uint_coder import EliasDeltaUintEncoder
from compressors.lz77 import LZ77Encoder, LZ77Sequence
from compressors.probability_models import AdaptiveOrderKFreqModel
from core.data_block import DataBlock
from core.data_encoder_decoder import DataDecoder, DataEncoder
from typing import List, Tuple
from utils.bitarray_utils import BitArray, uint_to_bitarray


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
        encoder = ArithmeticEncoder(AECParams(),
                                    (list(combined_block.get_alphabet()), 0),
                                    AdaptiveOrderKFreqModel)
        combined_encoding = encoder.encode_block(combined_block)
        return (uint_to_bitarray(len(combined_encoding), 32) +
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

        encoder = ArithmeticEncoder(AECParams(),
                                    (list(literal_block.get_alphabet()), 1),
                                    AdaptiveOrderKFreqModel)
        literals_encoding = encoder.encode_block(literal_block)
        for i in range(256):
            if i not in counts:
                counts[i] = 0
        counts_list = [counts[i] for i in range(256)]
        counts_encoding = EliasDeltaUintEncoder().encode_block(
            DataBlock(counts_list))
        return (uint_to_bitarray(len(counts_encoding), 32) + counts_encoding +
                uint_to_bitarray(len(literals_encoding), 32) +
                literals_encoding)


# Just make sure nothing explodes.
def test_encoder_constructs():
    encoder = LzArithmeticEncoder()
    data_list = np.array([
        [1, 2, 3, 4, 2, 2, 2, 2, 2, 2],  # R-values
        [255, 254, 254, 254, 254, 254, 254, 254, 253, 252],  # G-values
        [67, 189, 53, 90, 67, 18, 40, 63, 12, 46],  # B- values
        [39, 82, 102, 85, 2, 2, 2, 2, 2, 2],  # A-values
    ]).flatten().tolist()
    data_block = DataBlock(data_list)

    encoded = encoder.encode_block(data_block)

    assert encoded is not None
