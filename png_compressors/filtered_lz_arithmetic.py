import numpy as np
from compressors.arithmetic_coding import AECParams, ArithmeticEncoder
from compressors.elias_delta_uint_coder import EliasDeltaUintEncoder
from compressors.lz77 import LZ77Encoder, LZ77Sequence
from compressors.probability_models import AdaptiveIIDFreqModel, AdaptiveOrderKFreqModel
from core.data_block import DataBlock
from core.prob_dist import Frequencies
from png_compressors.core_encoder import CoreEncoder
from typing import List
from utils.bitarray_utils import BitArray, uint_to_bitarray


class FilteredLzArithmetic(CoreEncoder):
    """Image compressor using PNG filters + LZ77 + adaptive arithmetic."""

    def __init__(self, width, height):
        super().__init__(width, height)

        # Instantiate other encoders here.
        self.lz_encoder = LZ77Encoder()

    def encode_block(self, data_block: DataBlock) -> BitArray:
        """Encode block function for filtered zlib.

        The DataBlock is expected to be the R, G, B (and optionally A) channels
        of the image. Any deviation from this format will result in errors.
        """

        # 1. Break `data_block` into R, G, B, (and optionally: A) channels.
        channels = self._channelify(data_block)

        # 2. Apply "best" filter to each channel, using some heuristic.
        filtered = self._filter_channels(channels)

        # 3. Throw into LZ then arithmetic.
        return self._lz_with_arithmetic(DataBlock(filtered))

    def _lz_with_arithmetic(self, data_block: DataBlock) -> BitArray:
        """LZ but with arithmetic encoding."""

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
        encoder = ArithmeticEncoder(
            AECParams(), Frequencies(freq_dict=combined_block.get_counts()),
            AdaptiveIIDFreqModel)
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
    encoder = FilteredLzArithmetic(2, 2)
    data_list = np.array([
        [1, 2, 3, 4],  # R-values
        [255, 254, 253, 252],  # G-values
        [67, 189, 53, 90],  # B- values
        [39, 82, 102, 85],  # A-values
    ]).flatten().tolist()
    data_block = DataBlock(data_list)

    encoded = encoder.encode_block(data_block)

    assert encoded is not None
