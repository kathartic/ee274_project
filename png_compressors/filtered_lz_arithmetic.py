import numpy as np
from core.data_block import DataBlock
from png_compressors.core_encoder import CoreDecoder, CoreEncoder
from png_compressors.lz_arithmetic import LzArithmeticDecoder, LzArithmeticEncoder
from typing import List, Tuple
from utils.bitarray_utils import BitArray, uint_to_bitarray


class FilteredLzArithmetic(CoreEncoder):
    """Image compressor using PNG filters + LZ77 + adaptive arithmetic."""

    def __init__(self, width, height):
        super().__init__(width, height)

        # Instantiate other encoders here.
        self.lz_encoder = LzArithmeticEncoder()

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
        return self.lz_encoder.encode_block(DataBlock(filtered))


class FilteredLzArithmeticDecoder(CoreDecoder):
    """See `FilteredLzArithmetic` for more details."""

    def __init__(self, width, height):
        super().__init__(width, height)

        self.lz_decoder = LzArithmeticDecoder()

    def decode_block(self, bitarray: BitArray) -> Tuple[DataBlock, int]:
        filtered_data, bits_consumed = self.lz_decoder.decode_block(bitarray)
        channel_data = self._reverse_filter_channels(filtered_data.data_list)
        return DataBlock(channel_data), bits_consumed

################################## TESTS #####################################

# Just make sure nothing explodes.
def test_encoder_constructs():
    encoder = FilteredLzArithmetic(2, 2)
    decoder = FilteredLzArithmeticDecoder(2, 2)
    data_list = np.array([
        [1, 2, 3, 4],  # R-values
        [255, 254, 253, 252],  # G-values
        [67, 189, 53, 90],  # B- values
        [39, 82, 102, 85],  # A-values
    ]).flatten().tolist()
    data_block = DataBlock(data_list)

    encoded = encoder.encode_block(data_block)
    assert encoded is not None

    decoded, num_bits = decoder.decode_block(encoded)
    assert decoded is not None
    assert num_bits > 0
