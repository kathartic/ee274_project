import numpy as np
from core.data_block import DataBlock
from external_compressors.zlib_external import ZlibExternalDecoder, ZlibExternalEncoder
from png_compressors.core_encoder import CoreDecoder, CoreEncoder
from typing import Tuple
from utils.bitarray_utils import BitArray, get_random_bitarray


class FilteredZlib(CoreEncoder):
    """Image compressor using PNG filters + zlib.

    Since zlib is essentially DEFLATE, this compressor most closely mimics the
    PNG format, sans headers and fancy color features (gamma, palette indexing,
    ... etc).
    """

    def __init__(self, width, height):
        super().__init__(width, height)

        # Instantiate here since zlib uses some common state across encoded
        # blocks.
        self.zlib_encoder = ZlibExternalEncoder()

    def encode_block(self, data_block: DataBlock) -> BitArray:
        """Encode block function for filtered zlib.

        The DataBlock is expected to be the R, G, B (and optionally A) channels
        of the image. Any deviation from this format will result in errors.
        """

        # 1. Break `data_block` into R, G, B, (and optionally: A) channels.
        channels = self._channelify(data_block)

        # 2. Apply "best" filter to each channel, using some heuristic.
        filtered = self._filter_channels(channels)

        # 3. Throw into zlib.
        return self.zlib_encoder.encode_block(DataBlock(filtered))


class FilteredZlibDecoder(CoreDecoder):
    """Image decompressor using PNG filters + zlib.

    See `FilteredZlib` for more details.
    """

    def __init__(self, width, height):
        super().__init__(width, height)

        # Instantiate here since zlib uses some common state across encoded
        # blocks.
        self.zlib_decoder = ZlibExternalDecoder()

    def decode_block(self, bitarray: BitArray) -> Tuple[DataBlock, int]:
        """Decode block method for filtered zlib."""

        filtered_data, bits_consumed = self.zlib_decoder.decode_block(bitarray)
        channel_data = self._reverse_filter_channels(filtered_data)
        return DataBlock(channel_data), bits_consumed


################################### TESTS ###################################


# Just make sure nothing explodes.
def test_encoder_constructs():
    encoder = FilteredZlib(2, 2)
    data_list = np.array([
        [1, 2, 3, 4],  # R-values
        [255, 254, 253, 252],  # G-values
        [67, 189, 53, 90],  # B- values
        [39, 82, 102, 85],  # A-values
    ]).flatten().tolist()
    data_block = DataBlock(data_list)

    encoded = encoder.encode_block(data_block)

    assert encoded is not None


# Make sure nothing explodes. Can replace this test (and the one above it) with
# an actual lossless compression check once reverse filtering implemented.
def test_decoder_constructs():
    decoder = FilteredZlibDecoder(3, 1)
    bitarray = get_random_bitarray(9)

    decoded, bits_decoded = decoder.decode_block(bitarray)

    assert decoded is not None
    assert bits_decoded > 0
