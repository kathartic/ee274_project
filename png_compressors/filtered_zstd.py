import numpy as np
from core.data_block import DataBlock
from external_compressors.zstd_external import ZstdExternalEncoder
from png_compressors.core_encoder import CoreDecoder, CoreEncoder
from typing import Tuple
from utils.bitarray_utils import BitArray, get_random_bitarray


class FilteredZstd(CoreEncoder):
    """Image compressor using PNG filters + zstd."""


    def encode_block(self, data_block: DataBlock) -> BitArray:
        """Encode block function for filtered zlib.

        The DataBlock is expected to be a single channel of the image. Any
        deviation from this format will result in errors.
        """
        assert len(data_block.data_list) == (
            self.width *
            self.height), "Expected block of size %d but got size %d" % (
                self.width * self.height, len(data_block.data_list))

        if (self.prepend_filter_type):
            # Break the channel into filter types and the filtered channel.
            filter_types, filtered_channel = self._filter_channel(
                data_block.data_list)
            # Now encode.
            encoded_filter_types = ZstdExternalEncoder().encode_block(
                DataBlock(filter_types))
            encoded_channel = ZstdExternalEncoder().encode_block(
                DataBlock(filtered_channel))

            if (self.debug_logs):
                print(
                    "[INFO]: Encoding the filter types for this block took %d bytes."
                    % (len(encoded_filter_types) / 8))

            return encoded_filter_types + encoded_channel

        # If we're not prepending the filter type, we can just encode the whole
        # block. First, prepend the filter type to each scanline.
        filtered = self._filter_channels([data_block.data_list])

        # Throw into zstd.
        return ZstdExternalEncoder().encode_block(DataBlock(filtered))


################################### TESTS ###################################


# Just make sure nothing explodes.
def test_encoder_constructs():
    encoder = FilteredZstd(4, 4)
    data_list = np.array([
        [1, 2, 3, 4],
        [255, 254, 253, 252],
        [67, 189, 53, 90],
        [39, 82, 102, 85],
    ]).flatten().tolist()
    data_block = DataBlock(data_list)

    encoded = encoder.encode_block(data_block)

    assert encoded is not None
