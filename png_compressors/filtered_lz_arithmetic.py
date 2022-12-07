import numpy as np
from core.data_block import DataBlock
from png_compressors.core_encoder import CoreEncoder
from png_compressors.lz_arithmetic import LzArithmeticEncoder
from typing import List, Tuple
from utils.bitarray_utils import BitArray, uint_to_bitarray


class FilteredLzArithmetic(CoreEncoder):
    """Image compressor using PNG filters + LZ77 + adaptive arithmetic."""

    def encode_block(self, data_block: DataBlock) -> BitArray:
        """Encode block function for filtered zlib.

        The DataBlock is expected to be a single channel of the image. Any
        deviation from this format will result in errors.
        """

        # Check to see the user followed input instructions.
        expected_size = self.width * self.height
        actual_size = len(data_block.data_list)
        assert actual_size == expected_size, f"Expected block of size {expected_size} but got size {actual_size}"

        if (self.prepend_filter_type):
            raise NotImplementedError("sorry i didn't get around to this")

        # Prepend the filter type to each scanline.
        filtered = self._filter_channels([data_block.data_list])

        # Throw into lz arithmetic encoder
        return LzArithmeticEncoder().encode_block(DataBlock(filtered))


################################## TESTS #####################################


# Just make sure nothing explodes.
def test_encoder_constructs():
    encoder = FilteredLzArithmetic(4, 4)
    data_list = np.array([
        [1, 2, 3, 4],
        [255, 254, 253, 252],
        [67, 189, 53, 90],
        [39, 82, 102, 85],
    ]).flatten().tolist()
    data_block = DataBlock(data_list)

    encoded = encoder.encode_block(data_block)
    assert encoded is not None
