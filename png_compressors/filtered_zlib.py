import numpy as np
from core.data_encoder_decoder import DataEncoder
from core.data_block import DataBlock
from external_compressors.zlib_external import ZlibExternalEncoder
from png_tools.png_filters import choose_filter
from typing import List
from utils.bitarray_utils import BitArray


class FilteredZlib(DataEncoder):
    """Image compressor using PNG filters + zlib.

    Since zlib is essentially DEFLATE, this compressor most closely mimics the
    PNG format, sans headers and fancy color features (gamma, palette indexing,
    ... etc).
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height

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

    def _channelify(self, data_block: DataBlock) -> List[List[int]]:
        """Breaks input data into channels.

        Expect input data to take the format of RGB(A). Any deviation will not
        be compressed properly.

        Args:
            data_block input data to compress

        Returns:
            Image data broken into color channels.
        """
        # 1. Break `data_block` into R, G, B, (and optionally: A) chunks.
        chunk_size = self.width * self.height
        chunks = []
        start = 0
        end = chunk_size

        while (end <= len(data_block.data_list)):
            chunks += [data_block.data_list[start:end]]
            start += chunk_size
            end += chunk_size

        # Sanity check time -- make sure only 3 - 4 channels.
        if (len(chunks) < 3 or len(chunks) > 4):
            raise ValueError("Expected only 3 - 4 channels, but got %d" %
                             len(chunks))
        return chunks

    def _filter_channels(self, chunks: List[List[int]]) -> List[int]:
        """Produces ndarray of filtered channels.

        Each filtered scanline will be prepended with the filter type.

        Args:
            chunks List of channel lists, ordered by RGB(A).
        Returns:
            List of filtered scanlines, ordered by RGB(A).
        """
        filtered = np.ndarray(shape=(self.height * len(chunks),
                                     self.width + 1),
                              dtype=int)
        row_index = 0
        for chunk in chunks:
            shaped_chunk = np.array(chunk).reshape(self.height, self.width)
            for r in range(self.height):
                curr = shaped_chunk[r]
                prev = np.zeros(self.width) if r > 0 else shaped_chunk[r - 1]
                filter_type, filtered_line = choose_filter(curr, prev)
                filtered[row_index][0] = filter_type
                filtered[row_index][1:] = filtered_line
                row_index += 1

        return filtered.flatten().tolist()


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
