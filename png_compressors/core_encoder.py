import numpy as np
from core.data_encoder_decoder import DataEncoder
from core.data_block import DataBlock
from png_tools.png_filters import choose_filter
from typing import List


class CoreEncoder(DataEncoder):
    """Extends `DataEncoder` with filtering methods."""

    def __init__(self, width, height):
        self.width = width
        self.height = height

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
