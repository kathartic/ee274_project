import numpy as np
from compressors.arithmetic_coding import AECParams, ArithmeticEncoder
from compressors.probability_models import AdaptiveOrderKFreqModel
from core.data_encoder_decoder import DataEncoder, DataDecoder
from core.data_block import DataBlock
from PIL import Image
from png_tools.png_filters import choose_filter
from typing import List, Tuple
from utils.bitarray_utils import BitArray, uint_to_bitarray


class CoreEncoder(DataEncoder):
    """Extends `DataEncoder` with filtering methods.

    Attributes:
        width: integer width of the image in pixels.
        height: integer height of the image in pixels.
        prepend_filter_type: boolean that controls if filter type is prepended
            as a block or not. If true, filter type will be encoded separately
            at the beginning instead of prepended to each scanline. 
        debug_logs: boolean that controls print logging.
    """

    def __init__(self,
                 width: int,
                 height: int,
                 prepend_filter_type: bool = False,
                 debug_logs: bool = False):
        self.width = width
        self.height = height
        self.prepend_filter_type = prepend_filter_type
        self.debug_logs = debug_logs

        # Let's just use arithmetic encoding for the filter type encoder. We
        # know the alphabet (since only have 5 filter types), and for now,
        # hardcode an assumption that it's a 1st-order Markov.
        self.filter_type_encoder = ArithmeticEncoder(AECParams(),
                                                     ([0, 1, 2, 3, 4], 1),
                                                     AdaptiveOrderKFreqModel)

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

    def _filter_channel(self,
                        channel: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Produces tuple of filter_type, filtered_channel.

        Instead of prepending to the scanline as PNG does, returns tuple of
        filter_types, filtered_channels. There should be self.height entries in
        the filter_type list.

        Args:
            channel Entries for a given channel.
        Returns:
            Tuple of filter_types, filtered_channels. filtered_channels is the
            filtered version of input `channel`.
        """
        channel_block = np.array(channel).reshape(self.height, self.width)
        filtered = np.ndarray(shape=(self.height, self.width))
        filter_types = np.ndarray(shape=(self.height, ), dtype=int)

        for i in range(self.height):
            curr = channel_block[i]
            prev = np.zeros(self.width) if i == 0 else channel_block[i - 1]
            filter_type, filtered_line = choose_filter(curr, prev)
            filter_types[i] = filter_type
            filtered[i] = filtered_line

        if (self.debug_logs):
            print("[INFO]: Filter type counts:")
            print(DataBlock(filter_types).get_counts())

        return filter_types, filtered.flatten()

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
        for i in range(len(chunks)):
            chunk = chunks[i]
            start = i * self.height
            end = (i + 1) * self.height
            filter_types, filtered_chunk = self._filter_channel(chunk)
            filtered[start:end, 0] = filter_types
            filtered[start:end,
                     1:] = filtered_chunk.reshape(self.height, self.width)

        return filtered.flatten().tolist()

    def encode_image(self, image: Image) -> BitArray:
        """Convenience method to encode image."""

        if (image.mode != "RGB" and image.mode != "RGBA"):
            if (self.debug_logs):
                print("[WARN]: converting image from %s to RGB." % image.mode)
            image = image.convert("RGB")

        # Encode image dimensions. For simplicity, height and width are maxed
        # out at 2^32 pixels.
        encoded_width = uint_to_bitarray(self.width)
        encoded_height = uint_to_bitarray(self.height)
        encoded_image = (uint_to_bitarray(len(encoded_width), 32) +
                         encoded_width +
                         uint_to_bitarray(len(encoded_height), 32) +
                         encoded_height)

        for i in range(len(image.getbands())):
            channel = list(image.getdata(i))
            encoded_image += self.encode_block(DataBlock(channel))

        return encoded_image


class CoreDecoder(DataDecoder):
    """Extends `DataDecoder` with reverse-filtering methods."""

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def _reverse_filter_channels(self, data: List[int]) -> List[int]:
        """Produces list of reverse-filtered channels.

        Args:
            data List of filter type and integers. Each scanline should be
                 prepended with a filter type. Scanline width is defined by
                 the property `self.width`. Each non-filter type integer
                 represents result of filter pre-processing.
        Returns:
            List of reverse-filtered channels.
        """

        # TODO: implement.
        return []


############################## TESTS ##############################


def test_filter_channel():
    test_channel = np.array([
        [1, 2, 3],  # expect sub filter
        [5, 5, 5],  # expect paeth filter
        [5, 5, 5],  # expect up filter
        [0, 1, 0],  # expect none filter here
    ]).flatten().tolist()
    expected_filter_types = [1, 4, 2, 0]
    expected_filtered = np.array([
        [1, 1, 1],
        [4, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
    ]).flatten().tolist()

    encoder = CoreEncoder(3, 4)
    filter_types, filtered = encoder._filter_channel(test_channel)

    np.testing.assert_array_equal(filter_types,
                                  expected_filter_types,
                                  err_msg="Filter types are not equal.")
    np.testing.assert_array_equal(
        filtered,
        expected_filtered,
        err_msg="Filtered channel arrays are not equal.")


def test_filter_channels():
    test_channel = np.array([
        [4, 8, 25],  # expect sub
        [2, 5, 15],  # expect avg
    ]).flatten().tolist()
    expected_channel = np.array([
        [1, 4, 4, 17],
        [3, 0, 0, 0],
    ]).flatten().tolist()
    expected = expected_channel + expected_channel + expected_channel
    encoder = CoreEncoder(3, 2)

    filtered = encoder._filter_channels(
        [test_channel, test_channel, test_channel])

    np.testing.assert_array_equal(np.array(filtered), np.array(expected))
