import numpy as np
from compressors.arithmetic_coding import AECParams, ArithmeticEncoder
from compressors.probability_models import AdaptiveOrderKFreqModel
from core.data_block import DataBlock
from png_compressors.core_encoder import CoreEncoder
from utils.bitarray_utils import BitArray
from typing import Tuple


class FilteredArithmetic(CoreEncoder):
    """Image compressor using PNG filters + arithmetic encoding.

    See parent class `CoreEncoder` for more attribute information.

    Attributes:
        order order of Markov chain.
    """

    def __init__(self,
                 width,
                 height,
                 prepend_filter_type: bool = False,
                 debug_logs: bool = False,
                 order: int = 0):
        super().__init__(width,
                         height,
                         prepend_filter_type=prepend_filter_type,
                         debug_logs=debug_logs)
        self.order = order

    def encode_block(self, data_block: DataBlock) -> BitArray:
        """Encode block function for filtered arithmetic.

        The DataBlock is expected to be a single channel of the image. Any
        deviation from this format will result in errors.
        """
        assert len(data_block.data_list) == (
            self.width *
            self.height), "Expected block of size %d but got size %d" % (
                self.width * self.height, len(data_block.data_list))

        if (self.debug_logs):
            print("[INFO]: beginning to encode block.")

        if (self.prepend_filter_type):
            if (self.debug_logs):
                print("[INFO]: prepending filter type.")
            # Break the channel into filter types and the filtered channel.
            filter_types, filtered_channel = self._filter_channel(
                data_block.data_list)

            # Now encode filter types.
            filter_type_encoder = ArithmeticEncoder(
                AECParams(), ([0, 1, 2, 3, 4], self.order),
                AdaptiveOrderKFreqModel)
            print("[INFO]: constructed filter type encoder.")

            encoded_filter_types = filter_type_encoder.encode_block(
                DataBlock(filter_types))

            if (self.debug_logs):
                print(
                    "[INFO]: Encoding the filter types for this block took %d bytes."
                    % (len(encoded_filter_types) / 8))

            # Encode channels.
            filtered_channel = DataBlock(filtered_channel)
            channel_encoder = ArithmeticEncoder(
                AECParams(),
                (list(filtered_channel.get_alphabet()), self.order),
                AdaptiveOrderKFreqModel)
            if (self.debug_logs):
                print("[INFO]: Constructed channel encoder.")
            encoded_channel = channel_encoder.encode_block(filtered_channel)
            return encoded_filter_types + encoded_channel

        # If we're not prepending the filter type, we can just encode the whole
        # block. First, prepend the filter type to each scanline.
        filtered = DataBlock(self._filter_channels([data_block.data_list]))
        alphabet = list(filtered.get_alphabet())

        # Throw into arithmetic encoder.
        # TODO(kathuan): why is this OOM-ing??
        channel_encoder = ArithmeticEncoder(AECParams(),
                                            (alphabet, self.order),
                                            AdaptiveOrderKFreqModel)

        if (self.debug_logs):
            print("[INFO]: Constructed channel encoder.")

        return channel_encoder.encode_block(DataBlock(filtered))


################################### TESTS ###################################


# Just make sure nothing explodes.
def test_encoder_constructs():
    encoder = FilteredArithmetic(4, 4)
    data_list = np.array([
        [1, 2, 3, 4],
        [255, 254, 253, 252],
        [67, 189, 53, 90],
        [39, 82, 102, 85],
    ]).flatten().tolist()
    data_block = DataBlock(data_list)

    encoded = encoder.encode_block(data_block)

    assert encoded is not None
