import numpy as np
from compressors.arithmetic_coding import AECParams, ArithmeticEncoder
from compressors.probability_models import AdaptiveOrderKFreqModel
from core.data_block import DataBlock
from png_compressors.core_encoder import CoreEncoder
from png_tools.png_filters import FilterHeuristic
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
                 heuristic: FilterHeuristic = FilterHeuristic.ABSOLUTE_MINIMUM_SUM,
                 order: int = 0):
        super().__init__(width,
                         height,
                         prepend_filter_type=prepend_filter_type,
                         debug_logs=debug_logs,
                         heuristic=heuristic)
        self.order = order

    def _arithmetic_encode(self, data_block: DataBlock) -> BitArray:
        aec_params = AECParams()
        freq_model_enc = AdaptiveOrderKFreqModel(
            alphabet=list(data_block.get_alphabet()),
            k=self.order,
            max_allowed_total_freq=aec_params.MAX_ALLOWED_TOTAL_FREQ,
        )
        arithmetic_encoder = ArithmeticEncoder(aec_params, freq_model_enc)
        if (self.debug_logs):
            print("[INFO]: Constructed arithmetic encoder.")

        encoding = arithmetic_encoder.encode_block(data_block)
        if (self.debug_logs):
            print("[INFO]: Encoded block with arithmetic encoder.")

        return encoding
        

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
            # Break the channel into filter types and the filtered channel.
            filter_types, filtered_channel = self._filter_channel(
                data_block.data_list)

            # Now encode filter types.
            if (self.debug_logs):
                print("[INFO]: encoding filter types.")
            encoded_filter_types = self._arithmetic_encode(DataBlock(filter_types))
            if (self.debug_logs):
                print(
                    "[INFO]: Encoding the filter types for this block took %d bytes."
                    % (len(encoded_filter_types) / 8))

            # Encode channels.
            if (self.debug_logs):
                print("[INFO]: encoding channel (sans filter types).")
            encoded_channel = self._arithmetic_encode(DataBlock(filtered_channel))
            return encoded_filter_types + encoded_channel

        # If we're not prepending the filter type, we can just encode the whole
        # block. First, prepend the filter type to each scanline.
        filtered = DataBlock(self._filter_channels([data_block.data_list]))

        # Throw into arithmetic encoder.
        if (self.debug_logs):
            print("[INFO]: encoding channel (with filter types)")

        # TODO(kathuan): why is this OOM-ing??
        return self._arithmetic_encode(filtered)


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
