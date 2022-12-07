import numpy as np
from compressors.arithmetic_coding import AECParams, ArithmeticEncoder
from compressors.elias_delta_uint_coder import EliasDeltaUintEncoder
from compressors.lz77 import LZ77Decoder, LZ77Encoder, LZ77Sequence
from compressors.probability_models import AdaptiveOrderKFreqModel
from core.data_block import DataBlock
from core.data_encoder_decoder import DataEncoder
from typing import List
from utils.bitarray_utils import BitArray, uint_to_bitarray


class LzArithmeticEncoder(DataEncoder):
    """Same as `LZ77Encoder`, but uses arithmetic encoding for literals."""

    def encode_block(self, data_block: DataBlock) -> BitArray:
        lz_encoder = LZ77Encoder()
        sequences, literals = lz_encoder.lz77_parse_and_generate_sequences(
            data_block)

        # Encode sequences and literals with arithmetic.
        sequence_encoding = lz_encoder.encode_lz77_sequences(sequences)
        literals_encoding = self._encode_literals_arithmetic(literals)
        return sequence_encoding + literals_encoding

    def _encode_literals_arithmetic(self, literals: List) -> BitArray:
        """Encode LZ77 literals with arithmetic.

        Same as `encode_literals()` method of LZ77Encoder, but using arithmetic
        encoding.
        """
        literal_block = DataBlock(literals)
        counts = literal_block.get_counts()
        if not len(counts):
            return uint_to_bitarray(0, 32)

        # Since we'll be encoding indexed values only, know that this ranges
        # from 0 to 255.  This will make things easier on the decoding side.
        encoder_alphabet = [i for i in range(256)]
        aec_params = AECParams()
        freq_model_enc = AdaptiveOrderKFreqModel(
            alphabet=encoder_alphabet,
            k=1,
            max_allowed_total_freq=aec_params.MAX_ALLOWED_TOTAL_FREQ,
        )
        encoder = ArithmeticEncoder(aec_params, freq_model_enc)
        return encoder.encode_block(literal_block)


################################ TESTS ###################################


# Just make sure nothing explodes.
def test_constructs():
    encoder = LzArithmeticEncoder()
    data_list = np.array([
        [1, 2, 3, 4, 2, 2, 2, 2, 2, 2],
        [255, 254, 254, 254, 254, 254, 254, 254, 253, 252],
        [67, 189, 53, 90, 67, 18, 40, 63, 12, 46],
        [39, 82, 102, 85, 2, 2, 2, 2, 2, 2],
    ]).flatten().tolist()
    data_block = DataBlock(data_list)

    encoded = encoder.encode_block(data_block)
    assert encoded is not None
