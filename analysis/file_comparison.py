# This script compares the size of an existing PNG and its image data after it
# has been run through a given compressor.
#
# Usage example:
# $ cd ~/ee274_project
# $ python analysis/file_comparison.py \
#     -f test_data/kodim03.png \
#     -c filtered_zlib

import argparse
import os
from core.data_encoder_decoder import DataEncoder
from core.data_block import DataBlock
from PIL import Image
from png_compressors.filtered_zlib import FilteredZlib
from png_compressors.filtered_lz_arithmetic import FilteredLzArithmetic
from png_tools.file import read_image


def get_encoder(encoder_name: str, width: int, height: int) -> DataEncoder:
    """Returns encoder based on `encoder_name`."""
    sanitized = encoder_name.lower()
    if (sanitized == "filteredzlib" or sanitized == "filtered_zlib"):
        return FilteredZlib(width, height)
    elif (sanitized == "lzarithmetic" or sanitized == "lz_arithmetic"):
        return FilteredLzArithmetic(width, height)

    raise ValueError("Unrecognized encoder type: %s" % encoder_name)


def compare_file(filename: str, encoder_name: str):
    """Opens file and prints comparison."""

    file_stats = os.stat(filename)
    print("Original PNG size: %d bytes" % file_stats.st_size)

    with Image.open(filename) as image_f:
        width, height, block = read_image(image_f)

    encoder = get_encoder(encoder_name, width, height)
    encoded = encoder.encode_block(block)

    print("Compressed with %s: %d bytes" % (encoder_name, len(encoded) / 8))


def create_parser():
    """Creates argument parser."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-f",
                        "--filename",
                        help="file name",
                        required=True,
                        type=str)
    parser.add_argument("-c",
                        "--compressor",
                        help="custom compressor",
                        type=str)
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    compare_file(args.filename, args.compressor)


if __name__ == "__main__":
    main()

