# This script compares the size of an existing PNG and its image data after it
# has been run through a given compressor.
#
# Usage example:
# $ cd ~/ee274_project
# $ TEST_DATA=test_data/kodim03.png
# $ python analysis/file_comparison.py \
#     -f $TEST_DATA \
#     -c filtered_zlib
#
# Since some compressors may take > 1 hour to compress, it's recommended to pipe
# results to run this script as a background process, or pipe output to a file
# instead (the latter is given as example here):
#
# $ echo \
#    "$(python analysis/file_comparison.py -f $TEST_DATA -c lzarithmetic)" > results.txt

import argparse
import os
from core.data_encoder_decoder import DataEncoder
from core.data_block import DataBlock
from datetime import datetime
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
    print("[%s]: Original PNG size: %d bytes" %
          (str(datetime.now()), file_stats.st_size))

    with Image.open(filename) as image_f:
        width, height, block = read_image(image_f)

    print("[%s]: Image data has been successfully read." % str(datetime.now()))
    print("[%s]: Beginning compression with %s." %
          (str(datetime.now()), encoder_name))
    encoder = get_encoder(encoder_name, width, height)
    encoded = encoder.encode_block(block)

    print("[%s]: Compressed with %s: %d bytes" %
          (str(datetime.now()), encoder_name, len(encoded) / 8))


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
