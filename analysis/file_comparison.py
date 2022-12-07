# This script compares the size of an existing PNG and its image data after it
# has been run through a given compressor.

import argparse
import os
from core.data_encoder_decoder import DataEncoder
from core.data_block import DataBlock
from datetime import datetime
from PIL import Image
from png_compressors.filtered_arithmetic import FilteredArithmetic
from png_compressors.filtered_lz_arithmetic import FilteredLzArithmetic
from png_compressors.filtered_zlib import FilteredZlib
from png_compressors.filtered_zstd import FilteredZstd
from png_tools.png_filters import FilterHeuristic


def get_encoder(encoder_name: str, width: int, height: int, separate: bool,
                verbose: bool, heuristic: FilterHeuristic) -> DataEncoder:
    """Returns encoder based on `encoder_name`."""
    sanitized = encoder_name.lower()
    if (sanitized == "filteredzlib" or sanitized == "filtered_zlib"):
        return FilteredZlib(width,
                            height,
                            prepend_filter_type=separate,
                            debug_logs=verbose,
                            heuristic=heuristic)
    elif (sanitized == "filteredzstd" or sanitized == "filtered_zstd"):
        return FilteredZstd(width,
                            height,
                            prepend_filter_type=separate,
                            debug_logs=verbose,
                            heuristic=heuristic)
    elif (sanitized == "filteredlzarithmetic"):
        return FilteredLzArithmetic(width,
                                    height,
                                    prepend_filter_type=separate,
                                    debug_logs=verbose,
                                    heuristic=heuristic)
    elif (sanitized == "arithmetic0" or sanitized == "filtered_arithmetic0"):
        return FilteredArithmetic(width,
                                  height,
                                  prepend_filter_type=separate,
                                  debug_logs=verbose,
                                  heuristic=heuristic,
                                  order=0)
    elif (sanitized == "arithmetic1" or sanitized == "filtered_arithmetic1"):
        return FilteredArithmetic(width,
                                  height,
                                  prepend_filter_type=separate,
                                  debug_logs=verbose,
                                  heuristic=heuristic,
                                  order=1)
    elif (sanitized == "arithmetic2" or sanitized == "filtered_arithmetic2"):
        return FilteredArithmetic(width,
                                  height,
                                  prepend_filter_type=separate,
                                  debug_logs=verbose,
                                  heuristic=heuristic,
                                  order=2)
    elif (sanitized == "arithmetic3" or sanitized == "filtered_arithmetic3"):
        return FilteredArithmetic(width,
                                  height,
                                  prepend_filter_type=separate,
                                  debug_logs=verbose,
                                  heuristic=heuristic,
                                  order=3)
    elif (sanitized == "arithmetic4" or sanitized == "filtered_arithmetic4"):
        return FilteredArithmetic(width,
                                  height,
                                  prepend_filter_type=separate,
                                  debug_logs=verbose,
                                  heuristic=heuristic,
                                  order=4)

    raise ValueError("Unrecognized encoder type: %s" % encoder_name)


def get_heuristic(heuristic: str) -> FilterHeuristic:
    sanitized = heuristic.lower()
    chosen = None
    if (sanitized == "sum"):
        chosen = FilterHeuristic.ABSOLUTE_MINIMUM_SUM
    elif sanitized == "diffsum":
        chosen = FilterHeuristic.MINIMUM_DIFFERENCE_SUM

    print(f'[{str(datetime.now())}]: Using heuristic: {chosen}')
    return chosen


def compare_file(filename: str, encoder_name: str, separate: bool,
                 verbose: bool, heuristic: str):
    """Opens file and prints comparison."""

    # Open file.
    file_stats = os.stat(filename)
    print("[%s]: Original PNG size: %d bytes" %
          (str(datetime.now()), file_stats.st_size))

    image = Image.open(filename)
    print("[%s]: Image data has been successfully opened." %
          str(datetime.now()))

    # Configure encoder.
    chosen_heuristic = get_heuristic(heuristic)
    encoder = get_encoder(encoder_name, image.width, image.height, separate,
                          verbose, heuristic)
    print("[%s]: Beginning compression with %s." %
          (str(datetime.now()), encoder_name))
    print("[%s]: Encoding prefix separately: " % str(datetime.now()), separate)

    # Encode image.
    encoded = encoder.encode_image(image)

    # Calculate and output results.
    original_byte_length = image.width * image.height * len(image.getbands())
    compressed_byte_length = len(encoded) / 8
    print("[%s]: Compressed with %s: %d bytes" %
          (str(datetime.now()), encoder_name, compressed_byte_length))
    print("[%s]: That's a compression ratio of %.2f" % (str(
        datetime.now()), original_byte_length / float(compressed_byte_length)))


def create_parser():
    """Creates argument parser."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-f",
                        "--filename",
                        help="file name",
                        required=True,
                        type=str)
    parser.add_argument(
        "-c",
        "--compressor",
        help=
        "compressor name: one of filteredzlib, filteredlzarithmetic, filteredzstd, arithmetic<K> where <K>=0, 1, 2, 3, or 4",
        type=str)
    parser.add_argument("-s",
                        "--separate",
                        help="encode filter types separately",
                        action="store_true")
    parser.add_argument("-v",
                        "--verbose",
                        help="print verbose logs",
                        action="store_true")
    parser.add_argument(
        "-o",
        "--heuristic",
        default="sum",
        help="filter choice heuristic: 'sum' or 'diffsum'. default: 'sum'",
        type=str)
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    compare_file(args.filename, args.compressor, args.separate, args.verbose,
                 args.heuristic)


if __name__ == "__main__":
    main()
