"""
Functions that perform PNG filtering. 

There are 5 PNG filter types: 
(0): None 
(1): Sub 
(2): Up
(3): Average
(4): Paeth 

Each filter function will take a list (or 2) of integers, representing the scanline. 
Each filter function will return a filtered list, and the filter type (int from 0 to 4). 
"""
import numpy as np
import random
from typing import List, Tuple


# None: Returns the same scanline (filter type 0)
def none(scanline: list) -> Tuple[List[int], int]:
    filter_type = 0
    return scanline, filter_type


# Sub: Each pixel value is replaced with the difference between it and the value to the left (filter type 1)
def sub(scanline: list) -> Tuple[list, int]:
    filter_type = 1
    length = len(scanline)
    sub = [int] * length

    for i in range(length):
        if (i == 0):
            sub[0] = scanline[0]
        else:
            sub[i] = (scanline[i] - scanline[i - 1]) % 256

    return sub, filter_type


# Up: Each pixel value is replaced with the difference between it and the pixel above it (filter type 2)
def up(curr_scanline: list, prev_scanline: list) -> Tuple[list, int]:
    filter_type = 2
    length = len(curr_scanline)
    up = [int] * length

    for i in range(0, length):
        up[i] = (curr_scanline[i] - prev_scanline[i]) % 256

    return up, filter_type


# Average: Each pixel value is replaced the difference between it and
# the average of the corresponding pixels to its left and above it, truncating any fractional part (filter type 3)
def average(curr_scanline: list, prev_scanline: list) -> Tuple[list, int]:
    filter_type = 3
    length = len(curr_scanline)
    average = [int] * length

    for i in range(0, length):
        if (i == 0):
            average[0] = int(
                (curr_scanline[0] - np.floor(prev_scanline[0] / 2)) % 256)
        else:
            average[i] = int((curr_scanline[i] - np.floor(
                (prev_scanline[i] + curr_scanline[i - 1]) / 2)) % 256)

    return average, filter_type


# Paeth Predictor: Special predictor used in the Paeth filter
def paethPredictor(left: int, upper: int, upper_left: int) -> int:
    p = left + upper - upper_left
    p_left = abs(p - left)
    p_upper = abs(p - upper)
    p_upper_left = abs(p - upper_left)

    if (p_left <= p_upper and p_left <= p_upper_left):
        return left
    elif (p_upper <= p_upper_left):
        return upper
    else:
        return upper_left


# Paeth: Operates on current pixel and left, above, upper left using Paeth operator (filter type 4)
def paeth(curr_scanline: list, prev_scanline: list) -> Tuple[list, int]:
    filter_type = 4
    length = len(curr_scanline)
    paeth = [int] * length

    for i in range(0, length):
        if (i == 0):
            paeth[0] = (curr_scanline[0] -
                        paethPredictor(0, prev_scanline[0], 0)) % 256
        else:
            paeth[i] = (curr_scanline[i] -
                        paethPredictor(curr_scanline[i - 1], prev_scanline[i],
                                       prev_scanline[i - 1])) % 256

    return paeth, filter_type


def choose_filter(curr: List[int], prev: List[int]) -> Tuple[int, List[int]]:
    """Returns best filter for `curr` scanline.

    The 'best' filter is chosen by smallest sum of absolute values of outputs.
    There may be better heuristics.

    Args:
        curr: Current scanline.
        prev: Previous scanline.

    Returns:
        tuple of filtered type, and filtered scanline.
    """
    filter_type = 0
    filtered = curr
    smallest_sum = np.sum(curr)

    # No need to even try if the current line is all 0s.
    if not smallest_sum:
        return filter_type, filtered

    # Try the sub() filter.
    sub_res, sub_filter = sub(curr)
    sub_sum = np.sum(sub_res)
    if (sub_sum < smallest_sum):
        filter_type = sub_filter
        filtered = sub_res
        smallest_sum = sub_sum

    if not smallest_sum:
        return filter_type, filtered

    # Try the up() filter.
    up_res, up_filter = up(curr, prev)
    up_sum = np.sum(up_res)
    if (up_sum < smallest_sum):
        filter_type = up_filter
        filtered = up_res
        smallest_sum = up_sum

    if not smallest_sum:
        return filter_type, filtered

    # Try the average() filter.
    avg_res, avg_filter = average(curr, prev)
    avg_sum = np.sum(avg_res)
    if (avg_sum < smallest_sum):
        filter_type = avg_filter
        filtered = avg_res
        smallest_sum = avg_sum

    if not smallest_sum:
        return filter_type, filtered

    # Try the paeth() filter.
    paeth_res, paeth_filter = paeth(curr, prev)
    paeth_sum = np.sum(paeth_res)
    if (paeth_sum < smallest_sum):
        filter_type = paeth_filter
        filtered = paeth_res

    return filter_type, filtered


#################################### TESTS #####################################


def test_none():
    test_line = [1, 2, 3, 4]

    filtered, filter_type = none(test_line)

    assert filtered == test_line
    assert filter_type == 0


def test_sub_simple():
    filtered, filter_type = sub([1, 2, 3, 4])
    assert filtered == [1, 1, 1, 1]
    assert filter_type == 1


def test_sub_modulo():
    filtered, filter_type = sub([255, 128, 71, 18])
    assert filtered == [255, 129, 199, 203]
    assert filter_type == 1


def test_up():
    filtered, filter_type = up([1, 2, 3, 4], [10, 9, 8, 7])
    assert filtered == [247, 249, 251, 253]
    assert filter_type == 2


def test_choose_filter_none():
    filter_type, filtered = choose_filter([0, 0, 0, 0], [255, 255, 255, 255])
    assert filtered == [0, 0, 0, 0]
    assert filter_type == 0


def test_choose_filter_sub():
    filter_type, filtered = choose_filter([1, 1, 1, 1], [255, 255, 255, 255])
    assert filtered == [1, 0, 0, 0]
    assert filter_type == 1


def test_choose_filter_up():
    filter_type, filtered = choose_filter([255, 255, 255], [255, 255, 255])
    assert filtered == [0, 0, 0]
    assert filter_type == 2


def test_choose_filter_average():
    filter_type, filtered = choose_filter([4, 10, 30], [8, 16, 50])
    assert filtered == [0, 0, 0]
    assert filter_type == 3
