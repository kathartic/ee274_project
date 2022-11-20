from core.data_block import DataBlock
from PIL import Image
import numpy as np
import os
from typing import Tuple
from utils.test_utils import are_blocks_equal


def read_image(img: Image) -> Tuple[int, int, DataBlock]:
    """Returns image as tuple of width, height, and channels as DataBlock.

    This function reads the entire image represented by `img` into memory and
    returns it as a `DataBlock`. Given in-memory representation, do NOT use this
    function to handle large images.

    The pixels will be returned as RGB or RGBA, concatenated.

    Args:
        img: image to open.
    Returns:
        file width, height, and bytes.
    """
    # Convert image to RGB or RGBA channel format.
    if (img.mode != "RGB" and img.mode != "RGBA"):
        img = img.convert("RGB")

    # Read in channel data.
    channels = []
    for i in range(len(img.getbands())):
        channels += list(img.getdata(i))

    return img.width, img.height, DataBlock(channels)


################################### TESTS ######################################


def test_read_rgb():
    test_img = Image.new('RGB', (3, 1))
    test_img.putdata([(1, 2, 3), (4, 5, 6), (7, 8, 9)])

    width, height, block = read_image(test_img)

    assert width == 3
    assert height == 1
    assert are_blocks_equal(DataBlock([1, 4, 7, 2, 5, 8, 3, 6, 9]), block)


def test_read_rgba():
    test_img = Image.fromarray(np.array([[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
    ]]).astype('uint8'), mode='RGBA')

    width, height, block = read_image(test_img)

    assert width == 3
    assert height == 1
    assert are_blocks_equal(DataBlock([1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]),
                            block)


def test_read_black_white():
    test_img = Image.fromarray(np.array([
        [1, 2],
        [3, 4],
        [5, 6],
    ]).astype('uint8'), mode='L')

    width, height, block = read_image(test_img)
    assert width == 2
    assert height == 3


def test_read_png():
    script_dir = os.path.dirname(__file__)
    test_image = Image.open(
        os.path.join(script_dir, '../test_data/kodim03.png'))

    width, height, block = read_image(test_image)

    assert width == 768
    assert height == 512
