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


# None: Returns the same scanline (filter type 0)
def none(scanline: list) -> tuple[list, int]: 
    filter_type = 0
    return scanline, filter_type


# Sub: Each pixel value is replaced with the difference between it and the value to the left (filter type 1) 
def sub(scanline:list) -> tuple[list, int]: 
    filter_type = 1
    length = len(scanline)
    sub = [int] * length

    for i in range(length): 
        if (i == 0): 
            sub[0] = scanline[0]
        else: 
            sub[i] = (scanline[i] - scanline[i-1]) % 256
    
    return sub, filter_type


# Up: Each pixel value is replaced with the difference between it and the pixel above it (filter type 2)
def up(curr_scanline:list, prev_scanline:list) -> tuple[list, int]: 
    filter_type = 2 
    length = len(curr_scanline)
    up = [int] * length

    for i in range(0, length): 
        up[i] = (curr_scanline[i] - prev_scanline[i]) % 256

    return up, filter_type


# Average: Each pixel value is replaced the difference between it and 
# the average of the corresponding pixels to its left and above it, truncating any fractional part (filter type 3)
def average(curr_scanline:list, prev_scanline:list) -> tuple[list, int]: 
    filter_type = 3
    length = len(curr_scanline)
    average = [int] * length

    for i in range(0, length): 
        if (i == 0): 
            average[0] = int((curr_scanline[0] - np.floor(prev_scanline[0] / 2)) % 256)
        else: 
            average[i] = int((curr_scanline[i] - np.floor((prev_scanline[i] + curr_scanline[i-1]) / 2)) % 256)
    
    return average, filter_type


# TODO
# Paeth: Operates on current pixel and left, above, upper left using Paeth operator (filter type 4)
def paeth(curr_scanline:list, prev_scanline:list) -> tuple[list, int]: 
    filter_type = 4
    length = len(curr_scanline)
    paeth = [int] * length 

    for i in range(0, length): 
        if(i == 0): 
            print("do something here")
        else: 
            print("do something else here")

    raise NotImplemented
    return paeth, filter_type
