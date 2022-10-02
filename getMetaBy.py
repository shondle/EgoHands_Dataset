import scipy.io as sio
import numpy as np

# just trying to get a sense of how to handle metadata.mat, so no function with parameters declared yet

# Selma added this
meta_contents = sio.loadmat('./metadata.mat')

## MATLAB can't pass nested struct objects through MATLAB ENGINE API to python, so, unless I missed something,
## we have to rewrite the getMetaBy function using python libraries.

## from below commented out code, we know that 'video' is one of the keys for this dict
print(meta_contents.keys())

## assigning a nested struct to a variable using the 'video' key
struct = meta_contents['video']

## from below, it is clear struct is a numpy.ndarray object, refer to https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html
print(type(struct))

## here, we can look at a segment of meta_contents' values
## it is clear this segment of the struct has nested structs
matrix = struct[0, 0]
# print(matrix)

## Below code proves the struct is mutable
# struct[0,0] = struct[0,1]
# print(struct[0,0])

# running `print(matrix)`, we are given dtypes of 'frame_num', 'my_left', 'my_right', 'yourleft', and 'yourright'

# I'm stumped for the next steps to parse metadata.mat correctly, however. When I run print(matrix[0][0]) it says there are
# "too many indices for array: array is 0-dimensional, but 2 were indexed." My question - Why is it 0-dimensional?

# update
# AHA!!! Only use one indexing number and it returns what input we should pass into getMetaBy. NICE!
print(matrix[2])
