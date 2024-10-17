import sys
import pandas as pd
import json
from matplotlib import pyplot as plt
import numpy as np

from io import BytesIO
import json
import itertools
import math
import os
import tarfile

from scipy.ndimage import find_objects
from skimage.morphology import square, binary_erosion, binary_dilation
from skimage.morphology import remove_small_objects
import skimage as sk  # for sk.measure.label
import pandas as pd
import zipfile
import tifffile


def load_data_local(filepath):
    f = zipfile.ZipFile(filepath, 'r')
    file_bytes = f.read("cells.json")
    with BytesIO() as b:
        b.write(file_bytes)
        b.seek(0)
        cells = json.load(b)
    file_bytes = f.read("divisions.json")
    with BytesIO() as b:
        b.write(file_bytes)
        b.seek(0)
        divisions = json.load(b)
    file_bytes = f.read("X.ome.tiff")
    with BytesIO() as b:
        b.write(file_bytes)
        b.seek(0)
        X = sk.io.imread(b, plugin="tifffile")
    file_bytes = f.read("y.ome.tiff")
    with BytesIO() as b:
        b.write(file_bytes)
        b.seek(0)
        y = sk.io.imread(b, plugin="tifffile")
    dcl_ob = {
        'X': np.expand_dims(X,3),
        'y': np.expand_dims(y,3),
        'divisions':divisions,
        'cells': cells}
    return dcl_ob
    