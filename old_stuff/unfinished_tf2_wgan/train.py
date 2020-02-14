import glob
from tqdm import tqdm
import itertools
from time import time
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

from comet_ml import Experiment
import tensorflow as tf
print('Using tensorflow version', tf.__version__)