"""
This file contains all the necessary packages that are required for the deep learning and data science. 
"""
import os
import sys
import unittest
from unittest import TestCase
from typing import Sequence, List, Tuple, Dict, Any, Union, Optional, Callable

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset


import torchvision
from torchvision import datasets, transforms, models

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image

from loguru import logger
from objprint import objprint