'''
Code borrowed from https://github.com/FrancescoSaverioZuppichini/ResNet/blob/master/ResNet.ipynb
'''
import torch
import torch.nn as nn

from functools import partial
from dataclasses import dataclass
from collections import OrderedDict