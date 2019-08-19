import torch 
import torch.nn as nn
import torchvision.datasets as dsets
from torch.autograd import Variable
import numpy as np;
from torch.utils.data import Dataset, DataLoader
import math;
import torch.optim as optim
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('grayscale')
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms
import time
#print("PyTorch Version: ",torch.__version__)
#print("Torchvision Version: ",torchvision.__version__)
import torchvision.models as models
from torchsummary import summary
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import PIL
from PIL import Image
import scipy
import scipy.io
import torch.utils.data as utils
from itertools import cycle,chain,repeat
import itertools
import tensorflow as tf

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")