import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import pandas as pd
import numpy as np


df = np.load('/data/physionet/physionet_saits.npy')

print(df.head())
