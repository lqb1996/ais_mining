import os, sys, glob
import numpy as np
import pandas as pd

import configparser
import time
import datetime

from joblib import Parallel, delayed
from sklearn.metrics import f1_score, log_loss, classification_report
from sklearn.model_selection import StratifiedKFold

# import lightgbm as lgb
from DataLoader import *
from plot import *

proDir = os.path.split(os.path.realpath(__file__))[0]
configPath = os.path.join(proDir, "setting.cfg")
cf = configparser.ConfigParser()
cf.read(configPath)
csv_path = os.path.join(proDir, cf.get("path", "csv_path"))
csv_loader = CSVDataLoader(csv_path)
# plot(csv_loader.df_total['longitude'], csv_loader.df_total['latitude'])
# print(csv_loader.df_total)
