import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime
from os import listdir
import time

class Iforest:
    def __init__(self,df,nodelists,metrics):
        self.df = df
        self.nodelists = nodelists
        self.metrics = metrics

