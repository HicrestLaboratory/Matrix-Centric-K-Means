
import pandas as pd
import numpy as np

import subprocess
import time
import os

from collections import OrderedDict



class Trial:

    features = []

    def __init__(self):
        self.df = pd.DataFrame(columns=self.features)
    
    def run_trial(self, cmd, n_iters):

        result = self.run_command(cmd)
        input_dict = self.parse_output(result, n_iters)
        self.add_sample(input_dict)

    
    def add_sample(self, input_dict):

        sample_dict = OrderedDict()

        for f in self.features:
            if f in input_dict.keys():
                sample_dict[f] = input_dict[f]
            else:
                sample_dict[f] = None
        
        
        nrows = self.df.shape[0]
        self.df.loc[nrows] = list(sample_dict.values())


    def run_command(self, cmd):
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        result.check_returncode()
        return result

    
    def save(self, fname):
        self.df.to_csv(fname+".csv")
    

    def parse_output(self, result, n_iters):
        return # should be overridden by subclasses
                


