import numpy as np
import pandas as pd


class Logger():
    def __init__(self, name="log", metrics=[]):
        self.name = name        
        # init data structure
        self.data = {}
        for m in metrics:
            self.data[m] = list()

    def add(self, key, value):
        self.data[key].append(value)

    def to_dataframe(self):
        df = pd.DataFrame.from_dict(self.data)
        return df
        
    