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
        try:
            self.data[key].append(value.item())
        except AttributeError:
            self.data[key].append(value)
        
        
    
    def save(self, path):
        df = self.as_dataframe()
        df.to_excel(path, engine='xlsxwriter')

    def as_dataframe(self):
        return pd.DataFrame.from_dict(self.data)
        
    