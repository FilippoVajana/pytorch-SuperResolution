from sr_imports import *

class Logger():
    def __init__(self):
        self.metrics = {}
        self.metrics_data = {}
        

    def attach(self, metric_name="", metric_function=lambda x: x):
        """
        Adds a new metric.
        """

        if self.metrics.__contains__(metric_name):
            raise Exception("Already existing metric.")  

        # add metric
        self.metrics[metric_name] = metric_function
        
        # create metric state
        self.metrics_data[metric_name] = list()


    def update(self, metric_name, value):
        if self.metrics.__contains__(metric_name) == False:
            raise Exception("Invalid metric.")
        
        self.metrics_data[metric_name].append(value)
            

    def get(self, metric_name):
        if self.metrics.__contains__(metric_name) == False:
            raise Exception("Invalid metric.")

        data = self.metrics_data[metric_name]
        func = self.metrics[metric_name]
        res = func(data)
        return res