class Logger():
    def __init__(self, metrics=None):
        self.metrics = {}
        self.metrics_data = {}
        
        if metrics != None:
            for name, fn in metrics:
                self.attach_metric(name, fn)
        

    def attach_metric(self, metric_name="", metric_function=lambda x: x):
        """
        Adds a new metric.
        """

        if self.metrics.__contains__(metric_name):
            raise Exception("Already existing metric.")  

        # add metric
        self.metrics[metric_name] = metric_function
        
        # create metric state
        self.metrics_data[metric_name] = list()


    def add_value(self, metric_name, value):
        """
        Append metric value.
        """
        if self.metrics.__contains__(metric_name) == False:
            raise Exception("Invalid metric.")
        
        self.metrics_data[metric_name].append(value)
            

    def get_value(self, metric_name):
        """
        Gets the computed value of the specified metric.
        """
        if self.metrics.__contains__(metric_name) == False:
            raise Exception("Invalid metric.")

        data = self.metrics_data[metric_name]
        func = self.metrics[metric_name]
        res = func(data)
        return res