import torch as torch


class Logger():
    def __init__(self):
        self.data = {}


    def is_valid_key(self, key):
        """
        Checks if the key is valid.
        
        Arguments:

            key {str} -- Key to check
        
        Returns:

            bool -- True if the key is valid
        """

        if key in self.data:
            return True
        else:
            return False


    def add_metric(self, name):
        """
        Adds a new metric to the logger.
        """

        if self.is_valid_key(name):
            raise Exception("Already existing metric.")  

        self.data[name] = list()


    def add_value(self, name, value):
        """
        Append the value to the specified metric.
        """

        if self.is_valid_key(name) == False:
            raise Exception("Invalid metric.")
        
        self.data[name].append(value)


    def add_batch(self, name, batch):        
        def batch_process():
            epoch, data = batch
            return (epoch, torch.mean(torch.stack(data)))
        
        if self.is_valid_key(name) == False:
            raise Exception("Invalid metric.")

        res = batch_process()

        self.data[name].append(res)
            

    def get_value(self, name):
        """
        Gets the computed value of the specified metric.
        """
        if self.is_valid_key(name) == False:
            raise Exception("Invalid metric.")

        return self.data[name]