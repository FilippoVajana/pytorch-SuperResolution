import jsonpickle
import os
import argparse

class ModelsConfig():
    def __init__(self):
        self.models = ["srcnn"] # list of models to init
        self.device = "cpu" # device for training
        self.epochs = 10 # train epochs
        self.batch_size = 5 # train batch size
        self.train_data = "" # path to train data folder
        self.test_data = "" # path to test data folder

    def save(self):
        name = "modelsconfig.json"
        path = os.path.join(os.getcwd(), name)
        with open(path, "w") as f:
            json = jsonpickle.encode(self)
            f.write(json)
        return path


    @staticmethod
    def load(path):
        with open(path, "r") as f:
            json_str = f.read()
            config = jsonpickle.decode(json_str)
            return config


if __name__ == "__main__":
    cfg = ModelsConfig()
    cfg.save()