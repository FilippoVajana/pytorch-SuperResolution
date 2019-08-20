import jsonpickle
import os
import argparse
import datetime as dt
from models import SRCNN, EDSR
from utilities.utils import create_folder
from engine.runner import Runner

class ModelsConfig():
    def __init__(self):
        self.models = ["srcnn"] # list of models to init
        self.device = "cpu"     # device for training
        self.epochs = 10        # train epochs
        self.batch_size = 5     # train batch size
        self.train_data = ""    # path to train data folder
        self.test_data = ""     # path to test data folder

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
    parser = argparse.ArgumentParser(description="Script to train and validate DNN models.")
    parser.add_argument('-cfg', type=str, action='store', default='modelsconfig.json', help='Load configuration file.')
    args = parser.parse_args()

    # load config
    config = ModelsConfig.load(args.cfg)

    # models
    MODELS = {
        "srcnn" : SRCNN.SRCNN(),
        "edsr" : EDSR.EDSR()
        }

    # create root dir
    root = dt.datetime.now().strftime("%d%m_%H%M") # Hour_Minute_Day_Month
    cwd = os.getcwd()
    root = create_folder(os.path.join(cwd, 'benchmarks', root))

    runner = Runner(config)
    for m in config.models:
        print("##########")
        print(str(m).upper())
        print("##########")
        
        # create output folder
        out_dir = create_folder(os.path.join(root, str(m)))
        
        # call runner
        runner.run(MODELS[m], out_dir)
