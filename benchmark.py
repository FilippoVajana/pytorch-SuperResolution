from sr_imports import *
import sr_utils as util
import data.dataset as data
import jsonpickle

class BenchmarkConfig(object):
    JSON_NAME = "runconfig.json"

    def __init__(self):
        # Data
        self.train_data = ""
        self.validation_data = ""
        self.test_data = ""

        # Train loop        
        self.batch_size = 0
        self.epochs = 0

        # Result
        self.save_output = False
    
    def save(self, destination):
        path = os.path.join(destination, BenchmarkConfig.JSON_NAME)
        with open(path, "w") as f:
            json_obj = jsonpickle.encode(self)
            f.write(json_obj)
        return path

    def load(path):
        with open(path, "r") as f:
            json_str = f.read()
            obj = jsonpickle.decode(json_str)
            return obj


class Benchmark():
    def __init__(self, config_path=None):
        # set configuration
        if config_path is None : 
            print("Initialize empty run configuration")
            self.configuration = BenchmarkConfig()
        else: 
            self.configuration = BenchmarkConfig.load(config_path)        
        
    def __get_id(self):
        # get run id as a time string
        import datetime as dt
        time = dt.datetime.now()
        id = time.strftime("%H_%M_%d_%m") # Hour_Minute_Day_Month
        return id        


    def run(self):
        root = os.getcwd()
        rf_path = root

        # get run id
        run_id = self.__get_id()

        # init run folders
        run_dir = util.create_folder(root, run_id)        
        tmp_dir = util.create_folder(run_dir, "tmp")

        # prepare data
        # 1. preprocessing --> list of functions
        # 2. init datasets
        train_ds = data.SRDataset(self.configuration.train_data)
        validation_ds = data.SRDataset(self.configuration.validation_data)
        test_ds = data.SRDataset(self.configuration.test_data)

        # 3. init dataloaders
        train_dl = tdata.DataLoader(train_ds, self.configuration.batch_size, shuffle=True)
        validation_dl = tdata.DataLoader(validation_ds, self.configuration.batch_size, shuffle=False)
        test_dl = tdata.DataLoader(test_ds, batch_size=1, shuffle=False)
        
        # init model runner
        # 1. metrics callbacks
        # 2. execution observer

        # run model
        # 1. train + validation
        # 2. test

        # get results (save + visualize)

        # clean tmp data

if __name__ == "__main__":
    # mock config
    config = BenchmarkConfig()
    config_path = config.save(os.getcwd())

    # mock run
    b = Benchmark(config_path)
    b.run()
