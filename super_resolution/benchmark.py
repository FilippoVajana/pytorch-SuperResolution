from models.SRCNN import SRCNN
from engine import model_trainer

from utilities.utils import create_folder
import data.dataset as data
import jsonpickle

class BenchmarkConfig(object):
    JSON_NAME = "runconfig.json"

    def __init__(self):
        # options
        self.device = 'cpu'

        # data
        self.use_small_dataset = False
        self.train_data = ''
        self.validation_data = ''
        self.test_data = ''

        # train options
        self.load_model = False    
        self.batch_size = 0
        self.epochs = 0

        # result options
        self.save_output = False
    
    def save(self, destination):
        """
        Saves the current configuration inside a JSON file.        
        Args:
        --------
            destination (string): The destination path.        
        Returns:
        --------
            string: The absolute path of the json file.
        """
        path = os.path.join(destination, BenchmarkConfig.JSON_NAME)
        with open(path, "w") as f:
            json_obj = jsonpickle.encode(self)
            f.write(json_obj)
        return path

    @staticmethod
    def save_empty(destination):
        cfg = BenchmarkConfig()
        cfg_path = cfg.save(destination)
        return cfg_path

    @staticmethod
    def load(path):
        with open(path, "r") as f:
            json_str = f.read()
            obj = jsonpickle.decode(json_str)
            return obj


class Benchmark():
    def __init__(self, configuration=None):
        # set logging
        logging.basicConfig(level=logging.DEBUG)     
        
        # set configuration
        if configuration is None : 
            logging.warning("Running with default configuration.")
            self.cfg = BenchmarkConfig()
        else: 
            self.cfg = configuration 
        
    def __get_id(self):
        # get run id as a time string
        import datetime as dt
        time = dt.datetime.now()
        id = time.strftime("%H_%M_%d_%m") # Hour_Minute_Day_Month
        return id       
    
    def run(self):
        root = os.getcwd()

        # get run id
        logging.info("Get run ID.")
        run_id = self.__get_id()

        # init run folders     
        logging.info("Creating run directory.")   
        run_dir = create_folder(root, run_id)

        # prepare data
        # 1. preprocessing --> list of functions
        # 2. init datasets
        # TODO: check small-dataset opt
        logging.info("Creating train set.")
        train_ds = data.SRDataset(self.cfg.train_data)

        # TODO: check and create
        logging.info("Creating validation set.")
        validation_ds = data.SRDataset(self.cfg.validation_data)

        # logging.info("Creating test set.")
        # test_ds = data.SRDataset(self.cfg.test_data)

        # 3. init dataloaders
        logging.info("Creating train dataloader.")
        train_dl = tdata.DataLoader(train_ds, self.cfg.batch_size, shuffle=True)

        logging.info("Creating validation dataloader.")
        validation_dl = tdata.DataLoader(validation_ds, self.cfg.batch_size, shuffle=False)

        # logging.info("Creating test dataloader.")
        # test_dl = tdata.DataLoader(test_ds, batch_size=1, shuffle=False)
        
        # init model runner
        # 1. metrics callbacks
        # 2. execution observer
        

        # run model
        # 1. train + validation        
        # 2. test
        logging.info("Initializing SRCNN model.")
        model = SRCNN.SRCNN()
        logging.info("Initializing model trainer.")
        trainer = model_trainer.Trainer(model, self.cfg.device)

        logging.info("Starting train phase.")
        trainer.run(self.cfg.epochs, train_dl, validation_dl)

        # get results (save + visualize)
        logging.debug(trainer.train_log)
        logging.debug(trainer.validation_log)
        
        # clean tmp data


if __name__ == "__main__":
    # mock config
    config = BenchmarkConfig()
    config_path = config.save(os.getcwd())

    # mock run
    b = Benchmark(config_path)
    b.run()
