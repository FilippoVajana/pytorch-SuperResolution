from imports import *
from engine import trainer, tester
from utilities.utils import create_folder
from benchmarks.utils import collect_result_imgs
import data.dataset as data


class BenchmarkConfig(object):
    JSON_NAME = "runconfig.json"

    def __init__(self):        
        # options
        self.device = 'cpu'

        # data
        self.train_data = ''        
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
        id = time.strftime("%d%m_%H%M") # Hour_Minute_Day_Month
        return id       
    
    def run(self):
        root = os.getcwd()

        # get run id
        logging.info("Get run ID.")
        run_id = self.__get_id()

        # prepare datasets and dataloader
        builder = data.DatasetBuilder()      

        logging.info("Creating train and validation sets.")
        train_ds , validation_ds = builder.build_splitted(self.cfg.train_data)

        logging.info("Creating train dataloader.")
        train_dl = tdata.DataLoader(train_ds, self.cfg.batch_size, shuffle=True)

        logging.info("Creating validation dataloader.")
        validation_dl = tdata.DataLoader(validation_ds, 1, shuffle=False)
 
        
        # init model
        # logging.info("Initializing SRCNN model.")
        # model = SRCNN()
        logging.info("Initializing EDSR model.")
        model = EDSR()

        # TRAIN PHASE
        #############
        logging.info("Initializing model trainer.")
        m_trainer = trainer.Trainer(model, self.cfg.device)

        logging.info("Starting train phase.")
        df_train = m_trainer.run(self.cfg.epochs, train_dl, validation_dl)

        
        # TEST PHASE
        #############
        test_ds = None
        test_dl = None     
        if os.path.isdir(self.cfg.test_data):
            logging.info("Creating test set.")
            test_ds = builder.build(self.cfg.test_data)

            logging.info("Creating test dataloader.")
            test_dl = tdata.DataLoader(test_ds, batch_size=1, shuffle=False)

        logging.info("Initializing model tester.")
        m_tester = tester.Tester(model, self.cfg.device)        
        df_test = m_tester.test(test_dl)

        

        # save results
        if self.cfg.save_output == True:
            # init run folders     
            logging.info("Creating run directory.")   
            run_dir = create_folder(os.path.join(root, 'benchmarks'), run_id)

            # save model
            torch.save(model.state_dict(), os.path.join(run_dir, 'model.pt'))

            # save log
            df_train.save(os.path.join(run_dir, f"{df_train.name}.xlsx"))
            df_test.save(os.path.join(run_dir, f"{df_test.name}.xlsx"))

            # save test imgs sample            
            collect_result_imgs(model, test_dl, save_path=run_dir)