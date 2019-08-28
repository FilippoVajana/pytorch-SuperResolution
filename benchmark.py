from imports import *
from engine.tester import Tester
from utilities.utils import create_folder
from benchmarks import plot
import data.dataset as data
import pandas as pd


class BenchmarkConfig():
    def __init__(self):        
        # run options
        self.device = 'cpu'

        # test data folder path     
        self.data = ''

        # models paths
        self.models = {'name' : 'path'}

        # models train data
        self.train_data = {'name' : 'path'}

    def save(self, destination):
        path = os.path.join(destination, 'benchmark_config.json')
        with open(path, "w") as f:
            json_obj = jsonpickle.encode(self)
            f.write(json_obj)
        return path

    @staticmethod
    def load(path):
        with open(path, "r") as f:
            json_str = f.read()
            obj = jsonpickle.decode(json_str)
            return obj


class Benchmark():
    MODELS = {
        "srcnn" : SRCNN(),
        "edsr" : EDSR(),
        "bicubic" : Bicubic()
        }
        
    def __init__(self, configuration : BenchmarkConfig):             
        self.cfg = configuration 
        self.logger = logging.getLogger("benchmark")
        self.logger.setLevel(20)
        
    def __get_id(self):
        # get run id as a time string
        import datetime as dt
        time = dt.datetime.now()
        id = time.strftime("%d%m_%H%M") # Hour_Minute_Day_Month
        return id       
    

    def run(self, save=True):
        root = os.getcwd()

        # get run id        
        run_id = self.__get_id()
        self.logger.info(f"Run ID {run_id}")

        # prepare test images
        self.logger.info("Preparing test data")
        builder = data.DatasetBuilder()        
        test_ds = builder.build(self.cfg.data)
        test_dl = tdata.DataLoader(test_ds, batch_size=1, shuffle=False) 

        # prepare images for comparison
        comp_imgs = [(e,l) for idx, (e,l) in enumerate(test_dl) if idx < 4] 
         
        # load models
        self.logger.info("Loading pre-trained models")
        models = []
        for name, df in self.cfg.models.items() :
            if name == "bicubic":
                models.append(Bicubic())
                continue
            m = self.MODELS[name]
            m.load_state_dict(torch.load(df))
            models.append(m)
                
        # test phase
        results = dict()
        for model in models:
            self.logger.info(f"Testing {model.__class__.__name__}")
            tester = Tester(model, self.cfg.device)        
            test_df = tester.test(test_dl).as_dataframe()
            results[model.__class__.__name__] = test_df
        

        # save results
        if save == True:            
            run_dir = create_folder(os.path.join(root, 'benchmarks'), run_id)            
            
            # train performance plots
            self.logger.info("Plotting train performance metrics") 
            for name, df in self.cfg.train_data.items():
                df = pd.read_excel(os.path.abspath(df))
                fig = plot.plot_train_performance(df, False)
                fig.suptitle(f"{name.upper()}", fontsize=18)
                plt.savefig(os.path.join(run_dir, f"train_{name}"))
                # fig.show()

            # test performance plots
            self.logger.info("Plotting test performance metrics")
            for name, df in results.items():                
                fig = plot.plot_test_performance(df, False)
                fig.suptitle(f"{name.upper()}", fontsize=18)
                plt.savefig(os.path.join(run_dir, f"test_{name}"))
                # fig.show()

            # comparison images
            self.logger.info("Plotting comparison images")            
            for idx, (e,l) in enumerate(test_dl):
                if idx > 3 : break
                fig = plot.plot_models_comparison(e, l, models, False)
                fig.suptitle(f"Sample #{idx+1}", fontsize=18)
                plt.savefig(os.path.join(run_dir, f"compare_{idx}"))
                # fig.show()
                
            # TODO: save figures

        input("Press Enter to continue...")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Super Resolution with DNN") 
    parser.add_argument('-cfg', type=str, action='store', default='./configs/benchmark_cpu_rgb.json', help='Load configuration')
    parser.add_argument('-empty', action='store_true', help='Save empty benchmark configuration file')
    args = parser.parse_args()

    # set logging
    logging.basicConfig(format='[%(asctime)s]  %(levelname)s: %(message)s', datefmt='%I:%M:%S')
    logger = logging.getLogger("benchmark")
    logger.setLevel(20)    

    # working directory
    wd = os.getcwd()
    
    if args.empty : 
        logger.info("Saving empty configuration file")
        BenchmarkConfig().save(wd)
        exit()

    # check if config file exists
    cfg_path = os.path.join(wd, args.cfg)
            

    # load config
    cfg = BenchmarkConfig.load(cfg_path)

    # run benchmark
    bench = Benchmark(cfg)
    bench.run()