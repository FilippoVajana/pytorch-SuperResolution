from imports import *
from benchmark import BenchmarkConfig, Benchmark
from utilities import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Super Resolution with DNN") 
    parser.add_argument('-cfg', type=str, action='store', default='runconfig_cpu.json', help='Load configuration')
    parser.add_argument('-gpu', type=int, action='store', default=None, help='Use GPU number i')
    args = parser.parse_args()


    # set logging
    logging.basicConfig(level=logging.WARNING, format='[%(asctime)s]  %(levelname)s: %(message)s', datefmt='%I:%M:%S')  


    # working directory
    wd = os.getcwd()

    
    # check if config file exists
    cfg_path = wd
    if os.path.isfile(os.path.join(wd, args.cfg)) == False:
        # create empty config        
        cfg_path = BenchmarkConfig.save_empty(wd)
    else:
        cfg_path = os.path.join(wd, args.cfg)


    # load config
    cfg = BenchmarkConfig.load(cfg_path)

    # run benchmark
    bench = Benchmark(cfg)
    bench.run()