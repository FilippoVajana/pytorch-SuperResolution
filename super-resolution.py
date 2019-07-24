import argparse
import os
from benchmark import BenchmarkConfig, Benchmark
from utilities import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Super Resolution with DNN")    
    parser.add_argument('-cfg', type=str, action='store', default='runconfig.json', help='Load configuration')
    parser.add_argument('-gpu', type=int, action='store', default=None, help='Use GPU number i')
    args = parser.parse_args()

    # working directory
    wd = os.getcwd()
    cfg_path = wd

    # check if config file exists
    if os.path.isfile(os.path.join(wd, args.cfg)) == False:
        # create empty config
        cfg_path = BenchmarkConfig.save_empty(wd)
    else:
        cfg_path = os.path.join(wd, args.cfg)

    # load config
    cfg = BenchmarkConfig.load(cfg_path)

    # fix config
    if args.gpu != None:
        device = utils.get_device(args.gpu)
        cfg.device = device
    print(cfg.device)

    # run benchmark
    bench = Benchmark(cfg)
    bench.run()