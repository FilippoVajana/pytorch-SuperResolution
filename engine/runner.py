import logging
import configparser
import time
from imports import *
from data.dataset import *
from engine import trainer, tester



class Runner():
    def __init__(self, config):
        self.cfg = config

        # set logging
        logging.basicConfig(level=logging.WARNING, format='[%(asctime)s]  %(levelname)s: %(message)s', datefmt='%I:%M:%S')
        
        # prepare datasets and dataloader
        builder = DatasetBuilder()

        logging.warning("Initializing datasets.")
        train_ds , validation_ds = builder.build_splitted(self.cfg.train_data)
        test_ds = builder.build(self.cfg.test_data)

        logging.warning("Creating dataloaders.")
        self.train_dl = tdata.DataLoader(train_ds, self.cfg.batch_size, shuffle=True)        
        self.validation_dl = tdata.DataLoader(validation_ds, 1, shuffle=False) 
        self.test_dl = tdata.DataLoader(test_ds, batch_size=1, shuffle=False)

    def build_report(self, model_name, min_psnr, max_psnr, tot_time):
        report = configparser.ConfigParser()
        report['INFO'] = {
            'model' :           model_name,
            'device' :          self.cfg.device,
            'epochs' :          self.cfg.epochs,
            'batch_size' :      self.cfg.batch_size,
            'train_data' :      len(self.train_dl.dataset),
            'validation_data' : len(self.validation_dl.dataset),
            'test_data' :       len(self.test_dl.dataset),
            'min_psnr' :        min_psnr,
            'max_psnr' :        max_psnr,
            'tot_time' :        tot_time
        }

        return report

    def run(self, model, output_dir):
        t_start = time.time()

        # train phase
        logging.warning("Starting train phase.")
        m_trainer = trainer.Trainer(model, self.cfg.device)
        df_train = m_trainer.run(self.cfg.epochs, self.train_dl, self.validation_dl)

        # test phase
        logging.warning("Starting test phase.")
        m_tester = tester.Tester(model, self.cfg.device)
        df_test = m_tester.test(self.test_dl)

        # get execution cpu time
        cpu_time = time.time() - t_start

        # save log
        df_train.save(os.path.join(output_dir, f"{df_train.name}.xlsx"))
        df_test.save(os.path.join(output_dir, f"{df_test.name}.xlsx"))

        # save model params
        torch.save(m_trainer.best_model, os.path.join(output_dir, f"{model.__class__.__name__}.pt"))

        
        # create and save run report
        with open(os.path.join(output_dir, "report.ini"), "w") as f:
            psnr_min = min(df_test.data['psnr'])
            psnr_max = max(df_test.data['psnr'])
            report = self.build_report(model.__class__.__name__, psnr_min, psnr_max, cpu_time)
            report.write(f)
        pass