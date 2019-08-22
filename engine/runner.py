import logging
import configparser
import time
from imports import *
from data.dataset import *
from engine import trainer, tester
from benchmarks.utils import collect_result_imgs


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

    def build_report(self, model_name, psnr, ssim, tot_time):
        report = configparser.ConfigParser()
        report['INFO'] = {
            'model' :           model_name,
            'device' :          self.cfg.device,
            'epochs' :          self.cfg.epochs,
            'batch_size' :      self.cfg.batch_size,
            'train_data' :      len(self.train_dl.dataset),
            'validation_data' : len(self.validation_dl.dataset),
            'test_data' :       len(self.test_dl.dataset),
            'min_psnr' :        np.min(psnr),
            'max_psnr' :        np.max(psnr),
            'avg_psnr' :        np.median(psnr),
            'min_ssim' :        np.min(ssim),
            'max_ssim' :        np.max(ssim),
            'avg_ssim' :        np.median(ssim),
            'tot_time (min)' :        tot_time / 60
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

        # save test images
        # collect_result_imgs(model, self.test_dl, 5, False, True, output_dir)


        from benchmarks import plot
        ############
        # DEBUG
        # save comparison images
        # from benchmarks.plot import compare_models
        # images = []
        # for e,l in self.test_dl:
        #     if len(images) >= 3:
        #         break
        #     images.append((e,l))

        # compare_models(model, model, "model1", "model2", images)
        #############


        ############
        # DEBUG
        # save comparison images 
        count = 0  
        for e,l in self.test_dl:
            if count >= 1:
                break
            plot.compare_outputs(e, l, [model, model, model])
            count += 1
        #############




        # create and save run report
        with open(os.path.join(output_dir, "report.ini"), "w") as f:
            psnr = df_test.data['psnr']
            ssim = df_test.data['ssim']
            report = self.build_report(model.__class__.__name__, psnr, ssim, cpu_time)
            report.write(f)
        pass