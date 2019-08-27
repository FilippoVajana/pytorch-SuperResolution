from imports import *
from engine.logger import Logger
from data.metrics import Metrics
from tqdm import tqdm


class Tester():
    def __init__(self, model, device):
        self.device = device
        self.model = model.to(device)
        self.loss_fn = torch.nn.MSELoss()
        self.log = Logger("test_log", ["loss", "psnr", "ssim", "inference_time"])


    def test(self, test_dataloader : tdata.DataLoader):
        """
        Tests the model.
        """
        self.model.eval()
        with torch.no_grad():            
            for data in tqdm(test_dataloader):
                s_time = time.time()
                examples, targets = data

                # move data to device                
                examples = examples.to(self.device)
                targets = targets.to(self.device)

                # predict
                predictions = self.model(examples)
                f_time = time.time()

                # compute metrics
                loss = self.loss_fn(predictions, targets)
                psnr = torch.tensor([Metrics.psnr(targets[idx], p) for idx, p in enumerate(predictions)]).mean()
                ssim = torch.tensor([Metrics.ssim(targets[idx], p) for idx, p in enumerate(predictions)]).mean()

                # update test log
                self.log.add("loss", loss)
                self.log.add("psnr", psnr)
                self.log.add("ssim", ssim)
                self.log.add("inference_time", f_time - s_time)
                
        return self.log