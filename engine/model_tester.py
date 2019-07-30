from imports import *
from engine.logger import Logger
from data.metrics import Metrics
from utilities.utils import show_results
from tqdm import tqdm


class Tester():
    def __init__(self, model):
        self.model = model.to("cpu")
        self.loss_fn = torch.nn.MSELoss()
        self.quality = Metrics()

        self.log = Logger()
        self.log.add_metric("test_loss")
        self.log.add_metric("psnr")

    def test(self, test_dataloader = None):
        """
        Tests the trained model.
        
        Keyword Arguments:
            test_dataloader {DataLoader} -- Test data (default: {None})
        """

        if test_dataloader == None :
            raise Exception("Invalid test dataloader.")

        self.model.eval()
        with torch.no_grad():
            for data in tqdm(test_dataloader):
                examples, targets = data

                # predict
                outputs = self.model(examples)

                # compute loss
                loss = self.loss_fn(outputs, targets)
                self.log.add_value("test_loss", loss)
                
                # compute psnr
                for e, l in zip(outputs, targets):
                    self.log.add_value("psnr", self.quality.psnr(e, l))


    def test_sample(self, test_dataloader = None, sample_count = 3):
        """
        Tests and visualizes results for a sample.
        
        Keyword Arguments:        
            test_dataloader {DataLoader} -- Test data (default: {None})            
            sample_count {int} -- Examples to visualize (default: {3})
        """

        if test_dataloader == None :
            raise Exception("Invalid test dataloader.")

        results = list()

        self.model.eval()
        with torch.no_grad():
            for idx, data in enumerate(test_dataloader):
                if idx >= sample_count:
                    break

                example, target = data

                # model output
                output = self.model(example)

                # original unscaled image
                toTensor = torchvision.transforms.ToTensor()
                original = toTensor(Image.open(test_dataloader.dataset.examples[idx]))

                # display results
                output = output.squeeze()
                target = target.squeeze()
                res_fig = show_results((original, output, target))
                results.append(res_fig)

        return results
                