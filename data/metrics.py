from imports import *


class Metrics():

    def mse(self, example, prediction):
        mse_3 = F.mse_loss(prediction, example)
        return mse_3


    def psnr(self, example, prediction):
        max_i = torch.Tensor(1).float()
        m = self.mse(example, prediction)
        psnr = (20 * max_i.log10()) - (10 * m.log10())
        return psnr
