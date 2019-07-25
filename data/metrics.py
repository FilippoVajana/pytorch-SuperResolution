from imports import *


class Metrics():

    def mse(self, example, prediction):
        mse_3 = F.mse_loss(prediction, example)
        return mse_3


    def psnr(self, example, prediction):
        max_i = torch.tensor(1).float()
        m = mse(example, prediction)
        psnr = (20 * torch.log10(max_i)) - (10 * torch.log10(m))
        return psnr
