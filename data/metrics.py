from imports import *
from skimage.measure import compare_psnr


class Metrics():
    @staticmethod
    def mse(target, prediction):
        mse_3 = F.mse_loss(prediction, target)
        return mse_3

    @staticmethod
    def psnr(target, prediction):
        t = target.data.numpy()
        p = prediction.data.numpy()
        return compare_psnr(t, p, 1)
