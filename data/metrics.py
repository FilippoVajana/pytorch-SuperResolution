from imports import *
from skimage.measure import compare_psnr, compare_ssim


class Metrics():
    @staticmethod
    def mse(target, prediction):
        mse_3 = F.mse_loss(prediction, target)
        return mse_3

    @staticmethod
    def psnr(target, prediction):
        t = target.cpu().squeeze().data.numpy()        
        p = prediction.cpu().squeeze().data.numpy()
        psnr = compare_psnr(t, p, data_range = np.max(p) - np.min(p))
        return psnr if psnr >= 0 else 0.0

    @staticmethod
    def ssim(target, prediction):
        t = target.cpu().permute(1,2,0).squeeze().data.numpy()
        p = prediction.cpu().permute(1,2,0).squeeze().data.numpy()
        rgb = True if len(p.shape) > 2 else False
        return compare_ssim(t, p, data_range = np.max(p) - np.min(p), multichannel=rgb)
