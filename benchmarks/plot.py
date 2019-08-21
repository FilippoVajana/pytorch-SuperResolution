from imports import *
from utilities.utils import create_folder
from data.metrics import Metrics
import torchvision as torchvision

from matplotlib.gridspec import GridSpec
from skimage.io import imread


def compare_models(model1, model2, name1="", name2="", images=[]):
    figures = []    
    toPIL = torchvision.transforms.ToPILImage()

    def set_metrics(axis, target, prediction):
        psnr = Metrics().psnr(target, prediction)
        ssim = Metrics().ssim(target, prediction)
        s = "PSNR: {:2.2f}\nSSIM: {:2.2f}".format(psnr, ssim)
        axis.set_xlabel(s)
        
    for idx, (example, label) in enumerate(images):
        # build plot figure
        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(2, 2, figure=fig)
            
        # get models outputs
        out1 = model1(example)
        out2 = model2(example)

        # transform tensors to valid images
        example_i = toPIL(example.squeeze())
        label_i = toPIL(label.squeeze())
        out1_i = toPIL(out1.squeeze())
        out2_i = toPIL(out2.squeeze())

        # compose grid
        ax1 = fig.add_subplot(gs[0,0])
        ax1.set_title('Source')
        ax1.imshow(example_i)

        ax2 = fig.add_subplot(gs[1,0])
        ax2.set_title(name1)
        set_metrics(ax2, label, out1)
        ax2.imshow(out1_i)

        ax3 = fig.add_subplot(gs[1,1])
        ax3.set_title(name2)
        set_metrics(ax3, label, out2)
        ax3.imshow(out2_i)

        ax4 = fig.add_subplot(gs[0,1])
        ax4.set_title('Ground Truth')
        ax4.imshow(label_i)

        plt.show()

        return figures
