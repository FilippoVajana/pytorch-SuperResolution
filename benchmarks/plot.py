from imports import *
from utilities.utils import create_folder
from data.metrics import Metrics
import torchvision as torchvision

from matplotlib.gridspec import *
from skimage.io import imread

def _set_metrics(axis, target, prediction):
        psnr = Metrics().psnr(target, prediction)
        ssim = Metrics().ssim(target, prediction)
        s = "PSNR: {:2.2f}\nSSIM: {:2.2f}".format(psnr, ssim)
        axis.set_xlabel(s)

def compare_models(model1, model2, name1="", name2="", images=[]):
    figures = []    
    toPIL = torchvision.transforms.ToPILImage()    
        
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
        _set_metrics(ax2, label, out1)
        ax2.imshow(out1_i)

        ax3 = fig.add_subplot(gs[1,1])
        ax3.set_title(name2)
        _set_metrics(ax3, label, out2)
        ax3.imshow(out2_i)

        ax4 = fig.add_subplot(gs[0,1])
        ax4.set_title('Ground Truth')
        ax4.imshow(label_i)

        plt.show()

        return figures


def compare_outputs(source, target, models=[]):
    toPIL = torchvision.transforms.ToPILImage()

    # build grids
    fig = plt.figure(constrained_layout=False)
    gs = GridSpec(2, 1, figure=fig, hspace=.5)
    sub_gs1 = gs[0].subgridspec(1,2)
    sub_gs2 = gs[1].subgridspec(1,3)

    # get models outputs
    predictions = []
    for m in models:
        predictions.append(m(source))        

    # transform tensors to valid images
    predictions_i = [toPIL(p.squeeze()) for p in predictions]
    source_i = toPIL(source.squeeze())
    target_i = toPIL(target.squeeze())

    # compose first subgrid
    ax1 = fig.add_subplot(sub_gs1[0,0])
    ax1.set_title('Source')
    ax1.imshow(source_i)

    ax2 = fig.add_subplot(sub_gs1[0,1])
    ax2.set_title('Ground Truth')
    ax2.imshow(target_i)

    # compose second subgrid
    ax3 = fig.add_subplot(sub_gs2[0,0])
    ax3.set_title(models[0].__class__.__name__)
    out3 = models[0](source)
    _set_metrics(ax3, target, out3)
    ax3.imshow(toPIL(out3.squeeze()))

    ax4 = fig.add_subplot(sub_gs2[0,1])
    ax4.set_title(models[1].__class__.__name__)
    out4 = models[1](source)
    _set_metrics(ax4, target, out4)
    ax4.imshow(toPIL(out4.squeeze()))

    ax5 = fig.add_subplot(sub_gs2[0,2])
    ax5.set_title(models[2].__class__.__name__)
    out5 = models[2](source)
    _set_metrics(ax5, target, out5)
    ax5.imshow(toPIL(out5.squeeze()))

    plt.show()

    return fig