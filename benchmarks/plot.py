import torchvision as torchvision
from imports import *
from data.metrics import Metrics
from matplotlib.gridspec import *
from skimage.io import imread


def _set_metrics(axis, target, prediction):
        psnr = Metrics().psnr(target, prediction)
        ssim = Metrics().ssim(target, prediction)
        s = "PSNR: {:2.2f}\nSSIM: {:2.2f}".format(psnr, ssim)
        axis.set_xlabel(s)

def plot_model_perf(model, source, target, show=False):
    toPIL = torchvision.transforms.ToPILImage()    
    
    # build plot figure
    fig = plt.figure(constrained_layout=False)
    gs = GridSpec(1, 3, figure=fig)   

    # get model output
    out = model(source)  

    # transform tensors to valid images
    source_i = toPIL(source.squeeze())
    target_i = toPIL(target.squeeze())
    out_i = toPIL(out.squeeze())

    # compose grid
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title('Source')
    ax1.imshow(source_i)

    ax2 = fig.add_subplot(gs[1])
    ax2.set_title(model.__class__.__name__)
    _set_metrics(ax2, target, out)
    ax2.imshow(out_i)
    
    ax3 = fig.add_subplot(gs[2])
    ax3.set_title('Ground Truth')
    ax3.imshow(target_i)

    if show : plt.show()

    return fig

def plot_models_comparison(source, target, models=[], show=False):
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

    if show : plt.show()

    return fig