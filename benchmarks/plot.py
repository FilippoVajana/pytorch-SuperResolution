import torchvision as torchvision
from imports import *
from data.metrics import Metrics
from matplotlib.gridspec import *
from skimage.io import imread

plt.style.use('tableau-colorblind10')


def _set_metrics(axis, target, prediction):
        psnr = Metrics().psnr(target.squeeze(), prediction.squeeze())
        ssim = Metrics().ssim(target.squeeze(), prediction.squeeze())
        s = "PSNR: {:2.2f}\nSSIM: {:2.2f}".format(psnr, ssim)
        axis.set_xlabel(s)

def plot_model_perf(model, source, target, show=False):
    toPIL = torchvision.transforms.ToPILImage()    
    
    # build plot figure
    w,h = plt.figaspect(1/3)    
    fig = plt.figure(figsize=(w,h))
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
    w,h = plt.figaspect(1/4)    
    fig = plt.figure(figsize=(w,h))
    gs = GridSpec(1, 4, figure=fig)

    # get models outputs
    predictions = []
    for model in models:
        predictions.append(model(source))        

    # transform tensors to valid images
    predictions_i = [toPIL(p.squeeze()) for p in predictions]
    source_i = toPIL(source.squeeze())
    target_i = toPIL(target.squeeze())

    # compose first subgrid
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title('Source')    
    ax1.imshow(source_i, cmap='gray')

    ax2 = fig.add_subplot(gs[3])
    ax2.set_title('Ground Truth')    
    ax2.imshow(target_i, cmap='gray')

    # compose second subgrid
    ax3 = fig.add_subplot(gs[1])
    ax3.set_title(models[0].__class__.__name__)    
    out3 = models[0](source)
    _set_metrics(ax3, target, out3)
    ax3.imshow(toPIL(out3.squeeze()), cmap='gray')

    ax4 = fig.add_subplot(gs[2])
    ax4.set_title(models[1].__class__.__name__)    
    out4 = models[1](source)
    _set_metrics(ax4, target, out4)
    ax4.imshow(toPIL(out4.squeeze()), cmap='gray')

    if show : plt.show()

    return fig

def plot_train_performance(dataframe, show=False):
    # build grid
    w,h = plt.figaspect(1/3)    
    fig = plt.figure(figsize=(w,h))
    gs = GridSpec(1,3, figure=fig)

    # plot loss
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title("Loss")
    dataframe[['t_loss', 'v_loss']].plot(ax=ax1, legend=None)
    ax1.set_xlabel("epoch") 

    # plot psnr
    ax2 = fig.add_subplot(gs[1])
    ax2.set_title("PSNR")    
    dataframe[['t_psnr', 'v_psnr']].plot(ax=ax2, legend=None)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("dB") 

    # plot ssim
    ax3 = fig.add_subplot(gs[2])
    ax3.set_title("SSIM")
    dataframe[['t_ssim', 'v_ssim']].plot(ax=ax3)
    ax3.set_xlabel("epoch") 
    ax3.legend(['train', 'validation'], loc='center right')

    if show : plt.show()

    return fig

def plot_test_performance(dataframe, show=False):
    # build grid
    w,h = plt.figaspect(1/3)    
    fig = plt.figure(figsize=(w,h), constrained_layout=True)
    gs = GridSpec(1,3, figure=fig)

    # plot loss
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title("Loss")
    ax1 = dataframe[['loss']].plot(ax=ax1, legend=None)
    ax1.set_xlabel("image")    

    # plot psnr
    ax2 = fig.add_subplot(gs[1])
    ax2.set_title("PSNR")
    ax2 = dataframe[['psnr']].plot(ax=ax2, legend=None)
    ax2.set_xlabel("image") 
    ax2.set_ylabel("dB")

    # plot ssim
    ax3 = fig.add_subplot(gs[2])
    ax3.set_title("SSIM")
    ax3 = dataframe[['ssim']].plot(ax=ax3, legend=None)
    ax3.set_xlabel("image") 

    # plot inference time
    # ax4 = fig.add_subplot(gs[3])
    # ax4.set_title("Inference Time")
    # ax4 = dataframe[['inference_time']].plot(ax=ax4, legend=None)        
    # ax4.set_xlabel("image") 
    # ax4.set_ylabel("seconds")

    if show : plt.show()

    return fig