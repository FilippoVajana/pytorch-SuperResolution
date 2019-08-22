# TODO: remove all

from imports import *
from utilities.utils import create_folder
from data.metrics import Metrics
import torchvision as torchvision


# Result images
def create_result_img(original, output, target):
    cols = 3
    rows = 1
    toPil = torchvision.transforms.ToPILImage()

    fig = plt.figure()
    fig.set_size_inches(30, 10)    

    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.set_title(f"Train image {original.shape}")
    plt.imshow(toPil(original))

    ax1 = fig.add_subplot(rows, cols, 2)    
    ax1.set_title(f"Model output (PSNR={Metrics().psnr(output, target)}, SSIM={Metrics().ssim(output, target)})")
    plt.imshow(toPil(output))

    ax1 = fig.add_subplot(rows, cols, 3)
    ax1.set_title(f"Target image {target.shape}")
    plt.imshow(toPil(target))

    return fig

def save_result_img(fig, path, idx):
    fig.savefig(os.path.join(path, f"output_img{idx}"), dpi=250)

def display_result_img(fig):
    fig.show()

def collect_result_imgs(model, dataloader = None, sample_count = 5, display = False, save = True, save_path = "./"):
    """
    Collects test results.
    
    Arguments:
        model {SRCNN} -- The model
    
    Keyword Arguments:
        dataloader {DataLoader} -- Test data (default: {None})
        sample_count {int} -- Number of samples (default: {5})
        display {bool} -- Show result images (default: {False})
        save {bool} -- Save result images (default: {True})
        save_path {str} -- Save path (default: {"./"})
    
    Raises:
        Exception: For invalid data loader
    """

    if dataloader == None :
        raise Exception("Invalid test dataloader.")

    results = list()

    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if idx >= sample_count:
                break

            example, target = data

            # model output
            output = model(example)

            # original unscaled image
            toTensor = torchvision.transforms.ToTensor()
            original = toTensor(Image.open(dataloader.dataset.examples[idx]))

            # create result image
            logging.info("Creating result image.")
            output = output.squeeze()
            target = target.squeeze()
            res = create_result_img(original, output, target)
            results.append(res)

            if display:
                display_result_img(res)

    if save:
        logging.info(f"Saving {len(results)} result images.")
        path = create_folder(save_path, "images")
        for idx, fig in enumerate(results):
            save_result_img(fig, path, idx)