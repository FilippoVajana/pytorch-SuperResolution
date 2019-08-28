from imports import *
from utilities.utils import create_folder
from tqdm import tqdm


def is_valid(path):
    if not os.path.isdir(path) : return False
    else : return True

def resize(data, destination, size, rgb):
    """
    Resizes images and saves them as grey scale PIL.
    
    Arguments:

        data {list} -- List of paths
        destination {str} -- Destination folder path
        size {int} -- Final image size
        rgb {bool} -- If False converts to grey scale
    """

    for d in tqdm(data):
        with Image.open(d) as im:            
            im = im.resize((size, size), Image.BICUBIC)
            if rgb == False : im = im.convert('L')        
            im.save(os.path.join(destination, os.path.basename(d)))

def init_data(source_dir, dest_name, examples_num = 0, img_size = 0, rgb = False, img_mult = 2):
    # init logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()     

    # check source dir
    if is_valid(source_dir) == False:
        logger.critical("Invalid source directory.")
        raise Exception("Invalid source directory.")

    # set working dirs
    ex_dir = source_dir
    lab_dir = os.path.join(ex_dir, "labels")

    # get files
    img_regex = re.compile(r'.png')
    logger.info("Getting examples images.") 
    files = os.listdir(ex_dir)       
    examples = filter(img_regex.search, files)
    examples = list(map(lambda p: os.path.join(ex_dir, p), examples))
    examples.sort()
    
    logger.info("Getting labels images.") 
    files = os.listdir(lab_dir)
    labels = filter(img_regex.search, files)
    labels = list(map(lambda p: os.path.join(lab_dir, p), labels))
    labels.sort()
    
    # compute examples range
    if examples_num == 0:
        examples_num = len(examples)
    file_num = min(examples_num, len(examples))

    # create data folders
    dest_path = os.path.join(ex_dir, os.pardir, os.pardir)
    logger.info("Creating examples destination folder.")
    dest_ex = create_folder(dest_path, dest_name)
    logger.info("Creating labels destination folder.")
    dest_lab = create_folder(dest_ex, "labels")
        
    # process examples
    logger.info("Resizing examples.")
    tqdm.write("Resizing examples...")
    resize(examples[:file_num], dest_ex, img_size, rgb)

    # process labels
    logger.info("Resizing labels.")
    tqdm.write("Resizing labels...")
    resize(labels[:file_num], dest_lab, img_size*img_mult, rgb)



if __name__ == "__main__":   
    parser = argparse.ArgumentParser(description="Script to prepare the dataset for the SR instance.")
    parser.add_argument('-dn', help='New dataset folder name', type=str, action='store', default='2x_bicubic_32x32_x')
    parser.add_argument('-dl', help='Number of examples, 0=ALL', type=int, action='store', default=0)
    parser.add_argument('-src', help='Data source', type=str, action='store', default='./data/div2k/train/2x_bicubic')
    parser.add_argument('-s', help='Example image size', type=int, action='store', default=32)
    parser.add_argument('-rgb', help='RGB images', action='store_true', default=False)
    parser.add_argument('-im', help='Label image size multiplier', type=int, action='store', default=2)
    args = parser.parse_args()

    ds_name = args.dn
    ds_len = args.dl
    ds_src = args.src
    im_size = args.s
    im_rgb = args.rgb
    im_mult = args.im

    init_data(ds_src, ds_name, ds_len, im_size, im_rgb, im_mult)