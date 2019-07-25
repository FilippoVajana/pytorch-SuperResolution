from imports import *
from utilities.utils import create_folder


def is_valid(path):
    if not os.path.isdir(path) : return False
    else : return True

def resize(data, destination, size):
    """
    Resizes images and saves them as grey scale PIL.
    
    Arguments:

        data {list} -- List of paths
        destination {str} -- Destination folder path
        size {int} -- Final image size
    """

    for d in data:
        Image.open(d)\
            .resize((size, size), Image.BICUBIC)\
                .convert('L')\
                    .save(os.path.join(destination, os.path.basename(d)))

def init_data(source_dir, dest_name, examples_num = 0, img_size = 0):
    # check source dir
    if is_valid(source_dir) == False:
        raise Exception("Invalid source directory")

    # set working dirs
    ex_dir = source_dir
    lab_dir = os.path.join(ex_dir, "labels")

    # create data folders
    dest_path = os.path.join(ex_dir, os.pardir)
    dest_ex = create_folder(dest_path, dest_name)
    dest_lab = create_folder(dest_ex, "labels")

    # get files
    img_regex = re.compile(r'.png')
    files = os.listdir(ex_dir)    
    examples = filter(img_regex.search, files)
    examples = list(map(lambda p: os.path.join(ex_dir, p), examples))
    examples.sort()
    
    files = os.listdir(lab_dir)
    labels = filter(img_regex.search, files)
    labels = list(map(lambda p: os.path.join(lab_dir, p), labels))
    labels.sort()
    
    # compute examples range
    file_num = min(examples_num, len(examples))
        
    # process examples
    resize(examples[:file_num], dest_ex, img_size)
    # process labels
    resize(labels[:file_num], dest_lab, img_size*2)

if __name__ == "__main__":    
    init_data("./data/div2k/train/2x_bicubic", "test_bicubic", 10, 128)
    pass