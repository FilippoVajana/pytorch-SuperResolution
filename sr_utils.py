from sr_imports import *

def resize_img_batch(source_dir, target_dir, img_num, example_size, mult_factor = 2):
    print("Building train data")

    files = os.listdir(source_dir)

    # get examples path
    example_re = re.compile(r'\d*x\d*.png')
    examples = list(filter(example_re.match, files))
    examples.sort()

    # get labels path
    label_re = re.compile(r'\d{4}.png')
    labels = list(filter(label_re.match, files))
    labels.sort()

    # check img number
    img_num = len(examples) if img_num < 0 else img_num

    # load and resize images
    for i in range(img_num):
        try:
            e_name = examples[i]
            l_name = labels[i]

            e = Image.open(os.path.join(source_dir, e_name))\
                .resize((example_size, example_size))\
                    .convert('L')

            l = Image.open(os.path.join(source_dir, l_name))\
                .resize((example_size*mult_factor, example_size*mult_factor))\
                    .convert('L')

            # print(e.format, e.size, e.mode)
            # print(l.format, l.size, l.mode)

            e.save(os.path.join(target_dir, e_name))
            l.save(os.path.join(target_dir, l_name))

            # e.show()
            # l.show()
        except StopIteration:
            pass

def show_results(res, display=True):
    """
    Show model output compared with source and label images.

    Parameters
    ----------
        res : (original, output, label)
              A tuple of three Pytorch Tensor images.
    """
    cols = 3
    rows = int(len(res) / cols)
    toPil = tvision.transforms.ToPILImage()
    origin, output, label = res

    fig = plt.figure()
    fig.set_size_inches(30, 10)    

    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.set_title("train image {}".format(origin.shape))
    plt.imshow(toPil(origin))

    ax1 = fig.add_subplot(rows, cols, 2)
    ax1.set_title("output image {}".format(output.shape))
    plt.imshow(toPil(output))

    ax1 = fig.add_subplot(rows, cols, 3)
    ax1.set_title("label image {}".format(label.shape))
    plt.imshow(toPil(label))

    if display == True : plt.show()

    return fig

def get_device(device=None):
    d = torch.device("cpu") if device == None else torch.device("cuda:{}".format(device))
    
    print("Selected Device: ", d)
    return d

def create_folder(root, name=None):
    if name != None:
        root = os.path.join(root, name)

    try:
        os.mkdir(root)
    except OSError:
        print("Creation of the directory %s failed" % root)
    else:
        print("Successfully created the directory %s " % root)
    
    return root


if __name__ == "__main__":
    size = 5
    a = torch.rand((3,size,size))
    b = torch.rand((3,size,size))

    mse(a,b)
    psnr(a,b)