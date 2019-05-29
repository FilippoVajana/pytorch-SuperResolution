from sr_imports import *

def resize_img_batch(source_dir, target_dir, img_num, example_size, mult_factor = 2):
    print("Building train data")

    files = os.listdir(source_dir)

    # get examples path
    example_re = re.compile(r'\d*x\d*.png')
    examples = filter(example_re.match, files)

    # get labels path
    label_re = re.compile(r'\d{4}.png')
    labels = filter(label_re.match, files)

    # load and resize images
    for i in range(img_num):
        try:
            e_name = next(examples)
            l_name = next(labels)

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
    Show model output compared with source and label images.\\    
    Args:
        res (original, output, label): A tuple of three Pytorch Tensor images.
    """
    cols = 3
    rows = int(len(res) / cols)
    toPil = tvision.transforms.ToPILImage()
    origin, output, label = res

    fig = plt.figure()
    
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
    