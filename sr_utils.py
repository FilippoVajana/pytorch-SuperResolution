from sr_imports import *

def resize_img_batch(source_dir, target_dir, img_num, example_size, mult_factor = 2):

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
                .resize((example_size, example_size))
            l = Image.open(os.path.join(source_dir, l_name))\
                .resize((example_size*mult_factor, example_size*mult_factor))

            print(e.format, e.size, e.mode)
            print(l.format, l.size, l.mode)

            e.save(os.path.join(target_dir, e_name))
            l.save(os.path.join(target_dir, l_name))

            # e.show()
            # l.show()
        except StopIteration:
            pass

if __name__ == "__main__":
    src = "data/train"
    tgt = "data/s_train"
    num = 10
    size = 128

    resize_img_batch(src, tgt, num, size, 2)