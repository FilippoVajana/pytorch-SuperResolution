from sr_imports import *
from torchvision import transforms

class SRDataset(tdata.Dataset):
    """Dataset specific for Super Resolution.
    
    Attributes:
        root_dir: images source directory.
        examples: paths for example images.
        labels : paths for label images.
    """

    def __init__(self, root_dir):
        self.root_dir = root_dir

        files = os.listdir(root_dir)

        # get examples path
        example_re = re.compile(r'\d*x\d*.png')
        self.examples = [os.path.join(root_dir, name) for name in list(filter(example_re.match, files))]

        # get labels path
        label_re = re.compile(r'\d{4}.png')
        self.labels = [os.path.join(root_dir, name) for name in list(filter(label_re.match, files))]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        """ 
        Returns:
            array: a 2d array of tensors [sample_t, label_t]
        """

        # load image (think about a cache system)
        e = Image.open(self.examples[index])
        l = Image.open(self.labels[index])

        # print(e.format, e.size, e.mode)
        # print(l.format, l.size, l.mode)

        # define data transformations
        example_t = tvision.transforms.Compose([transforms.ToTensor()])
        label_t = tvision.transforms.Compose([transforms.ToTensor()])

        # print(example_t(e).shape)
        # print(label_t(l).shape)

        return [example_t(e), label_t(l)]

    def split(self, train_frac = .8):
        l = [int(len(self) * train_frac), int(len(self) - len(self) * train_frac)]
        ds = tdata.random_split(self, l)
        


