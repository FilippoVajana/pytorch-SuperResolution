from sr_imports import *
from torchvision import transforms
from cachetools import cached, cachedmethod, LRUCache
import operator

class SRDataset(tdata.Dataset):
    """Dataset specific for Super Resolution.
    
    Attributes:
        root_dir: images source directory.
        examples: paths for example images.
        labels : paths for label images.
    """

    def __init__(self, train_dir):
        self.train_dir = train_dir
        self.labels_dir = os.path.join(self.train_dir, "labels")

        self.examples_p = [os.path.join(self.train_dir, name) for name in os.listdir(self.train_dir)] 
        self.examples_p.sort()
        self.labels_p = [os.path.join(self.labels_dir, name) for name in os.listdir(self.labels_dir)] 
        self.labels_p.sort()

        self.cache = LRUCache(len(self.examples_p))

    def __len__(self):
        return len(self.examples_p)

    @cachedmethod(operator.attrgetter('cache'))
    def __getitem__(self, index):
        """ 
        Returns:
            array: a 2d array of tensors [sample_t, label_t]
        """
       
        # load image from disk
        e = Image.open(self.examples_p[index])
        l = Image.open(self.labels_p[index])

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
        


