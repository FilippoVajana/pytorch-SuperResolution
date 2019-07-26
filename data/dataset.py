from imports import *
from torchvision import transforms
from cachetools import cached, cachedmethod, LRUCache
import operator
from skimage import color, io
from skimage.transform import rescale
import scipy


class SRDataset(tdata.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.labels_dir = os.path.join(self.data_dir, "labels")

        self.examples_p = [os.path.join(self.data_dir, name) for name in os.listdir(self.data_dir) if name != "labels"] 
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
        
        # RGB to YCbCr
        # e = color.rgb2ycbcr(e)
        # l = color.rgb2ycbcr(l)

        # upscale example
        # e = rescale(e, 2, multichannel=True, anti_aliasing=True)


        # print("data example", e.size)
        # print("data label", l.size)

        # define data transformations
        data_tr = [transforms.Resize((l.size[0], l.size[1])), transforms.CenterCrop(512), transforms.ToTensor()]
        example_t = tvision.transforms.Compose(data_tr)
        label_t = tvision.transforms.Compose(data_tr)

        # print(example_t(e).shape)
        # print(label_t(l).shape)

        return [example_t(e), label_t(l)]
        


