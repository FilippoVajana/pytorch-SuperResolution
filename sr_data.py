from sr_imports import *

CACHE = dict()
class SRDataset(tdata.Dataset):
    """Dataset specific for Super Resolution.
    
    Attributes:
        root_dir: images source directory.
        transform: transformation to be applied to each image.
    """

    def __init__(self, root_dir, upscale_factor = 8):
        self.root_dir = root_dir
        self.upscale = upscale_factor

        files = os.listdir(root_dir)

        # get examples path
        example_re = re.compile(r'\d*x\d*.png')
        self.examples = list(filter(example_re.match, files))

        # get labels path
        label_re = re.compile(r'\d{4}.png')
        self.labels = list(filter(label_re.match, files))
        

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        """ 
        Returns:
            array: a 2d array of tensors [sample_t, label_t]
        """

        e_path = os.path.join(self.root_dir, self.examples[index])
        l_path = os.path.join(self.root_dir, self.labels[index])

        # load image (think about a cache system)
        e = Image.open(e_path)
        l = Image.open(l_path)

        # define data transformations
        size = 256
        example_t = tvision.transforms.Compose([
            #tvision.transforms.Resize((size,size)),
            tvision.transforms.ToTensor()
        ])
        label_t = tvision.transforms.Compose([
            #tvision.transforms.Resize((self.upscale * size, self.upscale * size)),
            tvision.transforms.ToTensor()
        ])

        return [example_t(e), label_t(l)]

    def show_data(self, index):
        ex, lab = self.__getitem__(index)

        fig, axarr = plt.subplots(1,2)

        ex = ex.permute((1,2,0))
        lab = lab.permute((1,2,0))
        
        axarr[0].imshow(ex)
        axarr[1].imshow(lab)

        axarr[0].set_title(self.examples[index])
        axarr[1].set_title(self.labels[index])

        axarr[0].set_axis_off()
        axarr[1].set_axis_off()
        plt.show()

    def split(self, train = .8, validation = .2):
        l = [int(len(self) * train), int(len(self) - len(self) * train)]
        ds = tdata.random_split(self, l)
        print(ds)
        


