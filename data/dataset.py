from imports import *
from torchvision import transforms
from cachetools import cached, cachedmethod, LRUCache
import operator
from skimage import color, io
from skimage.transform import rescale
import scipy


class DatasetBuilder():
    def __init__(self):
        self.log = logging.getLogger()


    def __get_data(self, folder):
        ex_dir = folder
        lab_dir = os.path.join(ex_dir, 'labels')

        img_regex = re.compile(r'.png')
        self.log.info("Getting examples images.") 
        files = os.listdir(ex_dir)       
        examples = filter(img_regex.search, files)
        examples = list(map(lambda p: os.path.join(ex_dir, p), examples))
        examples.sort()
        
        self.log.info("Getting labels images.") 
        files = os.listdir(lab_dir)
        labels = filter(img_regex.search, files)
        labels = list(map(lambda p: os.path.join(lab_dir, p), labels))
        labels.sort()

        return examples, labels

    def build(self, data_folder):
        """
        Build a SRDataset from folder.
        
        Arguments:\\
            data_folder : str -- Path to data

        Returns:\\
            dataset : SRDataset -- Dataset
        """

        # get all data      
        examples, labels = self.__get_data(data_folder)  
        data = {
            'examples' : examples,
            'labels' : labels
        }

        # build dataset
        dataset = SRDataset(data)

        return dataset

    def build_splitted(self, data_folder, ratio = 0.9):
        """
        Splits data into train and validation dataset.
        
        Arguments:\\   
            data_folder : str -- Path to data
        
        Keyword Arguments:\\
            ratio : float -- Ration between train and validation datasets (default: {0.9})
        
        Returns:\\
            pair : (SRDataset, SRDataset) -- Train and validation datasets
        """

        # get all data      
        examples, labels = self.__get_data(data_folder)  

        # split data
        split_idx = int(len(examples * ratio))

        train_data = {
            'examples' : examples[0 : split_idx],
            'labels' : labels[0 : split_idx]
            }

        validation_data = {
            'examples' : examples[split_idx : -1],
            'labels' : labels[split_idx : -1]
        }

        # build train dataset
        train_ds = SRDataset(train_data)
        
        # build validation dataset
        validation_ds = SRDataset(validation_data)

        return train_ds, validation_ds


class SRDataset(tdata.Dataset):
    def __init__(self, data):
        """
        Initializes the dataset object.
        
        Arguments:\\            
            data : dict -- Data dictionary with examples and labels keys
        """

        # get paths
        self.examples = data['examples']
        self.labels = data['labels']

        # init cache system
        self.cache = LRUCache(len(self.examples))

    def __len__(self):
        return len(self.examples)

    @cachedmethod(operator.attrgetter('cache'))
    def __getitem__(self, index):
        """        
        Returns:\\
            item : [example_t, label_t] -- A 2D array of tensors [sample_t, label_t]
        """
        
        # load image from disk
        e = Image.open(self.examples[index])
        l = Image.open(self.labels[index])

        # define data transformations
        data_tr = [transforms.ToTensor()]
        example_t = tvision.transforms.Compose(data_tr)
        label_t = tvision.transforms.Compose(data_tr)

        return [example_t(e), label_t(l)]