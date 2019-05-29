from sr_imports import *
from sr_data import SRDataset
from sr_model import *
from sr_trainer import Trainer
from sr_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Super Resolution - Neural Network")
    parser.add_argument('build_data', action="store", default=False, help='Build train data')
    args = parser.parse_args()

    # run parameters
    train_dir = "data/small_train_1"
    train_examples = 10
    train_epochs = 1
    train_img_size = 128
    label_img_size = train_img_size * 2

    # init model
    model = Upconv(train_img_size)

    # init train data
    # if args.build_data == True:  
    resize_img_batch("data/train", "data/small_train_1", 10, train_img_size)

    # init data loader
    dataset = SRDataset(train_dir)
    loader = tdata.DataLoader(dataset, batch_size=2, shuffle=True)

    # train
    trainer = Trainer(model)
    train_loss, _ = trainer.train(train_epochs, loader, None)    

    # evaluate 
    result_dir = "data/result/"    
    for idx, (e,l) in enumerate(dataset):    
        original = Image.open(dataset.examples[idx])
        t = tvision.transforms.ToTensor()
        original = t(original)       

        output = model(e.unsqueeze(0))
        output = output.squeeze(0).detach()  

        fig = show_results((original, output, l), display=False)
        fig.savefig(os.path.join(result_dir, "res_e{}_s{}".format(train_epochs, idx)))
        