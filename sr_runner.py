from sr_imports import *
from sr_data import SRDataset
from sr_model import *
from sr_trainer import Trainer
from sr_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Super Resolution - Neural Network")
    parser.add_argument('-d', action='store_true', help='Build train data')
    parser.add_argument('-ds', type=int, action='store', help='Train data size')
    parser.add_argument('-t', action='store_true', help='Train the model')
    parser.add_argument('-s', action='store_true', help='Save prediction result')
    parser.add_argument('-e', type=int, action='store', help='Number of epoch')    
    parser.add_argument('-tc', action='store_true', help='Load model and continue the training')

    args = parser.parse_args()
    # print(args)

    # default run parameters
    source_dir = "data/train"
    train_dir = "data/small_train_1"
    train_examples = 10
    train_epochs = 5
    train_img_size = 128
    label_img_size = train_img_size * 2
    result_dir = "data/result/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    

    # init folders
    try:  
        os.mkdir(train_dir)
    except OSError:  
        print ("Creation of the directory %s failed" % train_dir)
    else:  
        print ("Successfully created the directory %s " % train_dir)

    try:  
        os.mkdir(result_dir)
    except OSError:  
        print ("Creation of the directory %s failed" % result_dir)
    else:  
        print ("Successfully created the directory %s " % result_dir)

    # load cli params
    do_build_tdata = args.d
    train_examples = args.ds if do_build_tdata and args.ds is not None else train_examples
    do_train = args.t
    do_continue = args.tc
    train_epochs = args.e if args.e is not None else train_epochs

    # init model
    model = Upconv(train_img_size)

    # init train data
    if do_build_tdata:  
        resize_img_batch(source_dir, train_dir, train_examples, train_img_size)

    # init data loader
    dataset = SRDataset(train_dir)
    loader = tdata.DataLoader(dataset, batch_size=2, shuffle=True)

    # train
    trainer = Trainer(model, device)
    if do_train == True:
        print("Start training")
        trainer.train(train_epochs, loader, None)
        print("Saving model")
        torch.save(model.state_dict(), 'model.pt')

    if do_train == False:
        print("Loading model")
        model.load_state_dict(torch.load('model.pt'))
        if do_continue == True:
            print("Continue training")
            trainer.train(train_epochs, loader, None)
            print("Update model")
            torch.save(model.state_dict(), 'model.pt')


    # evaluate   
    model.eval()
    toTensor = tvision.transforms.ToTensor()
    for idx, (e,l) in enumerate(dataset):    
        original = Image.open(dataset.examples[idx])
        original = toTensor(original).to(device)
        e = e.to(device)
        l = l.to(device)  

        output = model(e.unsqueeze(0))
        output = output.squeeze(0).detach()  

        fig = show_results((original, output, l), display=False)
        fig.savefig(os.path.join(result_dir, "res_epochs{}_sample{}".format(train_epochs, idx)), dpi=250)
        