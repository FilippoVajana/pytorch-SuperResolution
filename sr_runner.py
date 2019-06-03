from sr_imports import *
from sr_data import SRDataset
from sr_model import *
from sr_trainer import Trainer
from sr_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Super Resolution - Neural Network")
    parser.add_argument('-t', action='store_true', help='Train the model')
    parser.add_argument('-s', action='store_true', help='Save prediction result')
    parser.add_argument('-e', type=int, action='store', help='Number of epoch')    
    parser.add_argument('-tc', action='store_true', help='Load model and continue the training')
    parser.add_argument('-clear', action='store_true', help='Remove used files')
    parser.add_argument('-gpu', type=int, action='store', default=0, help='Select the gpu')
    args = parser.parse_args()


    ### default run parameters
    root_dir = "data/div2k/"
    full_train_dir = os.path.join(root_dir, "train/bicubic_2x")
    small_train_dir = os.path.join(root_dir, "train/bicubic_2x_small")
    result_dir = os.path.join(root_dir, "result")

    train_examples = 10
    train_epochs = 5

    train_img_size = 128
    label_img_size = train_img_size * 2
    device = get_device(args.gpu)


    ### init folders
    create_directory(result_dir)
    

    ### load cli params
    do_train = args.t
    do_continue = args.tc
    train_epochs = args.e if args.e is not None else train_epochs

    ### init model
    model = Upconv(train_img_size)
    dev_count = torch.cuda.device_count()
    # if dev_count > 1:
    #     print("Let's use", dev_count, "GPUs!")
    #     model = nn.DataParallel(model, [1])
    
    # init data loader
    dataset = SRDataset(full_train_dir)
    loader = tdata.DataLoader(dataset, batch_size=10, shuffle=True)

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
    res_count = 5
    for idx, (e,l) in enumerate(dataset):    
        original = Image.open(dataset.examples[idx])
        original = toTensor(original)

        # move model to cpu
        model.cpu()

        output = model(e.unsqueeze(0))
        output = output.squeeze(0).detach() 

        # print(original.type(), output.type(), l.type())

        fig = show_results((original, output, l), display=False)

        if args.s:
            print("Saving result {}".format(idx))        
        fig.savefig(os.path.join(result_dir, "res_epochs{}_sample{}".format(train_epochs, idx)), dpi=250)
        res_count -= 1
        if res_count < 0: break

    # clear
    if args.clear:
        print("Cleaning")
        shutil.rmtree(full_train_dir)
        shutil.rmtree(result_dir)
        