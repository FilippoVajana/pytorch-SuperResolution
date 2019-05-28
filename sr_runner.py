from sr_imports import *
from sr_data import SRDataset
from sr_model import *
from sr_trainer import Trainer
import math

def show_data(ex, lab):    
    ex = ex.squeeze(0).detach().numpy()
    #ex *= 255.0
    ex = ex.clip(0, 255)

    fig, axarr = plt.subplots(1,2)

    ex = torch.from_numpy(ex).permute((1,2,0))
    lab = lab.permute((1,2,0))

    #print(ex[2])

    axarr[0].imshow(ex)
    axarr[1].imshow(lab)

    axarr[0].set_axis_off()
    axarr[1].set_axis_off()

    plt.show()
   


if __name__ == "__main__":
    ds = SRDataset("data/s_train", 1)
    model = Upconv()
    train_loader = tdata.DataLoader(ds, batch_size = 2, shuffle = True)
    trainer = Trainer(model)
    t = trainer.train(1, train_loader, None)

    for e,l in ds:    
        pred = model(e.unsqueeze(0))
        show_data(pred, l)
        break

    
