from sr_imports import *
from sr_data import SRDataset
import sr_utils
from tqdm import tqdm

class Trainer():
    def __init__(self, model, device):
        self.device = device
        print("Using {}".format(self.device))
        
        # set model
        self.model = model.to(device)

        # set default optimizer
        lr = 0.01
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = 1, gamma=0.9)

        # set default loss function
        self.loss_fn = torch.nn.MSELoss()

        # define log objects
        self.log_train = TrainLog()
    
    def train(self, n_epochs = 10, train_loader = None, validation_loader = None):  
        # loop    
        for epoch in range(n_epochs):
            tqdm.write("Epoch: {}".format(epoch + 1))

            # train loop
            self.model.train()
            for batch, targets in tqdm(train_loader):
                # move data to device
                batch = batch.to(self.device)
                targets = targets.to(self.device)

                # reset gradient computation
                self.optimizer.zero_grad()

                # forward
                predictions = self.model(batch)

                # compute loss
                t_loss = self.loss_fn(predictions, targets)
                # tqdm.write(str(t_loss.item()))

                # backpropagation and gradients computation
                t_loss.backward()

                # update weights
                self.optimizer.step()

                # save training loss
                self.log_train.add_t_loss(t_loss)

                # compute psnr
                for b,t in zip(batch, targets):
                    psnr = sr_utils.psnr(t, b)
                    self.log_train.add_psnr(psnr)

            # update learning rate
            self.scheduler.step()
            # validation loop
            if validation_loader != None:
                self.model.eval()
                validation_loss[epoch] = list()
                with torch.no_grad():
                    for batch, targets in validation_loader:
                        # move to device
                        batch = batch.to(self.device)
                        targets = targets.to(self.device)

                        # forward
                        predictions = self.model(batch)

                        # compute loss
                        v_loss = self.loss_fn(predictions, targets)

                        # save validation loss
                        self.log_train.add_v_loss(v_loss)

                        # compute psnr
                        for b,t in batch,targets:
                            psnr = sr_utils.psnr(t, b)
                            self.log_train.add_psnr(psnr)
            
            # tqdm.write("Update log")
            self.log_train.update_epoch()

    def test(self, test_loader = None):        
        self.model.eval()
        p_results = torch.zeros(len(test_loader.dataset))
        count = 0
        with torch.no_grad():
            for batch, targets in test_loader:
                # move to device
                batch = batch.to(self.device)
                targets = targets.to(self.device)

                # forward
                predictions = self.model(batch)

                # check prediction
                for i in range(len(predictions)):
                    p_results[count] = abs(predictions[i].item() - targets[i].item())
                    count += 1

        return p_results