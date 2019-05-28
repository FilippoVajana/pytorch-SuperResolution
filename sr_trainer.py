from sr_imports import *
from sr_data import SRDataset
from sr_model import SRModel

class Trainer():
    def __init__(self, model):
        # set gpu device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # set model
        self.model = model.to(self.device)
        # set default optimizer
        self.optimizer = torch.optim.Adam(model.parameters())
        # set default loss function
        self.loss_fn = torch.nn.MSELoss()
    
    def train(self, n_epochs = 10, train_loader = None, validation_loader = None):  
        train_loss = {}
        validation_loss = {}  
        count = 0
        # loop    
        for epoch in range(n_epochs):
            print("Epoch {}/{}".format(epoch + 1, n_epochs))
            # train loop
            self.model.train()
            train_loss[epoch] = list()
            for batch, targets in train_loader:
                count += train_loader.batch_size
                print("\texamples: {}/{}".format(count, len(train_loader.dataset)*n_epochs))
                # print(batch.shape)
                # move to device
                batch = batch.to(self.device)
                targets = targets.to(self.device)

                # reset gradient computation
                self.optimizer.zero_grad()

                # forward
                predictions = self.model(batch)

                # print(predictions.shape)
                # print(targets.shape)

                # calculate loss
                t_loss = self.loss_fn(predictions, targets)

                # backpropagation and gradients computation
                t_loss.backward()

                # update weights
                self.optimizer.step()

                # save training loss
                train_loss[epoch].append(t_loss.item())

            # validation loop
            if validation_loader == None:
                continue

            self.model.eval()
            validation_loss[epoch] = list()
            with torch.no_grad():
                for batch, targets in validation_loader:
                    # move to device
                    batch = batch.to(self.device)
                    targets = targets.to(self.device)

                    # forward
                    predictions = self.model(batch)

                    # calculate loss
                    v_loss = self.loss_fn(predictions, targets)

                    # save validation loss
                    validation_loss[epoch].append(v_loss)
            
        return train_loss, validation_loss

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