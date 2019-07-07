from sr_imports import *
from engine.logger import Logger
import tqdm

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
        self.train_log = None
        self.validation_log = None

    
    def run_train(self, epochs = 0, train_dataloader=None, validation_dataloader=None):
        """
        Starts the train-validation loop.
        """

        self.train_log = Logger([("t_loss", lambda x: x)])
        self.validation_log = Logger([("v_loss", lambda x: x)])
        
        for epoch in range(epochs):
            tqdm.write("Epoch: {}".format(epoch + 1))

            # train
            self.model.train()
            for batch in tqdm(train_dataloader):
                self.__train_batch(batch)

            # update learning rate
            self.scheduler.step()

            # validation
            self.model.eval()
            for batch in tqdm(validation_dataloader):
                self.__validate_batch(batch)
        pass


    def __train_batch(self, batch):
        """
        Train loop for a batch.
        """
        for examples, labels in batch:            
            # move data to device
            examples = examples.to(self.device)
            targets = targets.to(self.device)

            # reset gradient computation
            self.optimizer.zero_grad()

            # forward
            predictions = self.model(examples)

            # compute loss
            loss = self.loss_fn(predictions, targets)

            # backpropagation and gradients computation
            loss.backward()

            # update weights
            self.optimizer.step()

            # log training loss
            self.train_log.add_value("t_loss", loss)



    def __validate_batch(self, batch):
        """
        Validation loop for a batch.
        """
        pass

