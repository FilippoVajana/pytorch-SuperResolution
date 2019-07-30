from imports import *
from engine.logger import Logger
from tqdm import tqdm

class Trainer():
    def __init__(self, model, device):
        self.device = device

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

    
    def run(self, epochs = 0, train_dataloader=None, validation_dataloader=None):
        """
        Starts the train-validation loop.
        """

        if train_dataloader == None :
            raise Exception("Invalid train dataloader.")

        self.log = Logger()
        self.log.add_metric("t_loss")
        self.log.add_metric("v_loss")

        #########
        # DEBUG #
        ######### 
        logging.warning(f"Epochs: {epochs}")
        logging.warning(f"Batches: {len(train_dataloader)}")
        logging.warning(f"Batch size: {train_dataloader.batch_size}")

        

        for epoch in range(epochs):
            tqdm.write("Epoch: {}".format(epoch + 1))
            
            # train loop
            t_loss = list()

            self.model.train()
            for batch in tqdm(train_dataloader):
                loss = self.__train_batch(batch)
                t_loss.append(loss)
                
            # update train log
            self.log.add_batch("t_loss", (epoch, t_loss))           


            # update learning rate
            self.scheduler.step()
            
            if validation_dataloader == None : 
                continue


            # validation loop
            v_loss = list()

            self.model.eval()
            with torch.no_grad():
                for batch in tqdm(validation_dataloader):
                    loss = self.__validate_batch(batch)
                    v_loss.append(loss)
            
            # update validation log
            self.log.add_batch("v_loss", (epoch, v_loss))
        

    def __train_batch(self, batch):
        """
        Train work for a batch.
        """                 

        examples, targets = batch     

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

        return loss


    def __validate_batch(self, batch):
        """
        Validation work for a batch.
        """
        
        examples, targets = batch

        # move data to device
        examples = examples.to(self.device)
        targets = targets.to(self.device)

        # forward
        predictions = self.model(examples)

        # compute loss
        loss = self.loss_fn(predictions, targets)

        # log validation loss
        return loss

