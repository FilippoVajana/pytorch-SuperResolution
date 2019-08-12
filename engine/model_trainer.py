from imports import *
from engine.logger import Logger
from tqdm import tqdm
from data.metrics import Metrics

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
        self.log = Logger("train_log", ["t_loss", "t_psnr", "v_loss", "v_psnr"])
        
    
    def run(self, epochs = 0, train_dataloader=None, validation_dataloader=None):
        """
        Starts the train-validation loop.
        """

        if train_dataloader == None :
            raise Exception("Invalid train dataloader.")
    

        #########
        # DEBUG #
        ######### 
        logging.warning(f"Epochs: {epochs}")
        logging.warning(f"Batches: {len(train_dataloader)}")
        logging.warning(f"Batch size: {train_dataloader.batch_size}")

        

        for epoch in tqdm(range(epochs)):            
            # train loop
            tmp_loss = torch.zeros(len(train_dataloader), device=self.device)
            tmp_psnr = torch.zeros(len(train_dataloader), device=self.device)

            self.model.train()
            for idx, batch in enumerate(train_dataloader):
                b_loss, b_psnr = self.__train_batch(batch)
                tmp_loss[idx] = b_loss
                tmp_psnr[idx] = b_psnr

            # update train log
            self.log.add("t_loss", tmp_loss.mean())  
            self.log.add("t_psnr", tmp_psnr.mean())     


            # update learning rate
            self.scheduler.step()

            # validation loop
            if validation_dataloader == None : 
                continue

            tmp_loss = torch.zeros(len(validation_dataloader), device=self.device)
            tmp_psnr = torch.zeros(len(validation_dataloader), device=self.device)

            self.model.eval()
            with torch.no_grad():
                for idx, batch in enumerate(validation_dataloader):
                    b_loss, b_psnr = self.__validate_batch(batch)
                    tmp_loss[idx] = b_loss
                    tmp_psnr[idx] = b_psnr
            
            # update validation log
            self.log.add("v_loss", tmp_loss.mean())
            self.log.add("v_psnr", tmp_psnr.mean())

        return self.log.to_dataframe()
        

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

        # compute psnr
        psnr_t = torch.tensor([Metrics.psnr(targets[idx], p) for idx, p in enumerate(predictions)])

        return loss.detach(), psnr_t.mean()


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

        # compute psnr
        psnr_t = torch.tensor([Metrics.psnr(targets[idx], p) for idx, p in enumerate(predictions)])

        
        return loss.detach(), psnr_t.mean()

