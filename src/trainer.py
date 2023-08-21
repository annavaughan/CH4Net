from matplotlib import pyplot as plt
import torch 
import scipy
import numpy as np 
from tqdm import tqdm
from torch import autograd
from torch.utils.data import DataLoader
from loader import *

class Trainer():
    """
    Training class for the neural process models
    """
    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 train_dataset,
                 loss_function,
                 save_path,
                 learning_rate):
      
        # Model and data
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_dataset = train_dataset
        self.save_path = save_path

        # Training parameters
        self.opt = torch.optim.Adam(model.parameters(), lr=learning_rate) 
        self.loss_function = loss_function

        # Losses
        self.losses = []

    def plot_losses(self):
        """
        Plot losses for the trained model
        """
        plt.plot(np.array(self.losses))
        plt.xlabel("epoch")
        plt.ylabel("NLL")
        plt.show()

    def _unravel_to_numpy(self, x):
        return x.view(-1).detach().cpu().numpy()

            
    def eval_epoch(self, verbose=False):

        self.model.eval()
        lf = []

        outs = []
        ts = []

        with torch.no_grad():
            for task in self.val_loader:

                out = self.model(task["pred"])
                lf.append(self.loss_function(out[...,0], task["target"]))
                outs.append(out.detach().cpu().numpy())
                ts.append(task["target"].detach().cpu().numpy())

        # Get loss function
        log_loss = torch.mean(torch.tensor(lf))
        print("- Log loss: {}".format(log_loss))

        if verbose:
            return log_loss, np.concatenate(outs, axis=0), np.concatenate(ts, axis=0)

        return log_loss

    def train(self, n_epochs = 100):

        # Init progress bar
        best_loss = 100000
        
        for epoch in range(n_epochs):

            autograd.set_detect_anomaly(True)

            print("Training epoch {}".format(epoch))

            if epoch <500:
                self.model.train()
                self.train_dataset.sample_labels_and_combine()
                self.train_loader = DataLoader(self.train_dataset, 
                            batch_size = 16, 
                            shuffle = True)

            with tqdm(self.train_loader, unit="batch") as tepoch:
                for task in tepoch:

                    out = self.model(task["pred"])

                    #print("loss")
                    loss = self.loss_function(out[...,0], task["target"])

                    loss.backward()
                    self.opt.step()
                    self.opt.zero_grad()
                    #print("out")
                    tepoch.set_postfix(loss=loss.item())
            epoch_loss, o, t = self.eval_epoch(verbose=True)
            if np.logical_or(epoch_loss <= best_loss, epoch>=1000):
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.opt.state_dict(),
                    'loss': epoch_loss
                    }, self.save_path+"epoch_{}".format(epoch))
                best_loss = epoch_loss
                np.save("outs.npy", o)
                np.save("ts.npy", t)

            self.losses.append(epoch_loss)
            np.save(self.save_path+"losses.npy", np.array(self.losses))

        
        print("Training complete!")
