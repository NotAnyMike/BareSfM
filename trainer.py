import itertools
from pdb import set_trace
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import networks
import dataloaders
from operations import projection

class SfM():
    def __init__(
            self,
            height, width,
            pose_network, encoder_network, decoder_network,
            batch_size, num_epochs,
            lr, beta1, beta2,
            dataset_name, dataset_folder, frames, num_workers):

        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_workers = num_workers

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        # TODO use to(device) it is not used so far

        # Get networks
        pose_class = getattr(networks, pose_network)
        encoder_class = getattr(networks, encoder_network)
        decoder_class = getattr(networks, decoder_network)

        # Initialize networks
        self.decoder_ntw = decoder_class().to(self.device)
        self.encoder_ntw = encoder_class().to(self.device)
        self.depth = nn.Sequential(self.encoder_ntw, self.decoder_ntw).to(self.device)
        self.pose_ntw = pose_class(height=self.height,
                                   width=self.width).to(self.device)
        self.params = itertools.chain(
            self.decoder_ntw.parameters(),
            self.encoder_ntw.parameters(),
            self.pose_ntw.parameters())

        # Get dataloader
        dataset_class = getattr(dataloaders, dataset_name)
        self.dataset = dataset_class(
            height=self.height, width=self.width, main_folder=dataset_folder,
            is_train=True, frames=frames, K_dim=(3,3))
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers)

        # TODO Structural similarity loss (SSIM) is implemented on pytorchgeometry/kornia
        self.criterion = nn.MSELoss(reduction="mean") # TODO
        self.optimizer = optim.Adam(self.params, # TODO
                                    lr=self.lr,
                                    betas=(self.beta1, self.beta2))

    def train(self):
        for epoch in range(self.num_epochs):
            for batch_num, batch in enumerate(self.dataloader):
                for key, val in batch.items():
                    # Sending everything to gpu
                    batch[key] = val.to(self.device)

                self.optimizer.zero_grad()

                depth_b = self.depth(batch[('color', 'b')]) # TODO Check if it is a or b, I think it is b
                #tmp =self.encoder_ntw(batch[('color'),'b'])
                #depth_b = self.decoder_ntw(tmp)

                ab = torch.cat((batch[('color', 'a')], batch[('color', 'b')]), 1)
                pose_ab = self.pose_ntw(ab)

                K = batch['K'].view(-1,1,3,3).expand(-1,int(self.height*self.width),-1,-1)
                K_inv = batch['K_inv'].view(-1,1,3,3).expand(-1,int(self.height*self.width),-1,-1)

                b_prime = projection(img=batch[('color','a')],
                                     z=depth_b,
                                     pose=pose_ab,
                                     K=batch['K'],
                                     K_inv=batch['K_inv'])

                loss = self.criterion(b_prime, batch[('color','b')]) 
                loss.backward()

                self.optimizer.step()

                if batch_num > 0:
                    break
            text = "Epoch {:>3d} | Loss: {:>6.4f} | ETA: 00h:00m".format(
                    epoch, loss.item())
            print(text)
