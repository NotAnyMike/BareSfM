import itertools
from pdb import set_trace
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import networks
import dataloaders

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
        if torch.cuda_is_available():
            self.device = torch.device('cuda')
        # TODO use to(device) it is not used so far

        # Get networks
        pose_class = getattr(networks, pose_network)
        encoder_class = getattr(networks, encoder_network)
        decoder_class = getattr(networks, decoder_network)

        # Initialize networks
        self.decoder_ntw = decoder_class()
        self.encoder_ntw = encoder_class()
        self.depth = nn.Sequential(self.encoder_ntw, self.decoder_ntw)
        self.pose_ntw = pose_class(height=self.height,
                                   width=self.width)
        self.params = itertools.chain(
            self.decoder_ntw.parameters(),
            self.encoder_ntw.parameters(),
            self.pose_ntw.parameters())

        # Get dataloader
        dataset_class = getattr(dataloaders, dataset_name)
        self.dataset = dataset_class(
            height=self.height, width=self.width, main_folder=dataset_folder, is_train=True, frames=frames)
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers)

        # TODO Structural similarity loss (SSIM) is implemented on pytorchgeometry/kornia
        self.l1_loss = nn.L1Loss(reduction="mean") # TODO
        self.optimizer = optim.Adam(self.params, # TODO
                                    lr=self.lr,
                                    betas=(self.beta1, self.beta2))

    def train(self):
        for epoch in range(self.num_epochs):
            for batch_num, batch in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                depth_a = self.depth(batch[('color', 'a')]) # TODO Check if it is a or b

                ab = torch.cat((batch[('color', 'a')], batch[('color', 'b')]), 1)
                pose_ab = self.pose_ntw(ab)
                # TODO 
                # TODO test inputs are valid for networks

                if batch_num > 0:
                    break
            if epoch > 2: break
