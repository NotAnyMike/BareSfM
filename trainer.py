# Python libraries
import itertools
import random
import os
from pdb import set_trace

# Third party libraries
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Own modules
import networks
import dataloaders
from operations import Projection

class SfM():
    def __init__(
            self, *, 
            no_gpu,
            height, width,
            pose_network, encoder_network, decoder_network,
            batch_size, num_epochs,
            lr, beta1, beta2,
            dataset_name, dataset_folder, frames, num_workers,
            log_frequency, save_frequency, model_name, save_folder):

        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.log_frequency = log_frequency
        self.save_frequency = save_frequency
        self.model_name = model_name
        self.save_folder = save_folder

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.device = torch.device('cpu')
        if torch.cuda.is_available() and not no_gpu:
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

        # Get operations
        self.projection = Projection(height=self.height,width=self.width,
                                     batch_size=self.batch_size).to(self.device)
        self.projection_inference = Projection(height=self.height,width=self.width,
                                               batch_size=1).to(self.device)

        # Get dataloader
        dataset_class = getattr(dataloaders, dataset_name)
        self.dataset = dataset_class(
            height=self.height, width=self.width, main_folder=dataset_folder,
            is_train=True, frames=frames, K_dim=(3,3))
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, drop_last=True)

        # TODO Structural similarity loss (SSIM) is implemented on pytorchgeometry/kornia
        self.criterion = nn.MSELoss(reduction="mean") # TODO
        self.optimizer = optim.Adam(self.params, # TODO
                                    lr=self.lr,
                                    betas=(self.beta1, self.beta2))

        # Logger
        self.writer = SummaryWriter(os.path.join(self.save_folder, self.model_name))

    def train(self):
        """ Training loop for model """
        img_id = random.choice(range(0, len(self.dataset)))
        total_batch_num = 0
        for epoch in range(self.num_epochs):
            for batch_num, batch in enumerate(self.dataloader):
                for key, val in batch.items():
                    # Sending everything to gpu
                    batch[key] = val.to(self.device)

                self.optimizer.zero_grad()
                b_prime, depth_b, pose_ab = self.predict(source=batch[('color', 'a')],
                                                         target=batch[('color', 'b')],
                                                         K=batch['K'],
                                                         K_inv=batch['K_inv'],
                                                         projection_fn=self.projection)

                loss = self.criterion(b_prime, batch[('color', 'b')]) 
                loss.backward()

                self.optimizer.step()

                if total_batch_num % self.log_frequency == 0:
                    # log
                    self.log(loss=loss.item(), step=total_batch_num, img_id=img_id)
                total_batch_num += 1

                #if batch_num > 0:
                    #break
            text = "Epoch {:>3d} | Loss: {:>6.4f} | ETA: 00h:00m".format(
                    epoch, loss.item())
            print(text)

    def predict(self, *, source, target, K, K_inv, projection_fn):
        """
        Predicts the new image and depth map

        source: source of the sampling
        target: the image we want to reconstruct
        """
        target_depth = self.depth(target) # TODO Check if it is a or b, I think it is b
        #tmp =self.encoder_ntw(batch[('color'),'b'])
        #depth_b = self.decoder_ntw(tmp)

        imgs = torch.cat((source, target), 1)
        pose = self.pose_ntw(imgs)

        K = K.view(-1, 1, 3, 3).expand(-1, int(self.height*self.width), -1, -1)
        K_inv = K_inv.view(-1, 1, 3, 3).expand(-1, int(self.height*self.width), -1, -1)

        proj = projection_fn(img=source,
                               z=target_depth,
                               pose=pose,
                               K=K,
                               K_inv=K_inv)
        return proj, target_depth, pose

    def log(self, *, loss, step, img_id=None):
        """ if img_id == None, each time it logs, the prediction image will be random """
        if img_id == None:
            img_id = random.choice(range(0, len(self.dataloader)))

        data = self.dataset[img_id]

        # Get images
        source = data[('color', 'a')].view(1, 3, self.height, self.width).to(self.device)
        target = data[('color', 'b')].view(1, 3, self.height, self.width).to(self.device)
        K = data['K'].to(self.device)
        K_inv = data['K_inv'].to(self.device)

        # Predict image using current model
        proj, depth, _ = self.predict(source=source,
                                     target=target,
                                     K=K,
                                     K_inv=K_inv,
                                     projection_fn=self.projection_inference)

        self.writer.add_image("org", target[0], step)
        self.writer.add_image("pred", proj[0], step)
        self.writer.add_image("depth", depth[0], step)
        self.writer.add_scalar("loss", loss, step) 
