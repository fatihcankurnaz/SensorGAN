from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import optparse

from utils.core.config import config
from utils.data.Dataloader import lidar_camera_dataloader
from utils.core.config import load_config
from utils.helpers.helpers import save_model
from .Generator import Generator
from .Discriminator import Discriminator

import torch.optim as optim
import torch.nn as nn
import torch



parser = optparse.OptionParser()

parser.add_option('-c', '--core', dest="core",
                  help="load this core file", metavar="FILE")








def train(dataloader, config, device):
    lidar_gen_losses = []
    camera_gen_losses = []
    lidar_disc_losses = []
    camera_disc_losses = []
    cycle_loss = []

    # nn.BCEWithLogitsLoss(reduction='mean') # works better with log loss
    criterion = nn.BCELoss(reduction=config.TRAIN.DISCRIMINATOR_CRITERION_REDUCTION)
    cycleLossCalculation = nn.MSELoss(reduction=config.TRAIN.CYCLE_LOSS_REDUCTION)

    real_sample_label = 1
    fake_sample_label = 0

    # each sensor should have their own Generator and Discriminator because their input size will probably not match
    lidar_gen = Generator(1,5).to(device)
    camera_gen = Generator(1,5).to(device)
    lidar_disc = Discriminator(1).to(device)
    camera_disc = Discriminator(1).to(device)

    # Setup Adam optimizers for both G and D
    optimizer_lidar_gen = optim.Adam(lidar_gen.parameters(), lr=config.LIDAR_GENERATOR.BASE_LR)
    optimizer_camera_gen = optim.Adam(camera_gen.parameters(), lr=config.CAMERA_GENERATOR.BASE_LR)
    optimizer_lidar_disc = optim.Adam(lidar_disc.parameters(), lr=config.LIDAR_DISCRIMINATOR.BASE_LR)
    optimizer_camera_disc = optim.Adam(camera_disc.parameters(), lr=config.CAMERA_DISCRIMINATOR.BASE_LR)


    if config.TRAIN.START_EPOCH > 0:
        print("loading previous model")
        checkpoint = torch.load(config.TRAIN.LOAD_WEIGHTS)
        lidar_gen.load_state_dict(checkpoint['lidar_gen'])
        camera_gen.load_state_dict(checkpoint['camera_gen'])
        lidar_disc.load_state_dict(checkpoint['lidar_disc'])
        camera_disc.load_state_dict(checkpoint['camera_disc'])
        optimizer_lidar_gen.load_state_dict(checkpoint['optimizer_lidar_gen'])
        optimizer_camera_gen.load_state_dict(checkpoint['optimizer_camera_gen'])
        optimizer_lidar_disc.load_state_dict(checkpoint['optimizer_lidar_disc'])
        optimizer_camera_disc.load_state_dict(checkpoint['optimizer_camera_disc'])

        lidar_gen.train()
        camera_gen.train()
        lidar_disc.train()
        camera_disc.train()
        print("done")

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.END_EPOCH):
        for current_batch, data in enumerate(dataloader, 0):

            label_fake = torch.full((config.TRAIN.BATCH_SIZE,), real_sample_label, device=device)
            label_real = torch.full((config.TRAIN.BATCH_SIZE,), fake_sample_label, device=device)

            label_real = label_real.cuda()
            label_fake = label_fake.cuda()

            camera_sample = data["camera_data"].device()
            lidar_sample = data["lidar_data"].device()

            ################################################################################
            #                               Zero Gradients
            ################################################################################
            lidar_gen.zero_grad()
            camera_gen.zero_grad()
            lidar_disc.zero_grad()
            camera_disc.zero_grad()

            optimizer_lidar_gen.zero_grad()
            optimizer_camera_gen.zero_grad()
            optimizer_lidar_disc.zero_grad()
            optimizer_camera_disc.zero_grad()
            ################################################################################
            #                           Camera Discriminator
            ################################################################################

            output = camera_disc(camera_sample)

            output = output.view(-1)

            label_real.fill_(real_sample_label)

            # Calculate loss on all-real batch
            real_sample_error = criterion(output, label_real)

            # Calculate gradients for D in backward pass
            real_sample_error.backward()

            camera_disc_real_sample_output = output.mean().item()

            generated_camera_sample = camera_gen(lidar_sample)
            label_fake.fill_(fake_sample_label)

            # Classify all fake batch with D
            disc_on_fake_sample = camera_disc(generated_camera_sample.detach())
            disc_on_fake_sample = disc_on_fake_sample.view(-1)

            # Calculate D's loss on the all-fake batch
            fake_sample_error = criterion(disc_on_fake_sample, label_fake)

            # Calculate the gradients for this batch
            fake_sample_error.backward()

            camera_disc_fake_sample_output = output.mean().item()

            optimizer_camera_disc.step()

            ################################################################################
            #                           Camera Generator
            ################################################################################

            generated_camera_sample = camera_gen(lidar_sample)

            label_real.fill_(real_sample_label)  # fake labels are real for generator cost

            # Since we just updated D, perform another forward pass of all-fake batch through D
            disc_on_fake_sample = camera_disc(generated_camera_sample.detach())

            disc_on_fake_sample = disc_on_fake_sample.view(-1)

            # Calculate G's loss based on this output
            fake_sample_error = criterion(disc_on_fake_sample, label_real)
            # Calculate gradients for G
            fake_sample_error.backward()

            camera_gen_output = output.mean().item()

            optimizer_camera_gen.step()

            ################################################################################
            #                           Second Sensor Discriminator
            ################################################################################

            ################################################################################
            #                           Second Sensor Generator
            ################################################################################

            if epoch % 5 == 0:
                print(
                    '[%d/%d][%d/%d]\t\t Lidar to Cam GAN Loss_DF/R: %.4f/ %.4f \t'
                    'Loss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f \n'
                    % (epoch, config.TRAIN.END_EPOCH, current_batch, len(dataloader),
                       fake_sample_error.item(), real_sample_error.item(), fake_sample_error.item(),
                       camera_disc_fake_sample_output, camera_disc_real_sample_output, camera_gen_output))

    save_model(config, lidar_gen, camera_gen , lidar_disc , camera_disc, optimizer_lidar_gen,
               optimizer_camera_gen, optimizer_lidar_disc, optimizer_camera_disc )




def main(opts):
    load_config(opts.config)
    dataloader = lidar_camera_dataloader(config)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    train(dataloader,config, device)


if __name__ == "__main__":
    options, args = parser.parse_args()
    main(options)
