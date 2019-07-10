from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import optparse

from utils.config import config
from data.dataloader import DataLoader
from utils.config import load_config
from .Generator import Generator
from .Discriminator import Discriminator

import torch.optim as optim
import torch.nn as nn
import torch



parser = optparse.OptionParser()

parser.add_option('-c', '--config', dest="config",
                  help="load this config file", metavar="FILE")


def save_model(config, sensor1_gen, sensor2_gen, sensor1_dis, sensor2_dis,
               optimizer_sensor1_gen, optimizer_sensor2_gen, optimizer_sensor1_dis, optimizer_sensor2_dis):

    torch.save({
        'sensor1_gen': sensor1_gen.state_dict(),
        'sensor2_gen': sensor2_gen.state_dict(),
        'sensor1_dis': sensor1_dis.state_dict(),
        'sensor2_dis': sensor2_dis.state_dict(),
        'optimizer_sensor1_gen': optimizer_sensor1_gen.state_dict(),
        'optimizer_sensor2_gen': optimizer_sensor2_gen.state_dict(),
        'optimizer_sensor1_dis': optimizer_sensor1_dis.state_dict(),
        'optimizer_sensor2_dis': optimizer_sensor2_dis.state_dict()
    }, config.TRAIN.SAVE_WEIGHTS)


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

        label_fake = torch.full((config.TRAIN.BATCH_SIZE,), real_sample_label, device=device)
        label_real = torch.full((config.TRAIN.BATCH_SIZE,), fake_sample_label, device=device)

        label_real = label_real.cuda()
        label_fake = label_fake.cuda()

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

        camera_sample = ...

        output = camera_disc(camera_sample)

        output = output.view(-1)

        label_real.fill_(real_sample_label)

        # Calculate loss on all-real batch
        real_sample_error = criterion(output, label_real)

        # Calculate gradients for D in backward pass
        real_sample_error.backward()

        real_sample_error_value_disc = output.mean().item()

        generated_camera_sample = camera_gen(lidar_sample)
        label_fake.fill_(fake_sample_label)

        # Classify all fake batch with D
        disc_on_fake_sample = camera_disc(generated_camera_sample.detach())
        disc_on_fake_sample = disc_on_fake_sample.view(-1)

        # Calculate D's loss on the all-fake batch
        fake_sample_error = criterion(disc_on_fake_sample, label_fake)

        # Calculate the gradients for this batch
        fake_sample_error.backward()

        fake_sample_error_value_disc = output.mean().item()

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

        fake_sample_error_gen = output.mean().item()

        optimizer_camera_gen.step()

        ################################################################################
        #                           Second Sensor Discriminator
        ################################################################################

        ################################################################################
        #                           Second Sensor Generator
        ################################################################################

    


    save_model(config, lidar_gen, camera_gen , lidar_disc , camera_disc, optimizer_lidar_gen,
               optimizer_camera_gen, optimizer_lidar_disc, optimizer_camera_disc )


def main(opts):
    load_config(opts.config)
    dataloader = DataLoader(config)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    train(dataloader,config, device)


if __name__ == "__main__":
    options, args = parser.parse_args()
    main(options)
