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
    sensor1_gen_losses = []
    sensor2_gen_losses = []
    sensor1_disc_losses = []
    sensor2_disc_losses = []
    cycle_loss = []

    # nn.BCEWithLogitsLoss(reduction='mean') # works better with log loss
    criterion = nn.BCELoss(reduction=config.TRAIN.DISCRIMINATOR_CRITERION_REDUCTION)
    cycleLossCalculation = nn.MSELoss(reduction=config.TRAIN.CYCLE_LOSS_REDUCTION)

    real_sample_label = 1
    fake_sample_label = 0

    # each sensor should have their own Generator and Discriminator because their input size will probably not match
    sensor1_gen = Generator(1,5).to(device)
    sensor2_gen = Generator(1,5).to(device)
    sensor1_dis = Discriminator(1).to(device)
    sensor2_dis = Discriminator(1).to(device)

    # Setup Adam optimizers for both G and D
    optimizer_sensor1_gen = optim.Adam(sensor1_gen.parameters(), lr=config.SENSOR1_GENERATOR.BASE_LR)
    optimizer_sensor2_gen = optim.Adam(sensor2_gen.parameters(), lr=config.SENSOR2_GENERATOR.BASE_LR)
    optimizer_sensor1_dis = optim.Adam(sensor1_dis.parameters(), lr=config.SENSOR1_DISCRIMINATOR.BASE_LR)
    optimizer_sensor2_dis = optim.Adam(sensor2_dis.parameters(), lr=config.SENSOR2_DISCRIMINATOR.BASE_LR)

    if config.TRAIN.START_EPOCH > 0:
        print("loading previous model")
        checkpoint = torch.load(config.TRAIN.LOAD_WEIGHTS)
        sensor1_gen.load_state_dict(checkpoint['sensor1_gen'])
        sensor2_gen.load_state_dict(checkpoint['sensor2_gen'])
        sensor1_dis.load_state_dict(checkpoint['sensor1_dis'])
        sensor2_dis.load_state_dict(checkpoint['sensor2_dis'])
        optimizer_sensor1_gen.load_state_dict(checkpoint['optimizer_sensor1_gen'])
        optimizer_sensor2_gen.load_state_dict(checkpoint['optimizer_sensor2_gen'])
        optimizer_sensor1_dis.load_state_dict(checkpoint['optimizer_sensor1_dis'])
        optimizer_sensor2_dis.load_state_dict(checkpoint['optimizer_sensor2_dis'])

        sensor1_gen.train()
        sensor2_gen.train()
        sensor1_dis.train()
        sensor2_dis.train()
        print("done")

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.END_EPOCH):

        ################################################################################
        #                               Zero Gradients
        ################################################################################
        sensor1_gen.zero_grad()
        sensor2_gen.zero_grad()
        sensor1_dis.zero_grad()
        sensor2_dis.zero_grad()

        optimizer_sensor1_gen.zero_grad()
        optimizer_sensor2_gen.zero_grad()
        optimizer_sensor1_dis.zero_grad()
        optimizer_sensor2_dis.zero_grad()
        ################################################################################
        #                           First Sensor Discriminator
        ################################################################################


        sensor2_sample = ...

        # Forward pass real batch through D
        output = sensor_1(sensor2_sample)

        o

        ################################################################################
        #                           First Sensor Generator
        ################################################################################

        ################################################################################
        #                           Second Sensor Discriminator
        ################################################################################

        ################################################################################
        #                           Second Sensor Generator
        ################################################################################

    save_model(config, sensor1_gen, sensor2_gen , sensor1_dis , sensor2_dis, optimizer_sensor1_gen,
               optimizer_sensor2_gen, optimizer_sensor1_dis, optimizer_sensor2_dis )


def main(opts):
    load_config(opts.config)
    dataloader = DataLoader(config)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    train(dataloader,config, device)


if __name__ == "__main__":
    options, args = parser.parse_args()
    main(options)
