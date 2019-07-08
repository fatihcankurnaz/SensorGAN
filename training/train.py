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

import torch.optim as optim
import torch.nn as nn



parser = optparse.OptionParser()

parser.add_option('-c', '--config', dest="config",
                  help="load this config file", metavar="FILE")



def trian(dataloader,config):
    sensor1_gen_losses = []
    sensor2_gen_losses = []
    sensor1_disc_losses = []
    sensor2_disc_losses = []
    cycle_loss = []

    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    cycleLossCalculation = nn.MSELoss(reduction='mean')

    real_sample_label = 1
    fake_sample_label = 0

    sensor1_gen = Generator(1,5).to(device)
    sensor2_gen = Generator(1,5).to(device)
    sensor1_dis = Discriminator(...).to(device)
    sensor2_dis = Discriminator(...).to(device)

    # Setup Adam optimizers for both G and D
    optimizer_sensor1_gen = optim.Adam(sensor1_gen.parameters(), lr=config.SENSOR1_GENERATOR.BASE_LR)
    optimizer_sensor2_gen = optim.Adam(sensor2_gen.parameters(), lr=config.SENSOR2_GENERATOR.BASE_LR)
    optimizer_sensor1_dis = optim.Adam(sensor1_dis.parameters(), lr=config.SENSOR1_DISCRIMINATOR.BASE_LR)
    optimizer_sensor2_dis = optim.Adam(sensor2_dis.parameters(), lr=config.SENSOR2_DISCRIMINATOR.BASE_LR)





def main(opts):
    load_config(opts.config)
    dataloader = DataLoader(config)
    train(dataloader,config)


if __name__ == "__main__":
    options, args = parser.parse_args()
    main(options)
