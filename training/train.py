from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import optparse

from utils.data.Dataloader import lidar_camera_dataloader
from utils.core.config import config

from utils.core.config import load_config
from utils.helpers.helpers import save_model
from utils.helpers.helpers import display_two_images


from Generator import Generator
from Discriminator import Discriminator


import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt



parser = optparse.OptionParser()

parser.add_option('-c', '--config', dest="config",
                  help="load this config file", metavar="FILE")








def train(dataloader, config, device):
    lidar_gen_losses = []
    camera_gen_losses = []
    lidar_disc_losses = []
    camera_disc_losses = []
    cycle_loss = []

    # nn.BCEWithLogitsLoss(reduction='mean') # works better with log loss
    criterion = nn.BCELoss(reduction=config.TRAIN.DISCRIMINATOR_CRITERION_REDUCTION)

    real_sample_label = 1
    fake_sample_label = 0

    # each sensor should have their own Generator and Discriminator because their input size will probably not match
    #lidar_gen = Generator(1,5).to(device)
    camera_gen = Generator(5, 5, config.NUM_GPUS).to(device)
    #lidar_disc = Discriminator(1).to(device)
    camera_disc = Discriminator(5, config.NUM_GPUS).to(device)

    if (device.type == 'cuda') and (config.NUM_GPUS > 1):
        camera_gen = nn.DataParallel(camera_gen, list(range(config.NUM_GPUS)))
    if (device.type == 'cuda') and (config.NUM_GPUS > 1):
        camera_disc = nn.DataParallel(camera_disc, list(range(config.NUM_GPUS)))

    # Setup Adam optimizers for both G and D
    #optimizer_lidar_gen = optim.Adam(lidar_gen.parameters(), lr=config.LIDAR_GENERATOR.BASE_LR)
    optimizer_camera_gen = optim.Adam(camera_gen.parameters(), lr=config.CAMERA_GENERATOR.BASE_LR)
    #optimizer_lidar_disc = optim.Adam(lidar_disc.parameters(), lr=config.LIDAR_DISCRIMINATOR.BASE_LR)
    optimizer_camera_disc = optim.Adam(camera_disc.parameters(), lr=config.CAMERA_DISCRIMINATOR.BASE_LR)

    camera_gen_total_params = sum(p.numel() for p in camera_gen.parameters())
    print("Camera Generator ", camera_gen_total_params)
    camera_disch_total_params = sum(p.numel() for p in camera_disc.parameters())
    print("Camera Discriminator ", camera_disch_total_params)
    camera_gen_loss = []
    camera_disc_loss = []
    example_camera_output = []

    if config.TRAIN.START_EPOCH > 0:
        print("loading previous model")

        camera_gen.load_state_dict(checkpoint['camera_gen'])
        camera_disc.load_state_dict(checkpoint['camera_disc'])
        optimizer_camera_gen.load_state_dict(checkpoint['optimizer_camera_gen'])
        optimizer_camera_disc.load_state_dict(checkpoint['optimizer_camera_disc'])
        camera_gen.train()
        camera_disc.train()
        print("done")

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.MAX_EPOCH):
        for current_batch, data in enumerate(dataloader, 0):
            if len(dataloader) - current_batch< config.TRAIN.BATCH_SIZE:
                continue
            label_real = torch.full((config.TRAIN.BATCH_SIZE,), real_sample_label, device=device)
            label_fake = torch.full((config.TRAIN.BATCH_SIZE,), fake_sample_label, device=device)

            label_real = label_real.cuda()
            label_fake = label_fake.cuda()


            #display_two_images(data["camera_data"][0], data["lidar_data"][0])
            camera_sample = data["camera_data"].to(device = device, dtype=torch.float)
            lidar_sample = data["lidar_data"].to(device = device, dtype=torch.float)

            ################################################################################
            #                               Zero Gradients
            ################################################################################
            #lidar_gen.zero_grad()

            #lidar_disc.zero_grad()


            #optimizer_lidar_gen.zero_grad()
            #optimizer_camera_gen.zero_grad()
            #optimizer_lidar_disc.zero_grad()
            #optimizer_camera_disc.zero_grad()

            ################################################################################
            #                           Camera Generator
            ################################################################################

            camera_gen.zero_grad()

            # By using lidar samples as input we generate Camera data
            generated_camera_sample = camera_gen(lidar_sample)

            disc_on_fake_sample_gen = camera_disc(generated_camera_sample)
            disc_on_fake_sample_gen = disc_on_fake_sample_gen.view(-1)


            #m Measure the generators capacity to trick discriminator

            fake_sample_error_gen = criterion(disc_on_fake_sample_gen, label_real)

            # Calculate gradients for G
            fake_sample_error_gen.backward()
            optimizer_camera_gen.step()

            camera_gen_output = disc_on_fake_sample_gen.mean().item()




            ################################################################################
            #                           Camera Discriminator
            ################################################################################

            camera_disc.zero_grad()

            real_data_output = camera_disc(camera_sample)
            real_data_output = real_data_output.view(-1)

            real_sample_error = criterion(real_data_output, label_real)

            real_sample_error.backward()

            camera_disc_real_sample_output = real_data_output.mean().item()

            # Classify all fake batch with D
            disc_on_fake_sample = camera_disc(generated_camera_sample.detach())
            disc_on_fake_sample = disc_on_fake_sample.view(-1)


            # Calculate D's loss on the all-fake batch
            fake_sample_error_disc = criterion(disc_on_fake_sample, label_fake)

            # Calculate the gradients for this batch
            fake_sample_error_disc.backward()

            camera_disc_fake_sample_output = disc_on_fake_sample.mean().item()


            sum_of_disc_errors = real_sample_error.item()+ fake_sample_error_disc.item()
            optimizer_camera_disc.step()



            ################################################################################
            #                           Second Sensor Discriminator
            ################################################################################

            ################################################################################
            #                           Second Sensor Generator
            ################################################################################

            if current_batch % 5 == 0:
                print(
                    '[%d/%d][%d/%d]\t\t Lidar to Cam GAN Loss_DF/R: %.4f/ %.4f = %.4f \t'
                    'Loss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f  '
                    % (epoch, config.TRAIN.MAX_EPOCH, current_batch, len(dataloader),
                       fake_sample_error_disc.item(), real_sample_error.item(), sum_of_disc_errors,
                       fake_sample_error_gen.item(), camera_disc_real_sample_output, camera_gen_output))

            camera_gen_loss.append(fake_sample_error_gen.item())
            camera_disc_loss.append(fake_sample_error_disc.item() + real_sample_error.item())


        with torch.no_grad():
            fakeCamera = camera_gen(lidar_sample.detach())
            example_camera_output.append(fakeCamera)
            np.savez_compressed(config.TRAIN.EXAMPLE_SAVE_PATH, data=example_camera_output[-1].cpu().numpy())

        plt.figure(figsize=(20, 14))
        plt.title("GAN Losses  During Training")
        plt.plot(camera_gen_loss, label="Generator Loss")
        plt.plot(camera_disc_loss, label="Discriminator Loss")

        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(config.TRAIN.GRAPH_SAVE_PATH+str(epoch))
        plt.close()
            # del camera_sample, lidar_sample
            # del label_real, label_fake
            # del output, generated_camera_sample


    #save_model(config, lidar_gen, camera_gen , lidar_disc , camera_disc, optimizer_lidar_gen,
    #           optimizer_camera_gen, optimizer_lidar_disc, optimizer_camera_disc )




def main(opts):
    load_config(opts.config)
    dataloader = lidar_camera_dataloader(config)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and config.NUM_GPUS > 0) else "cpu")
    train(dataloader,config, device)


if __name__ == "__main__":
    options, args = parser.parse_args()
    main(options)
