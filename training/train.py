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
from Discriminator import PixelDiscriminator


import torch.optim as optim
import torch.nn as nn
import torch
from torch.autograd import Variable
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
    generator_losses = []
    cycle_losses = []


    # nn.BCEWithLogitsLoss(reduction='mean') # works better with log loss
    # lidar_weights = torch.tensor([
    #     1.0139991, 88.7063538, 410.3404964, 19369.19224, 22692.19007]).to(device=device, dtype=torch.float)
    # lidar generator_note softmax2d brokes either crossentropy or generator itself completely. Do not use it with lidar
    # log 2
    lidar_weights = torch.tensor([
        1.0139991, 6.4594316, 15.68067773024, 30.2414761695, 30.4699082327]).to(device=device, dtype=torch.float)
    lidar_multiplier = torch.ones(config.TRAIN.BATCH_SIZE, 5, 375, 1242).to(device=device, dtype=torch.float)
    for i in range(config.TRAIN.BATCH_SIZE):
        lidar_multiplier[i][0] = torch.ones(375, 1242) * lidar_weights[0]
        lidar_multiplier[i][1] = torch.ones(375, 1242) * lidar_weights[1]
        lidar_multiplier[i][2] = torch.ones(375, 1242) * lidar_weights[2]
        lidar_multiplier[i][3] = torch.ones(375, 1242) * lidar_weights[3]
        lidar_multiplier[i][4] = torch.ones(375, 1242) * lidar_weights[4]

    # camera_weights = torch.tensor([
    #     1.4859513, 3.9798364, 13.7709121, 483.6851552, 926.3148902]).to(device=device, dtype=torch.float)
    # log2
    # camera_weights = torch.tensor([
    #     1.0, 1.9927091, 3.7835522, 8.9179244, 9.8553588]).to(device=device, dtype=torch.float)
    # criterion = nn.CrossEntropyLoss( reduction=config.TRAIN.DISCRIMINATOR_CRITERION_REDUCTION)
    # criterion = nn.BCELoss(reduction=config.TRAIN.DISCRIMINATOR_CRITERION_REDUCTION)
    criterion_cam_to_lidar = nn.MSELoss(reduction=config.TRAIN.DISCRIMINATOR_CRITERION_REDUCTION)
    criterion_pixel = nn.L1Loss(reduction=config.TRAIN.DISCRIMINATOR_CRITERION_REDUCTION)
    criterion_cycle = nn.L1Loss(reduction=config.TRAIN.DISCRIMINATOR_CRITERION_REDUCTION)
    real_sample_label = 1
    fake_sample_label = 0

    # each sensor should have their own Generator and Discriminator because their input size will probably not match
    lidar_gen = Generator(5, 5, config.NUM_GPUS).to(device)
    camera_gen = Generator(5, 5, config.NUM_GPUS).to(device)
    lidar_disc = PixelDiscriminator(5, config.NUM_GPUS).to(device)
    camera_disc = PixelDiscriminator(5, config.NUM_GPUS).to(device)

    if (device.type == 'cuda') and (config.NUM_GPUS > 1):
        camera_gen = nn.DataParallel(camera_gen, list(range(config.NUM_GPUS)))
        lidar_gen  = nn.DataParallel(lidar_gen, list(range(config.NUM_GPUS)))
        lidar_disc = nn.DataParallel(lidar_disc, list(range(config.NUM_GPUS)))
        camera_disc = nn.DataParallel(camera_disc, list(range(config.NUM_GPUS)))

    # Setup Adam optimizers for both G and D
    optimizer_lidar_gen = optim.Adam(lidar_gen.parameters(), lr=config.LIDAR_GENERATOR.BASE_LR,
                                     betas=(config.TRAIN.BETA1, config.TRAIN.BETA2))
    lidar_gen_scheduler = optim.lr_scheduler.StepLR(optimizer_lidar_gen,
                                                    step_size=config.LIDAR_GENERATOR.STEP_SIZE,
                                                    gamma=config.LIDAR_GENERATOR.STEP_GAMMA)
    optimizer_camera_gen = optim.Adam(camera_gen.parameters(), lr=config.CAMERA_GENERATOR.BASE_LR)
    camera_gen_scheduler = optim.lr_scheduler.StepLR(optimizer_camera_gen,
                                                     step_size=config.CAMERA_GENERATOR.STEP_SIZE,
                                                     gamma=config.CAMERA_GENERATOR.STEP_GAMMA)
    optimizer_lidar_disc = optim.Adam(lidar_disc.parameters(), lr=config.LIDAR_DISCRIMINATOR.BASE_LR,
                                      betas=(config.TRAIN.BETA1, config.TRAIN.BETA2))
    lidar_disc_scheduler = optim.lr_scheduler.StepLR(optimizer_lidar_disc,
                                                     step_size=config.LIDAR_DISCRIMINATOR.STEP_SIZE,
                                                     gamma=config.LIDAR_DISCRIMINATOR.STEP_GAMMA)
    optimizer_camera_disc = optim.Adam(camera_disc.parameters(), lr=config.CAMERA_DISCRIMINATOR.BASE_LR)
    camera_disc_scheduler = optim.lr_scheduler.StepLR(optimizer_camera_disc,
                                                      step_size=config.CAMERA_DISCRIMINATOR.STEP_SIZE,
                                                      gamma=config.CAMERA_DISCRIMINATOR.STEP_GAMMA)
    test_lidar_path1 = "/home/fatih/Inputs/test/46cameraView_0000000000.npz"
    test_camera_path1 = "/home/fatih/Inputs/test/46segmented_0000000000.npz"

    test_lidar_path2 = "/home/fatih/Inputs/test/01cameraView_0000000000.npz"
    test_camera_path2 = "/home/fatih/Inputs/test/01segmented_0000000000.npz"

    test_lidar1 = torch.from_numpy(np.load(test_lidar_path1)["data"].reshape(1, 5, 375, 1242)).to(device=device,
                                                                                                  dtype=torch.float)
    test_camera1 = torch.from_numpy(np.load(test_camera_path1)["data"].reshape(1, 5, 375, 1242)).to(device=device,
                                                                                                    dtype=torch.float)

    test_lidar2 = torch.from_numpy(np.load(test_lidar_path2)["data"].reshape(1, 5, 375, 1242)).to(device=device,
                                                                                                  dtype=torch.float)
    test_camera2 = torch.from_numpy(np.load(test_camera_path2)["data"].reshape(1, 5, 375, 1242)).to(device=device,
                                                                                                    dtype=torch.float)
    # camera_gen_total_params = sum(p.numel() for p in camera_gen.parameters())
    # print("Camera Generator ", camera_gen_total_params)

    # camera_disch_total_params = sum(p.numel() for p in camera_disc.parameters())
    # print("Camera Discriminator ", camera_disch_total_params)

    example_camera_output = []

    # if config.TRAIN.START_EPOCH > 0:
    #     print("loading previous model")
    #     checkpoint = torch.load(config.TRAIN.LOAD_WEIGHTS)
    #     camera_gen.load_state_dict(checkpoint['camera_gen'])
    #     camera_disc.load_state_dict(checkpoint['camera_disc'])
    #     optimizer_camera_gen.load_state_dict(checkpoint['optimizer_camera_gen'])
    #     optimizer_camera_disc.load_state_dict(checkpoint['optimizer_camera_disc'])
    #     camera_gen.train()
    #     camera_disc.train()
    #     print("done")

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.MAX_EPOCH):
        for current_batch, data in enumerate(dataloader, 0):
            if len(dataloader) - current_batch< config.TRAIN.BATCH_SIZE:
                continue

            label_real = Variable(torch.cuda.FloatTensor(np.ones((config.TRAIN.BATCH_SIZE, 1, 23, 77))),
                                  requires_grad=False)
            label_fake = Variable(torch.cuda.FloatTensor(np.zeros((config.TRAIN.BATCH_SIZE, 1, 23, 77))),
                                  requires_grad=False)

            # display_two_images(data["camera_data"][0], data["lidar_data"][0])
            camera_sample = data["camera_data"].to(device=device, dtype=torch.float)
            lidar_sample = data["lidar_data"].to(device=device, dtype=torch.float)

            ################################################################################
            #                               Zero Gradients
            ################################################################################

            optimizer_lidar_gen.zero_grad()
            optimizer_camera_gen.zero_grad()
            optimizer_lidar_disc.zero_grad()
            optimizer_camera_disc.zero_grad()

            ###############################################################################
            #                          Generators
            ###############################################################################

            camera_gen.zero_grad()
            lidar_gen.zero_grad()

            generated_camera_sample = camera_gen(lidar_sample)
            generated_lidar_sample = lidar_gen(camera_sample)

            cycled_camera_sample = camera_gen(generated_lidar_sample)
            cycled_lidar_sample = lidar_gen(generated_camera_sample)

            camera_cycle_error = criterion_cycle(camera_sample, cycled_camera_sample)
            lidar_cycle_error = criterion_cycle(lidar_sample, cycled_lidar_sample)

            total_cycle_error = camera_cycle_error + lidar_cycle_error

            camera_disc_on_generated = camera_disc(generated_camera_sample, lidar_sample)
            lidar_disc_on_generated = lidar_disc(generated_lidar_sample, camera_sample)

            camera_gen_error_disc = criterion_cam_to_lidar(camera_disc_on_generated, label_real)
            lidar_gen_error_disc = criterion_cam_to_lidar(lidar_disc_on_generated, label_real)

            generated_lidar_with_weight = generated_lidar_sample * lidar_multiplier
            real_lidar_with_weight = lidar_sample * lidar_multiplier

            camera_gen_error_pixel = criterion_pixel(generated_camera_sample, camera_sample)
            lidar_gen_error_pixel = criterion_pixel(generated_lidar_with_weight, real_lidar_with_weight)

            camera_gen_error_pixel = config.CAMERA_GENERATOR.PIXEL_LAMBDA * camera_gen_error_pixel
            lidar_gen_error = config.LIDAR_GENERATOR.PIXEL_LAMBDA * lidar_gen_error_pixel

            camera_gen_error = camera_gen_error_disc + camera_gen_error_pixel
            lidar_gen_error = lidar_gen_error_disc + lidar_gen_error

            cycle_loss = config.TRAIN.CYCLE_LAMBDA * total_cycle_error

            generator_loss = camera_gen_error + lidar_gen_error + cycle_loss
            generator_loss.backward()

            optimizer_lidar_gen.step()
            optimizer_camera_gen.step()




            ################################################################################
            #                           Camera Discriminator
            ################################################################################

            camera_disc.zero_grad()
            camera_disc_real_output = camera_disc(camera_sample, lidar_sample)
            camera_disc_real_error = criterion_cam_to_lidar(camera_disc_real_output, label_real)

            camera_disc_fake_output = camera_disc(generated_camera_sample.detach(), lidar_sample)
            camera_disc_fake_error = criterion_cam_to_lidar(camera_disc_fake_output, label_fake)

            camera_disc_total_error = camera_disc_fake_error + camera_disc_real_error
            camera_disc_total_error.backward()
            optimizer_camera_disc.step()



            ################################################################################
            #                           Lidar Discriminator
            ################################################################################

            lidar_disc.zero_grad()
            lidar_disc_real_output = lidar_disc(lidar_sample, camera_sample)
            lidar_disc_real_error = criterion_cam_to_lidar(lidar_disc_real_output, label_real)

            lidar_disc_fake_output = lidar_disc(generated_lidar_sample.detach(), camera_sample)
            lidar_disc_fake_error = criterion_cam_to_lidar(lidar_disc_fake_output, label_fake)

            lidar_disc_total_error = lidar_disc_fake_error + lidar_disc_real_error
            lidar_disc_total_error.backward()
            optimizer_lidar_disc.step()

            if current_batch % 5 == 0:
                print(
                    '[%d/%d][%d/%d]'
                    % (epoch, config.TRAIN.MAX_EPOCH, current_batch, len(dataloader)))
                print(
                    'Camera to Lidar GAN Loss_D R/F: %.4f/%.4f \t Tot = %.4f  \t '
                    'Loss_G: %.4f \t PixelError: %.4f\t LidarCycleError: %.4f '
                    % (lidar_disc_real_error.item(), lidar_disc_fake_error.item(),
                       lidar_disc_total_error.item(), lidar_gen_error.item(), lidar_gen_error_pixel.item(),
                       lidar_cycle_error.item()))
                print(
                    'Lidar to Camera GAN Loss_D R/F: %.4f/%.4f \t Tot = %.4f  \t '
                    'Loss_G: %.4f \t PixelError: %.4f\t CameraCycleError: %.4f '
                    % (camera_disc_real_error.item(), camera_disc_fake_error.item(),
                       camera_disc_total_error.item(), camera_gen_error.item(), camera_gen_error_pixel.item(),
                       camera_cycle_error.item()))
                print(
                    'Cycle Error Total = %.4f \t  Generator Loss = %.4f'
                    % (cycle_loss.item(), generator_loss.item())
                )

            lidar_gen_losses.append(lidar_gen_error.item())
            lidar_disc_losses.append(lidar_disc_total_error.item())
            camera_gen_losses.append(camera_gen_error.item())
            camera_disc_losses.append(camera_disc_total_error.item())
            generator_losses.append(generator_loss.item())
            cycle_losses.append(cycle_loss.item())

            if current_batch == 0:
                with torch.no_grad():
                    fake_lidar1 = lidar_gen(test_camera1.detach())
                    fake_camera1 = camera_gen(test_lidar1.detach())

                    reconst_camera1 = camera_gen(fake_lidar1.detach())
                    reconst_lidar1 = lidar_gen(fake_camera1.detach())

                    np.savez_compressed(config.TRAIN.EXAMPLE_SAVE_PATH + str(epoch) + "_generated_lidar_1",
                                        data=fake_lidar1[-1].cpu().numpy())
                    np.savez_compressed(config.TRAIN.EXAMPLE_SAVE_PATH + str(epoch) + "_generated_camera_1",
                                        data=fake_camera1[-1].cpu().numpy())
                    np.savez_compressed(config.TRAIN.EXAMPLE_SAVE_PATH + str(epoch) + "_reconstructed_lidar_1",
                                        data=reconst_lidar1[-1].cpu().numpy())
                    np.savez_compressed(config.TRAIN.EXAMPLE_SAVE_PATH + str(epoch) + "_reconstructed_camera_1",
                                        data=reconst_camera1[-1].cpu().numpy())
                    np.savez_compressed(config.TRAIN.EXAMPLE_SAVE_PATH + str(epoch) + "_lidar_1",
                                        data=test_lidar1[-1].cpu().numpy())
                    np.savez_compressed(config.TRAIN.EXAMPLE_SAVE_PATH + str(epoch) + "_camera_1",
                                        data=test_camera1[-1].cpu().numpy())
                    fake_lidar2 = lidar_gen(test_camera2.detach())
                    fake_camera2 = camera_gen(test_lidar2.detach())

                    reconst_camera2 = camera_gen(fake_lidar2.detach())
                    reconst_lidar2 = lidar_gen(fake_camera2.detach())
                    np.savez_compressed(config.TRAIN.EXAMPLE_SAVE_PATH + str(epoch) + "_generated_lidar_2",
                                        data=fake_lidar2[-1].cpu().numpy())
                    np.savez_compressed(config.TRAIN.EXAMPLE_SAVE_PATH + str(epoch) + "_generated_camera_2",
                                        data=fake_camera2[-1].cpu().numpy())
                    np.savez_compressed(config.TRAIN.EXAMPLE_SAVE_PATH + str(epoch) + "_reconstructed_lidar_2",
                                        data=reconst_lidar2[-1].cpu().numpy())
                    np.savez_compressed(config.TRAIN.EXAMPLE_SAVE_PATH + str(epoch) + "_reconstructed_camera_2",
                                        data=reconst_camera2[-1].cpu().numpy())
                    np.savez_compressed(config.TRAIN.EXAMPLE_SAVE_PATH + str(epoch) + "_lidar_2",
                                        data=test_lidar2[-1].cpu().numpy())
                    np.savez_compressed(config.TRAIN.EXAMPLE_SAVE_PATH + str(epoch) + "_camera_2",
                                        data=test_camera2[-1].cpu().numpy())

            del generated_lidar_sample
            del generated_camera_sample
            del camera_sample, lidar_sample
            del label_real, label_fake

        camera_gen_scheduler.step()
        camera_disc_scheduler.step()
        lidar_gen_scheduler.step()
        lidar_disc_scheduler.step()

        plt.figure(figsize=(20, 14))
        plt.title("Generator Losses  During Training")
        plt.plot(lidar_gen_losses, label="Lidar Generator Loss")
        plt.plot(camera_gen_losses, label="Camera Generator Loss")
        plt.plot(cycle_losses, label="Cycle Loss")
        plt.plot(generator_losses, label="Total Generator Loss")

        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(config.TRAIN.GRAPH_SAVE_PATH+str(epoch)+"Generator")
        plt.close()

        plt.figure(figsize=(20, 14))
        plt.title("Discriminator Losses  During Training")
        plt.plot(lidar_disc_losses, label="Lidar Generator Loss")
        plt.plot(camera_disc_losses, label="Camera Generator Loss")

        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(config.TRAIN.GRAPH_SAVE_PATH + str(epoch)+"Discriminator")
        plt.close()

        if epoch != 0 and epoch % config.TRAIN.SAVE_AT == 0:
            print("Saving Model at ", epoch)
            save_model(config, lidar_gen, camera_gen, lidar_disc, camera_disc, optimizer_lidar_gen,
                       optimizer_camera_gen, optimizer_lidar_disc, optimizer_camera_disc, epoch)


def main(opts):
    load_config(opts.config)
    dataloader = lidar_camera_dataloader(config)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and config.NUM_GPUS > 0) else "cpu")
    train(dataloader, config, device)


if __name__ == "__main__":
    options, args = parser.parse_args()
    main(options)
