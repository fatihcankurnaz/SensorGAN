from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import optparse

from utils.data.Dataloader import lidar_camera_dataloader
from utils.core.config import config

from utils.core.config import load_config
from utils.helpers.helpers import save_vanilla_model
from utils.helpers.helpers import display_two_images


from Generator import Generator
from Discriminator import Discriminator, PixelDiscriminator


import torch.optim as optim
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt



parser = optparse.OptionParser()

parser.add_option('-c', '--config', dest="config",
                  help="load this config file", metavar="FILE")


# def lidar_error_calc(criterion, generated_lidar_sample, lidar_sample, weights):
#     loss = 0
#     # for i in range(0, 5):
#     #     temp_loss = criterion(generated_lidar_sample[:,i], lidar_sample[:,i])
#     #     print("i, " + str(i) + " loss "+str(temp_loss)+" new_loss "+str(temp_loss*weights[i]))
#     #     loss += temp_loss*weights[i]
#     loss = criterion(generated_lidar_sample, lidar_sample)
#
#     #loss /= weights.sum()
#     print("sum "+ str(loss))
#     return loss








def train(dataloader, config, device):
    lidar_gen_losses = []
    camera_gen_losses = []
    lidar_disc_losses = []
    camera_disc_losses = []
    cycle_loss = []

    # nn.BCEWithLogitsLoss(reduction='mean') # works better with log loss
    # lidar_weights = torch.tensor([
    #     1.0139991, 88.7063538, 410.3404964, 19369.19224, 22692.19007]).to(device=device, dtype=torch.float)
    # lidar generator_note softmax2d brokes either crossentropy or generator itself completely. Do not use it with lidar
    # log 2
    lidar_weights = torch.tensor([
        1.0139991,  6.4594316, 12.68067773024, 24.2414761695, 24.4699082327]).to(device=device, dtype=torch.float)
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
    real_sample_label = 1
    fake_sample_label = 0

    # each sensor should have their own Generator and Discriminator because their input size will probably not match
    lidar_gen = Generator(5, 5, config.NUM_GPUS).to(device)
    # camera_gen = Generator(5, 5, config.NUM_GPUS).to(device)
    lidar_disc = PixelDiscriminator(5, config.NUM_GPUS).to(device)
    # camera_disc = PixelDiscriminator(5, config.NUM_GPUS).to(device)

    # if (device.type == 'cuda') and (config.NUM_GPUS > 1):
    #
    #

    if (device.type == 'cuda') and (config.NUM_GPUS > 1):
        # camera_gen = nn.DataParallel(camera_gen, list(range(config.NUM_GPUS)))
        lidar_gen  = nn.DataParallel(lidar_gen, list(range(config.NUM_GPUS)))
        lidar_disc = nn.DataParallel(lidar_disc, list(range(config.NUM_GPUS)))
        # camera_disc = nn.DataParallel(camera_disc, list(range(config.NUM_GPUS)))



    # Setup Adam optimizers for both G and D
    optimizer_lidar_gen = optim.Adam(lidar_gen.parameters(), lr=config.LIDAR_GENERATOR.BASE_LR)
    lidar_gen_scheduler = optim.lr_scheduler.StepLR(optimizer_lidar_gen,
                                                    step_size=config.LIDAR_GENERATOR.STEP_SIZE,
                                                    gamma=config.LIDAR_GENERATOR.STEP_GAMMA)
    # optimizer_camera_gen = optim.Adam(camera_gen.parameters(), lr=config.CAMERA_GENERATOR.BASE_LR)
    # camera_gen_scheduler = optim.lr_scheduler.StepLR(optimizer_camera_gen,
    #                                                  step_size=config.CAMERA_GENERATOR.STEP_SIZE,
    #                                                 gamma=config.CAMERA_GENERATOR.STEP_GAMMA)
    optimizer_lidar_disc = optim.Adam(lidar_disc.parameters(), lr=config.LIDAR_DISCRIMINATOR.BASE_LR)
    lidar_disc_scheduler = optim.lr_scheduler.StepLR(optimizer_lidar_disc,
                                                     step_size=config.LIDAR_DISCRIMINATOR.STEP_SIZE,
                                                     gamma=config.LIDAR_DISCRIMINATOR.STEP_GAMMA)
    # optimizer_camera_disc = optim.Adam(camera_disc.parameters(), lr=config.CAMERA_DISCRIMINATOR.BASE_LR)
    # camera_disc_scheduler = optim.lr_scheduler.StepLR(optimizer_camera_disc,
    #                                                   step_size=config.CAMERA_DISCRIMINATOR.STEP_SIZE,
    #                                                   gamma=config.CAMERA_DISCRIMINATOR.STEP_GAMMA)
    test_lidar_path = "/home/fatih/Inputs/test/46cameraView_0000000000.npz"
    test_camera_path = "/home/fatih/Inputs/test/46segmented_0000000000.npz"

    test_lidar = torch.from_numpy(np.load(test_lidar_path)["data"].reshape(1, 5, 375, 1242)).to(device = device, dtype=torch.float)
    test_camera = torch.from_numpy(np.load(test_camera_path)["data"].reshape(1, 5, 375, 1242)).to(device = device, dtype=torch.float)
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

            label_real = Variable(torch.cuda.FloatTensor(np.ones((config.TRAIN.BATCH_SIZE, 1,23,77))), requires_grad=False)
            label_fake =  Variable(torch.cuda.FloatTensor(np.zeros((config.TRAIN.BATCH_SIZE, 1,23,77))), requires_grad=False)

            #display_two_images(data["camera_data"][0], data["lidar_data"][0])
            camera_sample = data["camera_data"].to(device = device, dtype=torch.float)
            lidar_sample = data["lidar_data"].to(device = device, dtype=torch.float)

            ################################################################################
            #                               Zero Gradients
            ################################################################################

            optimizer_lidar_gen.zero_grad()
            # optimizer_camera_gen.zero_grad()
            optimizer_lidar_disc.zero_grad()
            # optimizer_camera_disc.zero_grad()

            ###############################################################################
            #                          Camera Generator
            ###############################################################################
            #
            # camera_gen.zero_grad()
            #
            # # By using lidar samples as input we generate Camera data
            # generated_camera_sample = camera_gen(lidar_sample)
            #
            # camera_disc_on_generated = camera_disc(generated_camera_sample, lidar_sample)
            #
            # # disc_on_fake_sample_gen = disc_on_fake_sample_gen.view(-1)
            #
            # # generated_segmented_view = torch.max(generated_camera_sample, dim=1)
            # # real_segmented_view = torch.max(camera_sample, dim=1)
            # camera_gen_error = criterion_cam_to_lidar(camera_disc_on_generated, label_real )
            # camera_gen_error.backward()
            # optimizer_camera_gen.step()
            # # m Measure the generators capacity to trick discriminator
            # # fix after checking generator
            # # fake_sample_error_gen = criterion(disc_on_fake_sample_gen, label_real)
            # #
            # # # Calculate gradients for G
            # # fake_sample_error_gen.backward()
            # #
            # #
            # # camera_gen_output = disc_on_fake_sample_gen.mean().item()
            #
            #
            #
            #
            # ################################################################################
            # #                           Camera Discriminator
            # ################################################################################
            # # open after checking the generator
            # camera_disc.zero_grad()
            # #
            # camera_disc_real_output = camera_disc(camera_sample, lidar_sample)
            #
            #
            # camera_disc_real_error = criterion_cam_to_lidar(camera_disc_real_output, label_real)
            # #
            # # real_sample_error.backward()
            # #
            # # camera_disc_real_sample_output = real_data_output.mean().item()
            # #
            # # # Classify all fake batch with D
            # camera_disc_fake_output= camera_disc(generated_camera_sample.detach(), lidar_sample)
            #
            # #
            # #
            # # # Calculate D's loss on the all-fake batch
            # camera_disc_fake_error = criterion_cam_to_lidar(camera_disc_fake_output, label_fake)
            # camera_disc_total_error = camera_disc_fake_error + camera_disc_real_error
            # camera_disc_total_error.backward()
            # optimizer_camera_disc.step()
            # #
            # # # Calculate the gradients for this batch
            # # fake_sample_error_disc.backward()
            # #
            # # camera_disc_fake_sample_output = disc_on_fake_sample.mean().item()
            # #
            # #
            # # sum_of_disc_errors = real_sample_error.item()+ fake_sample_error_disc.item()
            # # optimizer_camera_disc.step()
            # if current_batch % 5 == 0:
            #     print(
            #         '[%d/%d][%d/%d]\t\t Lidar to Camera GAN Loss_D Real/Fake: %.4f/ %.4f = %.4f \t'
            #         'Loss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f  '
            #         % (epoch, config.TRAIN.MAX_EPOCH, current_batch, len(dataloader),
            #            camera_disc_real_error.item(), camera_disc_fake_error.item(), camera_disc_total_error,
            #            camera_gen_error.item(), camera_disc_real_output.mean().item(),
            #            camera_disc_fake_output.mean().item()))
            #
            # camera_gen_losses.append(camera_gen_error.item())
            # camera_disc_losses.append(camera_disc_total_error.item())
            #
            # if current_batch == 0:
            #     with torch.no_grad():
            #         fakeCamera = camera_gen(test_lidar.detach())
            #         #example_lidar_output.append(fakeLidar)
            #         np.savez_compressed(config.TRAIN.EXAMPLE_SAVE_PATH + str(epoch) + "_generated_",
            #                             data=fakeCamera[-1].cpu().numpy())
            #         np.savez_compressed(config.TRAIN.EXAMPLE_SAVE_PATH + str(epoch) + "_lidar_",
            #                             data=test_lidar[-1].cpu().numpy())
            #         np.savez_compressed(config.TRAIN.EXAMPLE_SAVE_PATH + str(epoch) + "_camera_",
            #                             data=test_camera[-1].cpu().numpy())
            #
            # del generated_camera_sample

            ################################################################################
            #                           Lidar Generator
            ################################################################################

            lidar_gen.zero_grad()

            # By using lidar samples as input we generate Camera data
            generated_lidar_sample = lidar_gen(camera_sample)
            #generated_lidar_with_weight = generated_lidar_sample * lidar_multiplier

            lidar_disc_on_generated = lidar_disc(generated_lidar_sample, camera_sample)


            # generated_segmented_view = torch.max(generated_camera_sample, dim=1)
            # real_segmented_view = torch.max(camera_sample, dim=1)
            # with torch.no_grad():
            #     new_version = torch.max(lidar_sample, dim=1)[1].view(config.TRAIN.BATCH_SIZE, -1, 1242)

            # m Measure the generators capacity to trick discriminator
            lidar_gen_error = criterion_cam_to_lidar(lidar_disc_on_generated, label_real)
            lidar_gen_error.backward()
            optimizer_lidar_gen.step()


            ################################################################################
            #                           Lidar Discriminator
            ################################################################################
            lidar_disc.zero_grad()
            lidar_disc_real_output = lidar_disc(lidar_sample, camera_sample)

            lidar_disc_real_error = criterion_cam_to_lidar(lidar_disc_real_output, label_real)
            # # Classify all fake batch with D
            lidar_disc_fake_output = lidar_disc(generated_lidar_sample.detach(), camera_sample)

            # # Calculate D's loss on the all-fake batch
            lidar_disc_fake_error = criterion_cam_to_lidar(lidar_disc_fake_output, label_fake)
            lidar_disc_total_error = lidar_disc_fake_error + lidar_disc_real_error
            lidar_disc_total_error.backward()
            optimizer_lidar_disc.step()


            if current_batch % 5 == 0:
                print(
                    '[%d/%d][%d/%d]\t\t Camera to Lidar GAN Loss_DF/R: %.4f/ %.4f = %.4f \t'
                    'Loss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f  '
                    % (epoch, config.TRAIN.MAX_EPOCH, current_batch, len(dataloader),
                       lidar_disc_real_error.item(), lidar_disc_fake_error.item(), lidar_disc_total_error,
                       lidar_gen_error.item(), lidar_disc_real_output.mean().item(),
                       lidar_disc_fake_output.mean().item()))

            lidar_gen_losses.append(lidar_gen_error.item())
            lidar_disc_losses.append(lidar_disc_total_error.item())

            if current_batch == 0:
                with torch.no_grad():
                    fakeLidar = lidar_gen(test_camera.detach())
                    #example_lidar_output.append(fakeLidar)
                    np.savez_compressed(config.TRAIN.EXAMPLE_SAVE_PATH + str(epoch) + "_generated_",
                                        data=fakeLidar[-1].cpu().numpy())
                    np.savez_compressed(config.TRAIN.EXAMPLE_SAVE_PATH + str(epoch) + "_lidar_",
                                        data=test_lidar[-1].cpu().numpy())
                    np.savez_compressed(config.TRAIN.EXAMPLE_SAVE_PATH + str(epoch) + "_camera_",
                                        data=test_camera[-1].cpu().numpy())


            del generated_lidar_sample

            del camera_sample, lidar_sample
            del label_real, label_fake


        # camera_gen_scheduler.step()
        # camera_disc_scheduler.step()
        lidar_gen_scheduler.step()
        lidar_disc_scheduler.step()

        plt.figure(figsize=(20, 14))
        plt.title("GAN Losses  During Training")
        plt.plot(lidar_gen_losses, label="Generator Loss")
        plt.plot(lidar_disc_losses, label="Discriminator Loss")

        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(config.TRAIN.GRAPH_SAVE_PATH+str(epoch))
        plt.close()



    #save_model(config, lidar_gen, camera_gen , lidar_disc , camera_disc, optimizer_lidar_gen,
    #           optimizer_camera_gen, optimizer_lidar_disc, optimizer_camera_disc )
        if epoch != 0 and epoch% config.TRAIN.SAVE_AT == 0 :
            print("Saving Model at ", epoch)
            save_vanilla_model(config, lidar_gen, lidar_disc,  optimizer_lidar_gen, optimizer_lidar_disc, epoch)




def main(opts):
    load_config(opts.config)
    dataloader = lidar_camera_dataloader(config)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and config.NUM_GPUS > 0) else "cpu")
    train(dataloader,config, device)


if __name__ == "__main__":
    options, args = parser.parse_args()
    main(options)
