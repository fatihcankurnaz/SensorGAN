import os

import torch
import torch.optim as optim
from ignite.engine import Engine
from torch import nn
from torchvision.utils import save_image

from models.Discriminator import PixelDiscriminator
from models.Generator import Generator


def init(loader, dataset, config):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    criterion_gan = nn.MSELoss(reduction=config.TRAIN.DISCRIMINATOR_CRITERION_REDUCTION)
    criterion_pixel = nn.L1Loss(reduction=config.TRAIN.DISCRIMINATOR_CRITERION_REDUCTION)

    if config.MODEL == 'baseline' or config.MODEL == 'pix2pix':
        camera_gen = Generator(1, 3).cuda()
        camera_disc = PixelDiscriminator(3, 1).cuda()
    else:
        camera_gen = Generator(5, 5).cuda()
        camera_disc = PixelDiscriminator(5, 5).cuda()

    if len(config.NUM_GPUS) > 1:
        camera_gen = nn.DataParallel(camera_gen, config.NUM_GPUS).cuda()
        camera_disc = nn.DataParallel(camera_disc, config.NUM_GPUS).cuda()

    optimizer_camera_gen = optim.Adam(camera_gen.parameters(), lr=config.CAMERA_GENERATOR.BASE_LR,
                                      betas=(config.TRAIN.BETA1, config.TRAIN.BETA2))

    camera_gen_scheduler = optim.lr_scheduler.StepLR(optimizer_camera_gen,
                                                     step_size=config.CAMERA_GENERATOR.STEP_SIZE,
                                                     gamma=config.CAMERA_GENERATOR.STEP_GAMMA)

    optimizer_camera_disc = optim.Adam(camera_disc.parameters(), lr=config.CAMERA_DISCRIMINATOR.BASE_LR,
                                       betas=(config.TRAIN.BETA1, config.TRAIN.BETA2))

    camera_disc_scheduler = optim.lr_scheduler.StepLR(optimizer_camera_disc,
                                                      step_size=config.CAMERA_DISCRIMINATOR.STEP_SIZE,
                                                      gamma=config.CAMERA_DISCRIMINATOR.STEP_GAMMA)

    x1, y1, x2, y2, label_real, label_fake = dataset.get_test()

    #############################################
    #           Training Function               #
    #############################################
    def step(engine, batch):
        x, y = batch['x'], batch['y']
        x = x.type(torch.float).cuda()
        y = y.type(torch.float).cuda()

        optimizer_camera_disc.zero_grad()
        optimizer_camera_gen.zero_grad()

        camera_gen.zero_grad()

        #############################################
        #           Camera Generator                #
        #############################################

        generated_camera_sample = camera_gen(x)
        camera_disc_on_generated = camera_disc(generated_camera_sample, x)

        camera_gen_loss_disc = criterion_gan(camera_disc_on_generated, label_real)
        camera_gen_loss_pixel = criterion_pixel(generated_camera_sample, y)
        camera_gen_loss_pixel = config.CAMERA_GENERATOR.PIXEL_LAMBDA * camera_gen_loss_pixel

        if config.MODEL == 'lidar2cam_new_loss':
            lidar_mask = y * generated_camera_sample.detach()
            camera_lidar_point_loss = criterion_pixel(lidar_mask, generated_camera_sample)
            camera_lidar_point_loss = config.CAMERA_GENERATOR.NEW_LOSS_LAMBDA * camera_lidar_point_loss
            camera_gen_loss_pixel = camera_gen_loss_pixel + camera_lidar_point_loss

        camera_gen_loss = camera_gen_loss_pixel + camera_gen_loss_disc
        camera_gen_loss.backward()
        optimizer_camera_gen.step()

        #############################################
        #           Camera Discriminator            #
        #############################################

        camera_disc.zero_grad()
        camera_disc_real_output = camera_disc(y, x)
        camera_disc_real_loss = criterion_gan(camera_disc_real_output, label_real - 0.1)
        camera_disc_fake_output = camera_disc(generated_camera_sample.detach(), x)
        camera_disc_fake_loss = criterion_gan(camera_disc_fake_output, label_fake)
        camera_disc_loss = (camera_disc_fake_loss + camera_disc_real_loss) * 0.5
        camera_disc_loss.backward()
        optimizer_camera_disc.step()

        return {'Real_D': camera_disc_real_loss.item(),
                'Fake_D': camera_disc_fake_loss.item(),
                'Tot_D': camera_disc_loss.item(),
                'GAN_G': camera_gen_loss_disc.item(),
                'Pixel_G': camera_gen_loss_pixel.item(),
                'Tot_G': camera_gen_loss.item(),
                'D': camera_disc_real_output.mean().item(),
                'D_G': camera_disc_real_loss.mean().item()}

    trainer = Engine(step)

    ret_objs = dict()
    ret_objs['trainer'] = trainer
    ret_objs['config'] = config
    ret_objs['loader'] = loader
    ret_objs['camera_gen_scheduler'] = camera_gen_scheduler
    ret_objs['camera_disc_scheduler'] = camera_disc_scheduler
    ret_objs['x1'] = x1
    ret_objs['y1'] = y1
    ret_objs['x2'] = x2
    ret_objs['y2'] = y2
    ret_objs['camera_gen'] = camera_gen
    ret_objs['camera_disc'] = camera_disc
    ret_objs['optimizer_camera_gen'] = optimizer_camera_gen
    ret_objs['optimizer_camera_disc'] = optimizer_camera_disc

    return ret_objs
