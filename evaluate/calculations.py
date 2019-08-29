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


from utils.models.Generator import GeneratorLowParameter, Generator
from os import listdir
from os.path import join, isdir
import time

import torch.optim as optim
import torch.nn as nn
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
import utils.helpers.ssim as pytorch_ssim
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from skimage import color
from utils.helpers.runKITTIDataGeneratorForObjectDataset import processData
import utils.helpers.PC2ImageConverter as PC2ImageConverter
from utils.helpers.visualizer import  Vis
torch.manual_seed(0)
parser = optparse.OptionParser()
colors = ['black', 'green', 'yellow', 'red', 'blue']
parser.add_option('-c', '--config', dest="config",
                  help="load this config file", metavar="FILE")


def turn_back_to_oneD(data):
    torch_version = torch.from_numpy(data).view(1, 5, -1, 1242)
    new_version = torch.max(torch_version, dim=1)[1].view(-1,1242)

    return new_version.numpy()




def quantitative(config, device):
    transforms_ = [
        transforms.ToTensor(),
    ]
    im_transform = transforms.Compose(transforms_)

    mse = nn.MSELoss(reduction="mean")

    input_dir = config.TEST.INPUT_DIR


    multip = np.ones((5, 375, 1242))
    multip[0] = multip[0] * 0
    multip[1] = multip[1] * 1
    multip[2] = multip[2] * 1
    multip[3] = multip[3] * 1
    multip[4] = multip[4] * 1

    real_image_gen = Generator(1, 3, config.NUM_GPUS).to(device)
    segment_gen = Generator(5, 5, config.NUM_GPUS).to(device)
    baseline_gen = Generator(1, 3, config.NUM_GPUS).to(device)

    if (device.type == 'cuda') and (config.NUM_GPUS > 1):
        real_image_gen = nn.DataParallel(real_image_gen, list(range(config.NUM_GPUS)))
        segment_gen = nn.DataParallel(segment_gen, list(range(config.NUM_GPUS)))
        baseline_gen = nn.DataParallel(baseline_gen, list(range(config.NUM_GPUS)))

    print("loading previous pix2pix model")
    real_image_checkpoint = torch.load(config.TEST.LOAD_PIX2PIX_WEIGHTS)
    real_image_gen.load_state_dict(real_image_checkpoint['sensor1_gen'])
    real_image_gen.eval()
    print("done")

    print("loading previous lidar_to_cam model")
    segment_checkpoint = torch.load(config.TEST.LOAD_LIDAR_TO_CAM_WEIGHTS)
    segment_gen.load_state_dict(segment_checkpoint['sensor1_gen'])
    segment_gen.eval()
    print("done")

    print("loading baseline model")
    baseline_checkpoint = torch.load(config.TEST.LOAD_BASELINE_WEIGHTS)
    baseline_gen.load_state_dict(baseline_checkpoint['sensor1_gen'])
    baseline_gen.eval()
    print("done")
    overall_sum_pipeline = 0
    overall_sum_baseline = 0
    overall_image_count = 0
    overall_psnr_pipeline = 0
    overall_psnr_baseline = 0

    for dir in sorted(listdir(input_dir)):
        segmented_lidar_root = join(input_dir, dir+"/lidar")
        rgb_root = join(config.TEST.RGB_ROOT, dir + "/image_02/data")
        sum_pipeline = 0
        sum_baseline = 0
        image_count = 0
        psnr_pipeline = 0
        psnr_baseline = 0
        for file in sorted(listdir(segmented_lidar_root)):
            print(file)
            image_count += 1
            overall_image_count += 1
            segmented_lidar_path = join(segmented_lidar_root, file)
            segmented_lidar_numpy = np.load(segmented_lidar_path)["data"].reshape(5, 375, 1242)
            segmented_lidar_torch = torch.from_numpy(segmented_lidar_numpy).to(device=device, dtype=torch.float).\
                view(1, 5, 375, 1242)

            expected_rgb_path = join(rgb_root, file.split(".")[0].split("_")[1]+ ".png")
            expected_rgb = Image.open(expected_rgb_path)

            segmented_lidar_baseline_input = np.load(segmented_lidar_path)["data"].reshape(5, 375, 1242) * multip
            segmented_lidar_baseline_input = np.sum(segmented_lidar_baseline_input, axis=0).reshape(375, 1242)
            segmented_lidar_baseline_input = \
                torch.from_numpy(segmented_lidar_baseline_input).to(device=device, dtype=torch.float).view(1, 1, 375, 1242)

            generated_cam_segment = segment_gen(segmented_lidar_torch)
            generated_cam_segment = generated_cam_segment.detach().cpu().numpy()
            generated_cam_segment = turn_back_to_oneD(generated_cam_segment)
            generated_rgb_input = torch.from_numpy(generated_cam_segment).to(device=device,
                                                                             dtype=torch.float).view(1, 1, 375, 1242)

            generated_rgb_from_pipeline = real_image_gen(generated_rgb_input)

            generated_rgb_from_pipeline = generated_rgb_from_pipeline.detach().cpu()
            save_image(generated_rgb_from_pipeline, config.TEST.RESULT_SAVE_PATH+"/pipeline_rgb.png", normalize=True)

            generated_rgb_from_pipeline = Image.open(config.TEST.RESULT_SAVE_PATH+"/pipeline_rgb.png")

            generated_rgb_from_baseline = baseline_gen(segmented_lidar_baseline_input)

            generated_rgb_from_baseline = generated_rgb_from_baseline.detach().cpu()
            save_image(generated_rgb_from_baseline, config.TEST.RESULT_SAVE_PATH+"/baseline_rgb.png",
                       normalize=True)

            generated_rgb_from_baseline = Image.open(config.TEST.RESULT_SAVE_PATH+"/baseline_rgb.png")

            generated_rgb_from_pipeline = im_transform(generated_rgb_from_pipeline).to(device=device, dtype=torch.float).\
                view(1, 3, 375, 1242)
            generated_rgb_from_baseline = im_transform(generated_rgb_from_baseline).to(device=device, dtype=torch.float).\
                view(1, 3, 375, 1242)

            expected_rgb = im_transform(expected_rgb).to(device=device, dtype=torch.float). \
                view(1, 3, 375, 1242)

            overall_sum_pipeline += pytorch_ssim.ssim(expected_rgb, generated_rgb_from_pipeline).item()
            overall_sum_baseline += pytorch_ssim.ssim(expected_rgb, generated_rgb_from_baseline).item()
            overall_psnr_pipeline += (10 * torch.log10(1 / mse(expected_rgb, generated_rgb_from_pipeline))).item()
            overall_psnr_baseline += (10 * torch.log10(1 / mse(expected_rgb, generated_rgb_from_baseline))).item()

            sum_pipeline += pytorch_ssim.ssim(expected_rgb, generated_rgb_from_pipeline).item()
            sum_baseline += pytorch_ssim.ssim(expected_rgb, generated_rgb_from_baseline).item()
            psnr_pipeline += (10*torch.log10(1 / mse(expected_rgb, generated_rgb_from_pipeline))).item()
            psnr_baseline += (10*torch.log10(1 / mse(expected_rgb, generated_rgb_from_baseline))).item()
            print("Pipeline SSIM : ", pytorch_ssim.ssim(expected_rgb, generated_rgb_from_pipeline))
            print("Baseline SSIM : ", pytorch_ssim.ssim(expected_rgb, generated_rgb_from_baseline))

            print("Pipeline PSNR : ", (10*torch.log10(1 / mse(expected_rgb, generated_rgb_from_pipeline))).item())
            print("Baseline PSNR : ", (10*torch.log10(1 / mse(expected_rgb, generated_rgb_from_baseline))).item())
        print("Dir : ", dir)
        print("Average Pipeline SSIM : ", sum_pipeline / image_count)
        print("Average Baseline SSIM : ", sum_baseline / image_count)
        print("Average Pipeline PSNR : ", psnr_pipeline / image_count)
        print("Average Baseline PSNR : ", psnr_baseline / image_count)

    print("Average Pipeline SSIM : ", overall_sum_pipeline / overall_image_count)
    print("Average Baseline SSIM : ", overall_sum_baseline / overall_image_count)
    print("Average Pipeline PSNR : ", overall_psnr_pipeline / overall_image_count)
    print("Average Baseline PSNR : ", overall_psnr_baseline / overall_image_count)



def main(opts):

    load_config(opts.config)


    device = torch.device("cuda:0" if (torch.cuda.is_available() and config.NUM_GPUS > 0) else "cpu")
    quantitative(config, device)


if __name__ == "__main__":
    options, args = parser.parse_args()
    main(options)

