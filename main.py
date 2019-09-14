import optparse
import os

from torchvision import transforms

from configs.config import *
from training import trainer
from utils.core.engine_decorators import attach_decorators
from utils.data.Dataloader import *

if __name__ == '__main__':
    transforms_ = [transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transforms_seg = [transforms.ToTensor()]

    parser = optparse.OptionParser()

    parser.add_option('-c', '--config', dest="config",
                      help="load this config file", metavar="FILE")

    parser.add_option('-e', '--eval-epoch', type='int', default=0)

    options, args = parser.parse_args()
    load_config(options.config)

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config.NUM_GPUS))

    if options.eval_epoch == 0:
        config.NUM_GPUS = list(map(int, config.NUM_GPUS))
        dataloader, dataset = lidar_camera_dataloader(config, transforms_, transforms_seg)
        ret_objs = trainer.init(dataloader, dataset, config)
        attach_decorators(**ret_objs)
        ret_objs['trainer'].run(ret_objs['loader'], config.TRAIN.MAX_EPOCH)
    else:
        #TODO eval case
        pass
