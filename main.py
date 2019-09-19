import optparse
import os

from torchvision import transforms

from configs.config import *
from training import trainer
from utils.core.engine_decorators import attach_decorators
from utils.data.Dataloader import *
from utils.scripts.display_generated import display
from utils.evaluate import eval

if __name__ == '__main__':
    transforms_ = [transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transforms_seg = [transforms.ToTensor()]

    parser = optparse.OptionParser()

    parser.add_option('-c', '--config', dest="config",
                      help="load this config file", metavar="FILE")
    parser.add_option('-e', '--eval-epoch', type='int', default=0)
    parser.add_option('-d', '--display', action="store_true", default=False)
    parser.add_option('-o', '--one', dest="num", action="store", type="int",
                      help="display only one output", default=0)
    parser.add_option('-p', '--path', dest="path", action="store", type="string",
                      help="path to the files and no ending \"/\" ", default='')
    parser.add_option('-i', '--interval', dest="interval", action="store", type="string",
                      help="give an interval of numbers ex 1-14", default='')
    parser.add_option('-s', '--save', dest="save", action="store_true", default=False,
                      help="save the generated visual, if this is selected nothing will be shown")
    parser.add_option('-v', '--verbose', dest="verbose", action="store_true", default=False,
                      help="print more information")
    parser.add_option('-r', '--continue-epoch', type='int', default=0)
    parser.add_option('-n', '--model-name', type="string", default='')

    options, args = parser.parse_args()
    load_config(options.config)

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config.NUM_GPUS))
    config.NUM_GPUS = list(map(int, config.NUM_GPUS))
    config.TRAIN.CONTINUE = options.continue_epoch

    if options.eval_epoch == 0 and not options.display:
        config.TEST.EVAL_EPOCH = options.eval_epoch
        dataloader, dataset = lidar_camera_dataloader(config, transforms_, transforms_seg)
        ret_objs = trainer.init(dataloader, dataset, config)
        attach_decorators(**ret_objs)
        ret_objs['trainer'].run(ret_objs['loader'], config.TRAIN.MAX_EPOCH)
    else:
        if options.display:
            display(options)
        else:
            config.TEST.EVAL_EPOCH = options.eval_epoch
            if options.model_name != '':
                config.MODEL = options.model_name
            eval.main(config)


