import os
from torchvision.utils import save_image
import numpy as np
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events
from ignite.handlers import Timer, ModelCheckpoint
from ignite.metrics import RunningAverage


def attach_decorators(trainer, config, loader, camera_gen_scheduler, camera_disc_scheduler, x1, y1, x2, y2,
                      camera_gen, camera_disc, optimizer_camera_gen, optimizer_camera_disc):
    @trainer.on(Events.EPOCH_COMPLETED)
    def plot_graphs(engine):
        import pandas as pd
        import matplotlib.pyplot as plt
        if not os.path.exists(config.OUTPUT_DIR + '/' + config.MODEL + config.TRAIN.GRAPH_SAVE_PATH):
            os.makedirs(config.OUTPUT_DIR + '/' + config.MODEL + config.TRAIN.GRAPH_SAVE_PATH)
        df = pd.read_csv(os.path.join(config.OUTPUT_DIR + '/' + config.MODEL, config.MODEL), delimiter='\t')
        df = df[['D', 'D_G']]
        _ = df.plot(subplots=True, figsize=(20, 20))
        _ = plt.xlabel('Iteration number')
        fig = plt.gcf()
        path = os.path.join(config.OUTPUT_DIR + '/' + config.MODEL + config.TRAIN.GRAPH_SAVE_PATH,
                            config.MODEL + str(engine.state.epoch))
        fig.savefig(path)

    @trainer.on(Events.EPOCH_COMPLETED)
    def scheduler_step(engine):
        camera_gen_scheduler.step()
        camera_disc_scheduler.step()
        pbar.log_message("Scheduler step: G:{}\tD:{}\n".format(camera_gen_scheduler.get_lr(),
                                                               camera_disc_scheduler.get_lr()))

    timer = Timer(average=True)

    checkpoint_handler = ModelCheckpoint(config.OUTPUT_DIR + '/' + config.MODEL + config.TRAIN.SAVE_WEIGHTS, 'training',
                                         save_interval=1, n_saved=config.TRAIN.MAX_EPOCH, require_empty=False)

    monitoring_metrics = ['Real_D', 'Fake_D', 'Tot_D', 'GAN_G', 'Pixel_G', 'Tot_G', 'D', 'D_G']

    RunningAverage(alpha=0.98, output_transform=lambda x: x['Real_D']).attach(trainer, 'Real_D')
    RunningAverage(alpha=0.98, output_transform=lambda x: x['Fake_D']).attach(trainer, 'Fake_D')
    RunningAverage(alpha=0.98, output_transform=lambda x: x['Tot_D']).attach(trainer, 'Tot_D')
    RunningAverage(alpha=0.98, output_transform=lambda x: x['GAN_G']).attach(trainer, 'GAN_G')
    RunningAverage(alpha=0.98, output_transform=lambda x: x['Pixel_G']).attach(trainer, 'Pixel_G')
    RunningAverage(alpha=0.98, output_transform=lambda x: x['Tot_G']).attach(trainer, 'Tot_G')
    RunningAverage(alpha=0.98, output_transform=lambda x: x['D']).attach(trainer, 'D')
    RunningAverage(alpha=0.98, output_transform=lambda x: x['D_G']).attach(trainer, 'D_G')

    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=monitoring_metrics)

    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler,
                              to_save={
                                  'sensor_gen': camera_gen,
                                  'sensor_disc': camera_disc,
                                  'optim_gen': optimizer_camera_gen,
                                  'optim_disc': optimizer_camera_disc
                              })

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        pbar.log_message('Epoch {} done. Time per batch: {:.3f}[s]'.format(engine.state.epoch, timer.value()))
        timer.reset()

    if config.MODEL == 'baseline' or config.MODEL == 'pix2pix':
        @trainer.on(Events.ITERATION_COMPLETED)
        def generating_samples(engine):
            if not os.path.exists(config.OUTPUT_DIR + '/' + config.MODEL + config.TRAIN.EXAMPLE_SAVE_PATH):
                os.makedirs(config.OUTPUT_DIR + '/' + config.MODEL + config.TRAIN.EXAMPLE_SAVE_PATH)

            if (engine.state.iteration - 1) % 200 == 0:
                with torch.no_grad():
                    generated1 = camera_gen(y1.detach())
                    save_image(generated1,
                               filename=config.OUTPUT_DIR + '/' + config.MODEL + config.TRAIN.EXAMPLE_SAVE_PATH + '/' +
                                        "{}_{}_generated_rgb_1.png".format(engine.state.epoch, engine.state.iteration),
                               normalize=True)
                    # save_image(y1,
                    #            filename= config.TRAIN.EXAMPLE_SAVE_PATH + str(epoch) + "_segmented_1.png",
                    #            normalize=True)
                    np.savez_compressed(config.OUTPUT_DIR + '/' + config.MODEL + config.TRAIN.EXAMPLE_SAVE_PATH + '/' +
                                        "{}_{}_segmented_1".format(engine.state.epoch, engine.state.iteration),
                                        data=y1[-1].cpu().numpy())
                    save_image(x1,
                               filename=config.OUTPUT_DIR + '/' + config.MODEL + config.TRAIN.EXAMPLE_SAVE_PATH + '/' +
                                        "{}_{}_rgb_1.png".format(engine.state.epoch, engine.state.iteration),
                               normalize=True)
                    generated2 = camera_gen(y2.detach())

                    save_image(generated2,
                               filename=config.OUTPUT_DIR + '/' + config.MODEL + config.TRAIN.EXAMPLE_SAVE_PATH + '/' +
                                        "{}_{}_generated_rgb_2.png".format(engine.state.epoch, engine.state.iteration),
                               normalize=True)
                    np.savez_compressed(
                        config.OUTPUT_DIR + '/' + config.MODEL + config.TRAIN.EXAMPLE_SAVE_PATH + '/' +
                        "{}_{}_segmented_2".format(engine.state.epoch, engine.state.iteration),
                        data=y2[-1].cpu().numpy())
                    save_image(x2,
                               filename=config.OUTPUT_DIR + '/' + config.MODEL + config.TRAIN.EXAMPLE_SAVE_PATH + '/' +
                                        "{}_{}_rgb_2.png".format(engine.state.epoch, engine.state.iteration),
                               normalize=True)

    else:
        @trainer.on(Events.ITERATION_COMPLETED)
        def generating_samples(engine):
            if not os.path.exists(config.OUTPUT_DIR + '/' + config.MODEL + config.TRAIN.EXAMPLE_SAVE_PATH):
                os.makedirs(config.OUTPUT_DIR + '/' + config.MODEL + config.TRAIN.EXAMPLE_SAVE_PATH)

            if (engine.state.iteration - 1) % 200 == 0:
                with torch.no_grad():
                    fakeCamera1 = camera_gen(y1.detach())
                    np.savez_compressed(
                        config.OUTPUT_DIR + '/' + config.MODEL + config.TRAIN.EXAMPLE_SAVE_PATH + '/' + str(
                            engine.state.epoch) + "_generated_camera_1",
                        data=fakeCamera1[-1].cpu().numpy())
                    np.savez_compressed(
                        config.OUTPUT_DIR + '/' + config.MODEL + config.TRAIN.EXAMPLE_SAVE_PATH + '/' + str(
                            engine.state.epoch) + "_lidar_1",
                        data=y1[-1].cpu().numpy())
                    np.savez_compressed(
                        config.OUTPUT_DIR + '/' + config.MODEL + config.TRAIN.EXAMPLE_SAVE_PATH + '/' + str(
                            engine.state.epoch) + "_camera_1",
                        data=x1[-1].cpu().numpy())
                    fakeCamera2 = camera_gen(y2.detach())

                    np.savez_compressed(
                        config.OUTPUT_DIR + '/' + config.MODEL + config.TRAIN.EXAMPLE_SAVE_PATH + '/' + str(
                            engine.state.epoch) + "_generated_camera_2",
                        data=fakeCamera2[-1].cpu().numpy())
                    np.savez_compressed(
                        config.OUTPUT_DIR + '/' + config.MODEL + config.TRAIN.EXAMPLE_SAVE_PATH + '/' + str(
                            engine.state.epoch) + "_lidar_2",
                        data=y2[-1].cpu().numpy())
                    np.savez_compressed(
                        config.OUTPUT_DIR + '/' + config.MODEL + config.TRAIN.EXAMPLE_SAVE_PATH + '/' + str(
                            engine.state.epoch) + "_camera_2",
                        data=x2[-1].cpu().numpy())

    @trainer.on(Events.ITERATION_COMPLETED)
    def print_logs(engine):
        if (engine.state.iteration - 1) % config.TRAIN.OUTPUT_FREQ == 0:
            if not os.path.exists(config.OUTPUT_DIR + '/' + config.MODEL):
                os.makedirs(config.OUTPUT_DIR + '/' + config.MODEL)
            fname = os.path.join(config.OUTPUT_DIR + '/' + config.MODEL, config.MODEL)
            columns = engine.state.metrics.keys()
            values = [str(round(value, 5)) for value in engine.state.metrics.values()]

            with open(fname, 'a') as f:
                if f.tell() == 0:
                    print('\t'.join(columns), file=f)
                print('\t'.join(values), file=f)

            i = (engine.state.iteration % len(loader))
            message = '[{epoch}/{max_epoch}][{i}/{max_i}]'.format(epoch=engine.state.epoch,
                                                                  max_epoch=config.TRAIN.MAX_EPOCH,
                                                                  i=i,
                                                                  max_i=len(loader))
            for name, value in zip(columns, values):
                message += ' | {name}: {value}'.format(name=name, value=value)

            pbar.log_message(message)
