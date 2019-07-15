import torch
import matplotlib.pyplot as plt
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

def display_two_images(image1, image2):
    plt.subplot(2, 1, 1)
    plt.imshow(image1)
    plt.axis("off")
    plt.title("Camera")
    plt.subplot(2, 1, 2)
    plt.imshow(image2)
    plt.axis("off")
    plt.title("SEGMENTED IMAGE")
    plt.show()