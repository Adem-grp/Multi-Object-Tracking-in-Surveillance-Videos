import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import os

def view_random_image(target_dir):

    img = mpimg.imread(target_dir)
    plt.imshow(img)
    plt.axis("off");

    print(f"Image shape: {img.shape}")

    return img

img = view_random_image(target_dir=r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\avenue_yolo\train\images\01_000001.jpg")