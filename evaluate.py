import numpy as np

from model import UNet
import prepare_data
from ploting import imshow2d, imshow2d_overlay
import os
import tensorflow as tf

checkpoint_path = "checkpoints/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model = UNet()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])

try:
    model.load_weights(checkpoint_path)
    print("Resuming ... ")
except:
    print("No checkpoints found... ")
