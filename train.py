import os
import numpy as np
import tensorflow as tf

from lib import prepare_data

from lib.model import base_model
from lib.ploting import imshow2d, imshow2d_overlay


train_dataset, validation_dataset = prepare_data.load_data()

model = base_model()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
model.summary()

checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
def get_initial_epoch(checkpoint_path):
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print(" >>>>>>>>>>>>>>>>> ",latest)
    epoch_num = latest[15:19]
    return int(epoch_num)

initial_epoch = 0
try:
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    initial_epoch = get_initial_epoch(checkpoint_path)
    print("Resuming ... ")
except:
    print("No checkpoints found... ")

model.fit(x=train_dataset, validation_data=(validation_dataset) ,batch_size=2, epochs=10, callbacks=[cp_callback],  initial_epoch=initial_epoch)

