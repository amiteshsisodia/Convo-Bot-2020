import argparse

# Adding arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", default = 'bin/LibriSpeech', help = "Path to the dataset")
parser.add_argument("--checkpoint_path", default = 'bin/Training/cp.ckpt', help = "Path to the dataset")
parser.add_argument("-augment", default = False, help='whether to add augmentations', action = 'store_true')
parser.add_argument("-e", "--epochs", type=int, default = 10, help='Number of epochs')
parser.add_argument("-bs" "--batch_size", type  = int, default = 50, help = "batch size")

args = parser.parse_args()

DATASET_PATH = args.dataset_path
CKPT_PATH = args.checkpoint_path
AUGMENT = args.augment
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

# Dependencies
from ds_utils.augmentation import *
from ds_utils.data_manip import *
from ds_utils.model import *
import matplotlib.pyplot as plt
import os

# Defining checkpoint variables
import os
checkpoint_path = CKPT_PATH
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, period = 5)


# History-plotting function
def plothist(history):
    loss_hist = history.history['loss']
    acc_hist = history.history['accuracy']
    f, (plt1, plt2) = plt.subplots(1, 2)
    plt1.plot(acc_hist, label = 'accuracy')
    plt2.plot(loss_hist, label="loss/max_loss")
    plt1.xlabel("Epochs")
    plt1.ylabel("Loss")
    plt1.xlabel("Epochs")
    plt1.ylabel("Acc")
    plt.legend()
    plt.show()

# Dataset
datset = get_data(path = DATASET_PATH, verbose=True)
files = datset[:, 0]
file_to_label = { datset[i, 0]:datset[i,1] for i in range(datset.shape[0])}

# Model
ds_model = DSModel(num_conv = 2, num_rnn = 3)
model = ds_model.build()

try:
    ds_model.restore(checkpoint_path)
except:
    print("No previous checkpoints")

speechgen = SpeechDataGenerator(files, file_to_label, augment = AUGMENT, batch_size=BATCH_SIZE)

hist = model.fit(speechgen, epochs = EPOCHS)

plothist(hist)