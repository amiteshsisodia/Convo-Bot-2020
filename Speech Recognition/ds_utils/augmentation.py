import librosa 
import librosa.display 
import tensorflow as tf
from tensorflow_addons.image import sparse_image_warp
import numpy as np 
import matplotlib.pyplot as plt
from spela.spectrogram import Spectrogram

# Spectrogram Model
SR = 16000
max_time = 15

GetSpec = tf.keras.models.Sequential( [Spectrogram(n_dft=512, n_hop=256, input_shape=(1, SR*max_time), 
          return_decibel_spectrogram=False, power_spectrogram=2.0, 
          trainable_kernel=False, name='static_stft')] )


GetSpec.compile()
# Spectrogram from file
def to_spectrograms(audio_files, max_time = max_time):
    # Loads the audio files and returns the spectrograms
    X = []
    for audio_file in audio_files:
      x, _ = librosa.load(audio_file, sr = SR)
      if len(x)<SR*max_time :x = list(x) + [0]*((SR*max_time)-len(x))
      x = x[:SR*max_time]
      assert len(x)==SR*max_time, f"{len(x)}, expected : {SR*max_time}"
      X.append(x)
    X = np.array(X)
    X = np.reshape(X, (len(audio_files), 1, SR*max_time))
    spectrograms = GetSpec.predict(X)
    return spectrograms[:, :, :, 0]

def show_spectrogram(spec):
  spec = librosa.power_to_db(spec)
  librosa.display.specshow(spec, y_axis='linear', sr=SR)


"""
SpecAugment Methods
"""

def time_warp(spectrogram, W=80):
    # Returns the time-warped tensor given the spectrogram
    
    tau, f = spectrogram.shape

    # Source control point locations
    point = tf.random.uniform(shape = [], minval = W, maxval = tau - W, dtype = tf.int32)
    freq_at_point = tf.range(f//2) # The column of the spectorgram at point
    time_at_point = tf.ones_like(freq_at_point, dtype=tf.int32)*point # control points on the time axis 
    scpt = tf.cast(tf.stack((freq_at_point, time_at_point), axis = -1), dtype = tf.float32)
    scpt = tf.expand_dims(scpt, axis = 0)

    # Destination control point locations
    dt = tf.random.uniform(shape = [], minval = -W, maxval = W, dtype = tf.int32)
    dest_freq_at_point = freq_at_point
    dest_time_at_point = time_at_point + dt
    dcpt =  tf.cast(tf.stack((dest_freq_at_point, dest_time_at_point), axis = -1), dtype = tf.float32)
    dcpt = tf.expand_dims(dcpt, axis = 0)
    
    spect = tf.cast(tf.reshape(spectrogram, [1, *spectrogram.shape, 1]), dtype = tf.float32)
    warped_dat, _ = sparse_image_warp(spect, 
                                   source_control_point_locations = scpt, 
                                   dest_control_point_locations = dcpt,
                                   num_boundary_points=2) # Need to see if there is any way to have 1.5-equivalent
    warped_dat = tf.reshape(warped_dat, spectrogram.shape)
    return warped_dat



def frequency_mask(spectrogram, F=27):
    # Adding a frequency mask
    f = tf.random.uniform([], minval = 0, maxval = F, dtype = tf.int32)
    v, T = spectrogram.shape
    f0 = tf.random.uniform([], minval = 0, maxval = v//2-f, dtype = tf.int32)
    res1 = spectrogram[:f0,:]
    res2 = spectrogram[f0+f:, :]
    mask = tf.zeros_like(spectrogram[f0:f0+f,:])
    masked_spec = tf.concat([res1, mask, res2], axis = 0)
    assert masked_spec.shape == spectrogram.shape
    return tf.cast(masked_spec, dtype = tf.float64)


def time_mask(spectrogram, T=100):
    # Adding a time mask

    t = tf.random.uniform([], minval = 0, maxval = T, dtype = tf.int32)
    _, tau = spectrogram.shape
    t0 = tf.random.uniform([], minval = 0, maxval = tau-t, dtype = tf.int32)
    res1 = spectrogram[:,:t0]
    res2 = spectrogram[:, t0+t:]
    mask = tf.zeros_like(spectrogram[:, t0:t0+t])
    masked_spec = tf.concat([res1, mask, res2], axis = 1)
    assert masked_spec.shape == spectrogram.shape
    return tf.cast(masked_spec, dtype = tf.float64)

def specAugment(data,labels, W=80, F=27, T=100, mF=1, mT=1, add_random = False):
    """
    Apply specAugmentation to the batch of data

    params:
        data : batch of spectrograms to be augmented
        W : Time warp parameter 
        F : Frequency mask parameter
        T : Time mask parameter
        mF : no. of frequency masks to be applied
        mT : no. of time masks to be applied
        add_random : bool, determines whether random augmentations are added

    returns:
        augmented batch, with 2x (3x with add_random True) the number of samples in the input batch

    """
    aug_data = []
    Y = []
    for spect, y in zip(data, labels):
        aug_data.append(spect)
        Y.append(y)

        if add_random:
            augmentations = [lambda x : time_warp(x, W), 
                            lambda x :frequency_mask(x, F), 
                            lambda x:time_mask(x, T) ]
            
            num_aug = np.random.choice([1, 2, 3])
            aug = spect
            for i in range(num_aug):
                aug = augmentations[i](aug)
                aug_data.append(aug)
                Y.append(y)
        
        struct_aug = time_warp(spect, W)
        for i in range(mF) : struct_aug = frequency_mask(struct_aug, F)
        for i in range(mT) : struct_aug = time_mask(struct_aug, T)

        aug_data.append(struct_aug)
        Y.append(y)

    aug_data = np.array(aug_data)
    Y = np.array(Y)
    return tf.convert_to_tensor(aug_data), Y