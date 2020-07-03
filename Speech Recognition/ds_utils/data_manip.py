import numpy as np 
import os
import tensorflow as tf 
try:
    from augmentation import *
except:
    os.system("pwd")
    from ds_utils.augmentation import *
import aesthetix

def find_files(root_search_path = "", files_extension=""):
    files_list = []
    for root, _, files in os.walk(root_search_path):
        files_list.extend([os.path.join(root, file) for file in files if file.endswith(files_extension)])
    return files_list

def clean_label(_str):
        _str = _str.strip()
        _str = _str.lower()
        _str = _str.replace(".", "")
        _str = _str.replace(",", "")
        _str = _str.replace("?", "")
        _str = _str.replace("!", "")
        _str = _str.replace(":", "")
        _str = _str.replace("-", " ")
        _str = _str.replace("_", " ")
        _str = _str.replace("  ", " ")
        return _str

def get_data(path = 'LibriSpeech/', verbose = False):
    text_files = find_files(path, ".txt")
    data = []
    L = len(text_files)
    print(L, "Files have been found.")
    for i, text_file in enumerate(text_files):
        if verbose:
            aesthetix.progress_bar("Reading files", i, L)
        directory = os.path.dirname(text_file)
        with open(text_file, "r") as f:
            lines = f.read().split("\n")
            for line in lines:
                head = line.split(' ')[0]
                if len(head) < 5:
                    # Not a line with a file description
                    break
                audio_file = directory + "/" + head + ".flac"
                if os.path.exists(audio_file):
                    data.append([audio_file, clean_label(line.replace(head, "")), None])
    
    data = np.array(data)
    # print(data.shape)
    data = data[:, :-1] # The last index is NoneType
    print(f"Loaded dataset with shape {data.shape}")
    return data

import string
char_list = list(string.ascii_lowercase)
char_list.extend([' ', '_', '.'])
print(char_list)

char_to_idx = {ch:i for i, ch in enumerate(char_list)}
idx_to_char = {i:ch for i, ch in enumerate(char_list)}
print(idx_to_char)

def label_to_sequence(label):
  return [char_to_idx[ch] for ch in label]

def sequence_to_label(sq):
  prev = "0"
  label = ""
  for idx in sq:
    ch = idx_to_char[idx]
    if ch is not prev:
      if ch == '_':
        prev = ""
        continue
      label+=ch
      if ch == '.' : break
      prev = ch
  return label

print(label_to_sequence("hello sir"))
print(sequence_to_label(label_to_sequence("ccchhaaaaaattttt_tttteeeeeer jee_e")))


labels = datset[:,1]

from tensorflow.keras.preprocessing.sequence import pad_sequences
padding = 'post'
truncating = 'post'

max_len = max([len(lab) for lab in labels])



"""
Data Generator
"""

class SpeechDataGenerator(tf.keras.utils.Sequence):

  def __init__(self, ids, id_to_label, batch_size = 50, dims = (257, 938), augment = False):
    super(SpeechDataGenerator, self).__init__()
    self.ids = ids
    self.id_to_label = id_to_label
    self.batch_size = batch_size
    self.dims = dims
    self.indices = np.arange(len(self.ids))
    self.augment = augment
    self.epoch_num = 0 # for sortagrad
  
  def __len__(self):
    return int(np.floor(len(self.ids) / self.batch_size))
  
  def on_epoch_end(self):
    np.random.shuffle(self.indices)
    self.epoch_num += 1

  def __data_generation(self, idxes):
    y = []
    batch_files = [self.ids[idx] for idx in idxes]
    X = to_spectrograms(batch_files)
    for idx in idxes:
      lbl = clean_label(self.id_to_label[self.ids[idx]]) + "."
      seq = label_to_sequence(lbl)
      y.append(seq)

    y = pad_sequences(y, maxlen=257,padding = padding, truncating=truncating, value = char_to_idx['_'])
    if self.augment : X, y = specAugment(X, y,add_random=False)
    X = np.array(X)
    y = np.array(y)
    return X, y

  def __getitem__(self, index):
    den = 1 
    if self.augment : den = 2
    indices = self.indices[index*int(self.batch_size/den) : (index+1)*int(self.batch_size/den)]
    X, y = self.__data_generation(indices)
    return X, y