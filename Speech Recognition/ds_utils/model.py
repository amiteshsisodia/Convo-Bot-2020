import tensorflow as tf 
from tensorflow.keras.layers import Dense, LayerNormalization, Bidirectional, Reshape, LSTM, Conv2D, Input, TimeDistributed, LayerNormalization
from tensorflow.keras.models import Model, Sequential


# The Loss Function
@tf.function
@tf.autograph.experimental.do_not_convert
def ctc_batch_loss(y_true, y_pred):
  # From SO : Todo : var init, predlength objective
  # convert y_pred from one-hot to label indices
  eos_index = char_to_idx['.']
  y_pred_ind = tf.argmax(y_pred, axis=-1)

  #to make sure y_pred has one end_of_sentence (to avoid errors)
  y_pred_end = tf.concat([y_pred_ind[:,:-1], eos_index * tf.ones_like(y_pred_ind[:,-1:])], axis = 1)

  #to make sure the first occurrence of the char is more important than subsequent ones
  occurrence_weights = tf.keras.backend.arange(start = 0, stop=257,  dtype=tf.keras.backend.floatx())

  is_eos_true = tf.keras.backend.cast_to_floatx(tf.equal(y_true, eos_index))
  is_eos_pred = tf.keras.backend.cast_to_floatx(tf.equal(y_pred_end, eos_index))
  #lengths
  true_lengths = 1 + tf.argmax(occurrence_weights * is_eos_true, axis=1)
  pred_lengths = 1 + tf.argmax(occurrence_weights * is_eos_pred, axis=1)

  #reshape
  true_lengths = tf.reshape(true_lengths, (-1,1))
  pred_lengths = tf.reshape(pred_lengths, (-1,1))

  loss_ = tf.keras.backend.ctc_batch_cost(y_true, y_pred, pred_lengths, true_lengths)
  return tf.reduce_min(loss_)


# Model Definition
class DSModel():

  def __init__(self, num_conv=1, conv_filters = 4, conv_kernel = (5, 5), num_rnn=1, rnn_units = 500,spect_shape = (257, 938)):
    self.num_conv = num_conv
    self.conv_filters = conv_filters
    self.conv_kernel = conv_kernel
    self.num_rnn = num_rnn
    self.rnn_units = rnn_units
    self.spect_shape = spect_shape

    # Layers
    self.input = Input(shape = (self.spect_shape), name = "Input")
    self.spect2im = Reshape((*self.spect_shape, 1))
    self.convs = [Conv2D(conv_filters, conv_kernel, padding = 'same', name = f"Conv{i}") for i in range(num_conv)]
    self.conv2rnn = Reshape((spect_shape[0], spect_shape[1]*conv_filters))
    self.rnns = [Bidirectional(LSTM(self.rnn_units, return_sequences=True), name = f"RNN{i}") for i in range(num_rnn)]
    self.layernorms = [LayerNormalization(name = f"LayerNorm{i}") for i in range(num_rnn)]
    self.output = TimeDistributed(Dense(29, activation = 'softmax'), name="Logits_Output")

    # Model
    self.model = None

  def build(self, show_summary = True):

    X = self.input
    X_ = self.spect2im(X)
    for i in range(self.num_conv):
      X_ = self.convs[i](X_)
    X_ = self.conv2rnn(X_)
    for i in range(self.num_rnn):
      X_ = self.rnns[i](X_)
      X_ = self.layernorms[i](X_)
    
    logits = self.output(X_)

    self.model = tf.keras.models.Model(inputs = X, outputs = logits)
    self.model.compile(optimizer = 'adam', loss = ctc_batch_loss, metrics = ['accuracy'])
    if show_summary:
      self.model.summary()

    return self.model

  def restore(self,checkpoint_path):
    if self.model is None:
      print("Building the model first.")
      self.build()
    self.model.load_weights(checkpoint_path)