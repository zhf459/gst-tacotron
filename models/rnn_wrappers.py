import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell, LSTMStateTuple
from .modules import prenet


class DecoderPrenetWrapper(RNNCell):
  '''Runs RNN inputs through a prenet before sending them to the cell.'''
  def __init__(self, cell, is_training):
    super(DecoderPrenetWrapper, self).__init__()
    self._cell = cell
    self._is_training = is_training

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def call(self, inputs, state):
    prenet_out = prenet(inputs, self._is_training, scope='decoder_prenet')
    return self._cell(prenet_out, state)

  def zero_state(self, batch_size, dtype):
    return self._cell.zero_state(batch_size, dtype)



class ConcatOutputAndAttentionWrapper(RNNCell):
  '''Concatenates RNN cell output with the attention context vector.

  This is expected to wrap a cell wrapped with an AttentionWrapper constructed with
  attention_layer_size=None and output_attention=False. Such a cell's state will include an
  "attention" field that is the context vector.
  '''
  def __init__(self, cell):
    super(ConcatOutputAndAttentionWrapper, self).__init__()
    self._cell = cell

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size + self._cell.state_size.attention

  def call(self, inputs, state):
    output, res_state = self._cell(inputs, state)
    return tf.concat([output, res_state.attention], axis=-1), res_state

  def zero_state(self, batch_size, dtype):
    return self._cell.zero_state(batch_size, dtype)


class ZoneoutWrapper(RNNCell):
  """Operator adding zoneout to all states of the given cell."""

  def __init__(self, cell, zoneout_prob, is_training=True):
    super(ZoneoutWrapper, self).__init__()
    self._cell = cell
    self._zoneout_prob = zoneout_prob
    self.is_training = is_training

    if isinstance(self.state_size, LSTMStateTuple):
      if not isinstance(zoneout_prob, tuple):
        raise TypeError("Subdivided states need subdivided zoneouts.")
      if len(self.state_size) != len(self._zoneout_prob):
        raise ValueError("State and zoneout need equally many parts.")

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def call(self, inputs, state):
    output, new_state = self._cell(inputs, state)

    if isinstance(self.state_size, LSTMStateTuple):
      c, h = state
      new_c, new_h = new_state
      zoneout_prob_c, zoneout_prob_h = self._zoneout_prob

      if self.is_training:
        # Rescales the output of dropout (tf.nn.dropout scales it's output
        # by a factor of 1 / keep_prob).
        new_c = (1 - zoneout_prob_c) * tf.nn.dropout(new_c - c, (1 - zoneout_prob_c)) + c
        new_h = (1 - zoneout_prob_h) * tf.nn.dropout(new_h - h, (1 - zoneout_prob_h)) + h
        new_state = LSTMStateTuple(c=new_c, h=new_h)
      else:
        # Uses expectation at test time.
        new_c = zoneout_prob_c * c + (1 - zoneout_prob_c) * new_c
        new_h = zoneout_prob_h * h + (1 - zoneout_prob_h) * new_h
        new_state = LSTMStateTuple(c=new_c, h=new_h)
      return new_h, new_state
    else:
      if self.is_training:
        new_state = state + (1 - self._zoneout_prob) * tf.nn.dropout(
          new_state - state, (1 - self._zoneout_prob))
      else:
        new_state = self._zoneout_prob * state + (1 - self._zoneout_prob) * new_state
      return new_state, new_state
      #return output, new_state

  def zero_state(self, batch_size, dtype):
    return self._cell.zero_state(batch_size, dtype)
