import os
import errno
import collections
import json

import numpy as np
import tensorflow as tf
import pyhocon

def make_summary(value_dict):
  return tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k,v in value_dict.items()])

def flatten(l):
  return [item for sublist in l for item in sublist]

def get_config(filename):
  return pyhocon.ConfigFactory.parse_file(filename)

def print_config(config):
  print pyhocon.HOCONConverter.convert(config, "hocon")

def set_gpus(*gpus):
  os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
  print "Setting CUDA_VISIBLE_DEVICES to: {}".format(os.environ["CUDA_VISIBLE_DEVICES"])

def mkdirs(path):
  try:
    os.makedirs(path)
  except OSError as exception:
    if exception.errno != errno.EEXIST:
      raise
  return path

def load_char_dict(char_vocab_path):
  vocab = [u"<unk>"]
  with open(char_vocab_path) as f:
    vocab.extend(unicode(c, "utf-8").strip() for c in f.readlines())
  char_dict = collections.defaultdict(int)
  char_dict.update({c:i for i,c in enumerate(vocab)})
  return char_dict

def load_embedding_dict(embedding_path, embedding_size, embedding_format):
  print("Loading word embeddings from {}...".format(embedding_path))
  default_embedding = np.zeros(embedding_size)
  embedding_dict = collections.defaultdict(lambda:default_embedding)
  skip_first = embedding_format == "vec"
  with open(embedding_path) as f:
    for i, line in enumerate(f.readlines()):
      if skip_first and i == 0:
        continue
      splits = line.split()
      assert len(splits) == embedding_size + 1
      word = splits[0]
      embedding = np.array([float(s) for s in splits[1:]])
      embedding_dict[word] = embedding
  print("Done loading word embeddings.")
  return embedding_dict

def maybe_divide(x, y):
  return 0 if y == 0 else x / float(y)

def projection(inputs, output_size, initializer=None):
  return ffnn(inputs, 0, -1, output_size, dropout=None, output_weights_initializer=initializer)

def projection_comp(a, b, c, d, output_size, name, initializer=None):
  return ffnn_name(a, 0, -1, output_size, '{}_a'.format(name), dropout=None, output_weights_initializer=initializer)+\
            ffnn_name(b, 0, -1, output_size, '{}_b'.format(name), dropout=None, output_weights_initializer=initializer)+\
            ffnn_name(c, 0, -1, output_size, '{}_c'.format(name), dropout=None, output_weights_initializer=initializer)+\
            ffnn_name(d, 0, -1, output_size, '{}_d'.format(name), dropout=None, output_weights_initializer=initializer)

def projection_name(inputs, output_size, name, initializer=None):
  return ffnn_name(inputs, 0, -1, output_size, name, dropout=None, output_weights_initializer=initializer)

def ffnn_name(inputs, num_hidden_layers, hidden_size, output_size, name, dropout, output_weights_initializer=None):
  if len(inputs.get_shape()) > 2:
    current_inputs = tf.reshape(inputs, [-1, shape(inputs, -1)])
  else:
    current_inputs = inputs

  for i in xrange(num_hidden_layers):
    with tf.variable_scope("attention", reuse=tf.AUTO_REUSE):
      hidden_weights = tf.get_variable("hidden_weights_{}_{}".format(i, name), [shape(current_inputs, 1), hidden_size])
      hidden_bias = tf.get_variable("hidden_bias_{}_{}".format(i, name), [hidden_size])
      current_outputs = tf.nn.relu(tf.matmul(current_inputs, hidden_weights) + hidden_bias)

    if dropout is not None:
      current_outputs = tf.nn.dropout(current_outputs, dropout)
    current_inputs = current_outputs

  with tf.variable_scope("attention", reuse=tf.AUTO_REUSE):
    output_weights = tf.get_variable("output_weights_{}".format(name), [shape(current_inputs, 1), output_size], initializer=output_weights_initializer)
    output_bias = tf.get_variable("output_bias_{}".format(name), [output_size])
    outputs = tf.matmul(current_inputs, output_weights) + output_bias

  if len(inputs.get_shape()) == 3:
    outputs = tf.reshape(outputs, [shape(inputs, 0), shape(inputs, 1), output_size])
  elif len(inputs.get_shape()) == 4:
    outputs = tf.reshape(outputs, [shape(inputs, 0), shape(inputs, 1), shape(inputs, 2), output_size])
  elif len(inputs.get_shape()) > 4:
    raise ValueError("FFNN with rank {} not supported".format(len(inputs.get_shape())))
  return outputs

def shape(x, dim):
  return x.get_shape()[dim].value or tf.shape(x)[dim]

def ffnn(inputs, num_hidden_layers, hidden_size, output_size, dropout, output_weights_initializer=None):
  if len(inputs.get_shape()) > 2:
    current_inputs = tf.reshape(inputs, [-1, shape(inputs, -1)])
  else:
    current_inputs = inputs

  for i in xrange(num_hidden_layers):
    hidden_weights = tf.get_variable("hidden_weights_{}".format(i), [shape(current_inputs, 1), hidden_size])
    hidden_bias = tf.get_variable("hidden_bias_{}".format(i), [hidden_size])
    current_outputs = tf.nn.relu(tf.matmul(current_inputs, hidden_weights) + hidden_bias)

    if dropout is not None:
      current_outputs = tf.nn.dropout(current_outputs, dropout)
    current_inputs = current_outputs

  output_weights = tf.get_variable("output_weights", [shape(current_inputs, 1), output_size], initializer=output_weights_initializer)
  output_bias = tf.get_variable("output_bias", [output_size])
  outputs = tf.matmul(current_inputs, output_weights) + output_bias

  if len(inputs.get_shape()) == 3:
    outputs = tf.reshape(outputs, [shape(inputs, 0), shape(inputs, 1), output_size])
  elif len(inputs.get_shape()) == 4:
    outputs = tf.reshape(outputs, [shape(inputs, 0), shape(inputs, 1), shape(inputs, 1), output_size])
  elif len(inputs.get_shape()) > 4:
    raise ValueError("FFNN with rank {} not supported".format(len(inputs.get_shape())))
  return outputs


def cnn_name(inputs, filter_sizes, num_filters, name):
  num_words = shape(inputs, 0)
  num_chars = shape(inputs, 1)
  input_size = shape(inputs, 2)
  outputs = []
  for i, filter_size in enumerate(filter_sizes):
    with tf.variable_scope("conv_{}_{}".format(i, name)):
      w = tf.get_variable("w", [filter_size, input_size, num_filters])
      b = tf.get_variable("b", [num_filters])
    conv = tf.nn.conv1d(inputs, w, stride=1, padding="SAME") # [num_words, num_chars - filter_size, num_filters]
    h = tf.nn.relu(tf.nn.bias_add(conv, b)) # [num_words, num_chars - filter_size, num_filters]
    # pooled = tf.reduce_max(h, 1) # [num_words, num_filters]
    outputs.append(h)
  return tf.concat(outputs, 1) # [num_words, num_filters * len(filter_sizes)]

def one_hot(x, y):
  x[int(y)] = 1
  return list(x)

def cnn(inputs, filter_sizes, num_filters):
  num_words = shape(inputs, 0)
  num_chars = shape(inputs, 1)
  input_size = shape(inputs, 2)
  outputs = []
  for i, filter_size in enumerate(filter_sizes):
    with tf.variable_scope("conv_{}".format(i)):
      w = tf.get_variable("w", [filter_size, input_size, num_filters])
      b = tf.get_variable("b", [num_filters])
    conv = tf.nn.conv1d(inputs, w, stride=1, padding="VALID") # [num_words, num_chars - filter_size, num_filters]
    h = tf.nn.relu(tf.nn.bias_add(conv, b)) # [num_words, num_chars - filter_size, num_filters]
    pooled = tf.reduce_max(h, 1) # [num_words, num_filters]
    outputs.append(pooled)
  return tf.concat(outputs, 1) # [num_words, num_filters * len(filter_sizes)]

def normalize(v):
  norm = np.linalg.norm(v)
  if norm > 0:
    return v / norm
  else:
    return v

class RetrievalEvaluator(object):
  def __init__(self):
    self._num_correct = 0
    self._num_gold = 0
    self._num_predicted = 0

  def update(self, gold_set, predicted_set):
    self._num_correct += len(gold_set & predicted_set)
    self._num_gold += len(gold_set)
    self._num_predicted += len(predicted_set)

  def recall(self):
    return maybe_divide(self._num_correct, self._num_gold)

  def precision(self):
    return maybe_divide(self._num_correct, self._num_predicted)

  def metrics(self):
    recall = self.recall()
    precision = self.precision()
    f1 = maybe_divide(2 * recall * precision, precision + recall)
    return recall, precision, f1

class CustomLSTMCell(tf.contrib.rnn.RNNCell):
  def __init__(self, num_units, batch_size, dropout):
    self._num_units = num_units
    self._dropout = dropout
    self._dropout_mask = tf.nn.dropout(tf.ones([batch_size, self.output_size]), dropout)
    self._initializer = self._block_orthonormal_initializer([self.output_size] * 3)
    initial_cell_state = tf.get_variable("lstm_initial_cell_state", [1, self.output_size])
    initial_hidden_state = tf.get_variable("lstm_initial_hidden_state", [1, self.output_size])
    self._initial_state = tf.contrib.rnn.LSTMStateTuple(initial_cell_state, initial_hidden_state)

  @property
  def state_size(self):
    return tf.contrib.rnn.LSTMStateTuple(self.output_size, self.output_size)

  @property
  def output_size(self):
    return self._num_units

  @property
  def initial_state(self):
    return self._initial_state

  def preprocess_input(self, inputs):
    return projection(inputs, 3 * self.output_size)

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with tf.variable_scope(scope or type(self).__name__):  # "CustomLSTMCell"
      c, h = state
      h *= self._dropout_mask
      projected_h = projection(h, 3 * self.output_size, initializer=self._initializer)
      concat = inputs + projected_h
      i, j, o = tf.split(concat, num_or_size_splits=3, axis=1)
      i = tf.sigmoid(i)
      new_c = (1 - i) * c  + i * tf.tanh(j)
      new_h = tf.tanh(new_c) * tf.sigmoid(o)
      new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
      return new_h, new_state

  def _orthonormal_initializer(self, scale=1.0):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
      M1 = np.random.randn(shape[0], shape[0]).astype(np.float32)
      M2 = np.random.randn(shape[1], shape[1]).astype(np.float32)
      Q1, R1 = np.linalg.qr(M1)
      Q2, R2 = np.linalg.qr(M2)
      Q1 = Q1 * np.sign(np.diag(R1))
      Q2 = Q2 * np.sign(np.diag(R2))
      n_min = min(shape[0], shape[1])
      params = np.dot(Q1[:, :n_min], Q2[:n_min, :]) * scale
      return params
    return _initializer

  def _block_orthonormal_initializer(self, output_sizes):
    def _initializer(shape, dtype=np.float32, partition_info=None):
      assert len(shape) == 2
      assert sum(output_sizes) == shape[1]
      initializer = self._orthonormal_initializer()
      params = np.concatenate([initializer([shape[0], o], dtype, partition_info) for o in output_sizes], 1)
      return params
    return _initializer


class RecurrentMemNNCell(tf.contrib.rnn.RNNCell):
  def __init__(self, num_units, batch_size, dropout, tag_labels, tag_seq, training=1):
    self._tag_labels = tag_labels
    self._tag_seq = tag_seq
    self.training = tf.Variable(training)
    self._num_units = 100
    self._dropout = dropout
    self._max_entity = 100
    self.hidden_size = 200

    self._dropout_mask = tf.nn.dropout(tf.ones([batch_size, self.output_size]), dropout)
    self._initializer = self._block_orthonormal_initializer([100])

    initial_indexes = tf.constant([[0.0, 2.0]], name="initial_indexes")
    # initial_memory = tf.zeros_like(tf.variable([1, 20000], trainable=False))

    basic_tag = tf.get_variable("basic_tag_emb", [2, self.hidden_size])
    new_entity_tag = tf.zeros([98, self.hidden_size], name="new_entity_emb")
    entity_emb = tf.concat([basic_tag, new_entity_tag], 0)

    initial_memory = tf.reshape(entity_emb, [1, 20000])

    initial_cell_state = tf.concat([initial_indexes, initial_memory], 1)
    initial_cell_state = tf.reshape(initial_cell_state, [1, 20002])

    # initial_cell_state = tf.constant([[0.0, 2.0]], name="lstm_initial_cell_state")
    initial_hidden_state = tf.get_variable("lstm_initial_hidden_state", [1, 100])
    self._initial_state = tf.contrib.rnn.LSTMStateTuple(initial_cell_state, initial_hidden_state)

    self.word_index_update = tf.constant([[1.0, 0.0]])
    self.entity_index_update = tf.constant([[0.0, 1.0]])

    self.init_word_emb = tf.get_variable("init_word_emb", [1, self.hidden_size])

    self.word_index = 0
    self.entity_index = 2
    self.tf_entity_index = tf.Variable(np.array([2]))

  @property
  def state_size(self):
    return tf.contrib.rnn.LSTMStateTuple(20002, self.output_size)

  @property
  def output_size(self):
    return self._num_units

  @property
  def initial_state(self):
    self.word_index = 0
    return self._initial_state
  
  def evaluating(self):
    self._training = False

  def preprocess_input(self, inputs):
    self.num_words = tf.shape(inputs)[0] + 1
    input_emb = projection_name(inputs, self.hidden_size, 'input_emb')
    self.sentence_emb = tf.nn.tanh(tf.concat([self.init_word_emb, tf.squeeze(input_emb)], 0)) # [num_words, emb]
    self.sentence_head = tf.nn.tanh(tf.transpose(projection_name(self.sentence_emb, self.hidden_size, 'word_memnn')))

    # self.entity_emb = tf.concat([self.basic_tag, self.new_entity_tag], 0)
    return input_emb

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with tf.variable_scope(scope or type(self).__name__):  # "RecurrentMemNNCell"
      c, h = state
      # x = tf.matmul(c, tf.zeros([5, 5]))
      raw_cell = tf.reshape(c, [1, 20002])
      indexes, memories = tf.split(raw_cell, [2, 20000], 1)
      cell = tf.squeeze(indexes)

      entity_emb = tf.reshape(memories, [100, self.hidden_size])

      word_index = tf.cast(tf.gather(cell, [0]), tf.int64)
      entity_index = tf.cast(tf.gather(cell, [1]), tf.int64)

      word_mask = tf.reshape(tf.sequence_mask(word_index + 1, self.num_words, dtype=tf.float32), [1, -1])
      entity_mask = tf.reshape(tf.sequence_mask(entity_index + 1, self._max_entity, dtype=tf.float32), [1, -1])

      hist_attn = tf.nn.softmax(tf.matmul(inputs, self.sentence_head) + tf.log(word_mask), dim=1)
      hist_emb = tf.transpose(tf.reduce_sum(tf.transpose(hist_attn) * self.sentence_emb))

      hist_emb = tf.reshape(hist_emb, [1, -1])

      tag_query = tf.concat([hist_emb, inputs], 1, name='concat_hist')
      tag_query = tf.nn.tanh(projection_name(tag_query, self.hidden_size, 'tag_query'))
      tag_input = tf.tile(tag_query, [100, 1])

      tag_gate = tf.concat([tag_input, entity_emb], 1, name='concat_gate')

      logits = tf.nn.softmax(tf.matmul(tag_query, tf.transpose(entity_emb)) + tf.log(entity_mask))
      self.prediction = tf.argmax(logits)

      new_entity_query = tf.cond(tf.equal(self.training, 1),
              lambda: tf.cast(tf.gather(self._tag_seq, word_index), tf.int64),
              lambda: self.prediction)
      
      new_entity_cond = tf.reshape(tf.equal(new_entity_query, entity_index), [])

      e_update = tf.cond(new_entity_cond, lambda: self.entity_index_update, lambda: tf.zeros([1, 2]))

      write_head = tf.cond(new_entity_cond,
              lambda: tf.one_hot([self.entity_index], self._max_entity),
              lambda: tf.nn.sigmoid(projection_name(tag_gate, 1, 'write_head') * tf.transpose(entity_mask)))
      
      write_head = tf.reshape(write_head, [100, 1])
      new_entity_emb = tf.reshape(write_head * tag_input + (1 - write_head) * entity_emb, [1, 20000])

      # print 14
      new_indexes = indexes + self.word_index_update + e_update

      new_c = tf.concat([new_indexes, new_entity_emb], 1)

      new_state = tf.contrib.rnn.LSTMStateTuple(new_c, logits)

      return logits, new_state

  def _orthonormal_initializer(self, scale=1.0):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
      M1 = np.random.randn(shape[0], shape[0]).astype(np.float32)
      M2 = np.random.randn(shape[1], shape[1]).astype(np.float32)
      Q1, R1 = np.linalg.qr(M1)
      Q2, R2 = np.linalg.qr(M2)
      Q1 = Q1 * np.sign(np.diag(R1))
      Q2 = Q2 * np.sign(np.diag(R2))
      n_min = min(shape[0], shape[1])
      params = np.dot(Q1[:, :n_min], Q2[:n_min, :]) * scale
      return params
    return _initializer

  def _block_orthonormal_initializer(self, output_sizes):
    def _initializer(shape, dtype=np.float32, partition_info=None):
      assert len(shape) == 2
      assert sum(output_sizes) == shape[1]
      initializer = self._orthonormal_initializer()
      params = np.concatenate([initializer([shape[0], o], dtype, partition_info) for o in output_sizes], 1)
      return params
    return _initializer
