import os
import errno
import collections
import json

import numpy as np
import tensorflow as tf
import pyhocon

def check_tags(tags):
  current_high = -1
  for i, tag in enumerate(tags):
    if tag - current_high > 1:
      print "current_high: %d, tag: %d, location: %d" % (current_high, tag, i)
      return False
    if tag > current_high:
      current_high = tag
  return True

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
    output_weights = tf.get_variable("output_weights_{}".format(name), [shape(current_inputs, 1), output_size], initializer=output_weights_initializer, validate_shape=False)
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
  def __init__(self,
      inputs,
      num_units,
      batch_size,
      dropout,
      mention_scores,
      antecedent_features,
      word_emb,
      mention_starts,
      mention_ends,
      start_scores,
      end_scores,
      k
    ):
    self._num_units = 1 + k
    self._dropout = dropout
    self.hidden_size = 200
    self.inputs_dim = tf.shape(inputs)[2]
    # self.a_size = k * k
    # self.m_size = k * self.hidden_size
    # self._max_cell = 1 + 250 * 250 + 250 * self.hidden_size + 1
    # self.padding_size = self._max_cell - self.a_size - self.m_size - 1
    # self.padding = tf.zeros([1, self.padding_size])
    # self.k = tf.reshape(k, [])

    self._dropout_mask = tf.nn.dropout(tf.ones([batch_size, self.output_size]), dropout)
    self._initializer = self._block_orthonormal_initializer([100])

    self.num_words = tf.shape(word_emb)[0] + 1
    self.num_mentions = tf.shape(inputs)[0]

    self.antecedent_features = antecedent_features

    initial_index = tf.constant([[0.0]], name="initial_indexes")

    self.word_emb = word_emb
    self.mention_starts = mention_starts
    self.mention_ends = mention_ends
    self.start_scores = start_scores
    self.end_scores = end_scores

    # with tf.variable_scope("init_tags"):
    #   basic_tag = tf.get_variable("basic_tag", [1, self.hidden_size])
    #   new_tag = tf.get_variable("new_tag", [1, self.hidden_size])
    #   # new_entity_tag = tf.zeros([99, self.hidden_size], name="new_entity_emb")
    #   new_entity_tag = tf.tile(new_tag, [99, 1])
    
    # entity_emb = tf.concat([basic_tag, new_entity_tag], 0)
    # entity_emb = tf.tile(basic_tag, [100, 1])
    # A = tf.reshape(tf.eye(k), [1, -1])

    # self.input_emb = inputs

    # with tf.variable_scope("prepocess"):
    #   self.word_emb = tf.nn.relu(projection_name(word_emb, self.hidden_size, 'input_emb'))
    
    # sentence_emb = tf.reshape(tf.transpose(self.input_emb, [1, 0, 2])[0], [1, -1])
    # self.span_emb = tf.transpose(self.input_emb, [1, 0, 2])[0]
    init_word_emb = tf.zeros([1, tf.shape(self.word_emb)[1]])
    self.sentence_emb_init = tf.concat([init_word_emb, self.word_emb], 0) # [num_words, emb]
    # self.sentence_head = tf.nn.tanh(tf.transpose(projection_name(self.sentence_emb, self.hidden_size, 'word_memnn')))

    self.num_mentions = tf.shape(mention_scores)[0]
    # with tf.variable_scope("mention_scores"):
    self.mention_scores = tf.reshape(mention_scores, [self.num_mentions, 1])

    # self.non_entity_mask = tf.one_hot([0], 100)
    # self.scores_mask = tf.transpose(1 - self.non_entity_mask)

    # initial_memory = tf.concat([initial_index, A, sentence_emb, self.padding], 1)
    # initial_cell_state = tf.reshape(initial_memory, [1, self._max_cell])

    # initial_memory = tf.reshape(entity_emb, [1, 20000])

    # initial_cell_state = tf.concat([initial_indexes, initial_memory], 1)
    # initial_cell_state = tf.reshape(initial_cell_state, [1, 20002])

    initial_hidden_state = tf.zeros([1, 1 + k])
    self._initial_state = tf.contrib.rnn.LSTMStateTuple(initial_index, initial_hidden_state)

    self.word_index_update = tf.constant([[1.0]])

  @property
  def state_size(self):
    return tf.contrib.rnn.LSTMStateTuple(20002, self.output_size)

  @property
  def output_size(self):
    return self._num_units

  @property
  def initial_state(self):
    return self._initial_state
  
  def evaluating(self):
    self._training = False

  def preprocess_input(self, inputs, mention_scores, antecedent_features):
    '''
    self.num_words = tf.shape(inputs)[0] + 1
    self.antecedent_features = antecedent_features

    with tf.variable_scope("prepocess"):
      # input_emb = tf.nn.tanh(projection_name(inputs, self.hidden_size, 'input_emb'))
      input_emb = tf.nn.relu(projection_name(inputs, self.hidden_size, 'input_emb'))
    
    self.sentence_emb = tf.transpose(input_emb, [1, 0, 2])[0]
    init_word_emb = tf.zeros([1, tf.shape(input_emb)[2]])
    self.sentence_emb_init = tf.concat([init_word_emb, tf.squeeze(input_emb)], 0) # [num_words, emb]
    # self.sentence_head = tf.nn.tanh(tf.transpose(projection_name(self.sentence_emb, self.hidden_size, 'word_memnn')))

    self.num_mentions = tf.shape(mention_scores)[0]
    # with tf.variable_scope("mention_scores"):
    self.mention_scores = tf.reshape(mention_scores, [self.num_mentions, 1])

    self.non_entity_mask = tf.one_hot([0], 100)
    self.scores_mask = tf.transpose(1 - self.non_entity_mask)
    '''
    self.inputs_dim = tf.shape(inputs)[2]
    return tf.nn.relu(projection_name(inputs, self.hidden_size, 'input_emb'))

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with tf.variable_scope(scope or type(self).__name__):  # "RecurrentMemNNCell"
      c, h = state

      # raw_cell = tf.reshape(c, [1, self._max_cell])
      # index, A, memories, padding = tf.split(raw_cell, [1, self.a_size, self.m_size, self.padding_size], 1)
      
      # index = tf.reshape(c, [1])
      # A = tf.reshape(A, [self.k, self.k])
      # memories = tf.reshape(memories, [self.k, self.hidden_size])
      # cell = tf.squeeze(indexes)
      inputs_emb, current_score = tf.split(inputs, [self.hidden_size, 1], 1)
      # inputs = tf.reshape(inputs, [1, self.inputs_dim])
      # x = tf.matmul(self.sentence_emb_init, tf.zeros([4, 10]), name='test')
      context_mention_score = tf.tile(current_score, [self.num_words - 1, 1]) * 0.5
      # entity_mention_score = tf.tile(current_score, [100, 1]) * self.scores_mask

      # sentence_emb = tf.reshape(memories, [self.k, self.hidden_size])
      # with tf.variable_scope("entity_emb"):
      #   entity_emb = tf.nn.relu(projection(sentence_emb, self.hidden_size))

      word_index = tf.reshape(tf.cast(c, tf.int64), [1])

      mention_start = tf.gather(self.mention_starts, word_index)
      mention_end = tf.gather(self.mention_ends, word_index)
      # entity_index = tf.cast(tf.gather(cell, [1]), tf.int64)

      pair_feature = tf.gather(self.antecedent_features, word_index)[0]

      # pair_feature = tf.matmul(pair_feature, tf.zeros([4, 8]))

      inputs_tiled = tf.tile(inputs_emb, [self.num_words - 1, 1])
      # inputs_tiled = tf.reshape(inputs_tiled, [self.num_words - 1, self.inputs_dim])
      # similarity_emb = inputs_tiled * self.span_emb

      # inputs_tiled = tf.matmul(tf.cast(tf.shape(inputs_tiled), tf.float32), tf.zeros([4, 8]))
      # self.word_emb = tf.matmul(self.word_emb, tf.zeros([4, 8]))

      # entity_sim_emb = inputs_tiled * entity_emb

      # pair_emb = tf.concat([pair_feature, inputs_tiled, entity_emb, self.span_emb, similarity_emb], 1)
      # pair_emb = tf.concat([pair_feature, inputs_tiled, entity_emb, self.span_emb, similarity_emb, entity_sim_emb], 1)
      sword_emb = tf.concat([pair_feature, inputs_tiled, self.word_emb], 1)
      # sword_emb = tf.matmul(tf.cast(shape(sword_emb, 1), tf.float32), tf.zeros([4, 8]))
      # sword_emb.set_shape([None, None])

      # sword_emb = tf.matmul(sword_emb, tf.zeros([4, 8]))

      # mention_att = projection_name(pair_emb, 1, 'context_att') + self.mention_scores + context_mention_score
      with tf.variable_scope("start_att"):
        start_att = ffnn(sword_emb, 2, 150, 1, self._dropout) + self.start_scores + context_mention_score


      start_mask = tf.reshape(tf.sequence_mask(mention_start + 1, self.num_words, dtype=tf.float32), [-1, 1])
      # entity_mask = tf.reshape(tf.sequence_mask(entity_index + 1, self._max_entity, dtype=tf.float32), [1, -1])
      # entity_mask_write = tf.reshape(tf.sequence_mask(entity_index, self._max_entity, dtype=tf.float32), [1, -1]) - self.non_entity_mask

      # entity_mention_score = tf.tile(current_score, [100, 1]) * tf.transpose(1 - entity_mask_write)

      start_score = tf.concat([tf.zeros([1, 1]), start_att], 0) + tf.log(start_mask)
      # top_antecedent = tf.argmax(tf.transpose(mention_att), axis=1)
      # hist_emb = tf.gather(self.sentence_emb_init, top_antecedent)
      # attention = tf.transpose(tf.nn.softmax(start_score, dim=0))
      # _, ante_vec = tf.split(attention, [1, self.k], 1)

      # '''
      hist_attn = tf.nn.softmax(start_score, dim=0)
      # hist_attn = tf.matmul(mention_att, tf.zeros([4, 10]))
      hist_emb = tf.reduce_sum(hist_attn * self.sentence_emb_init, 0)

      hist_emb = tf.tile(tf.reshape(hist_emb, [1, -1]), [self.num_words - 1, 1])
      # '''

      eword_emb = tf.concat([pair_feature, inputs_tiled, self.word_emb, hist_emb], 1)
      # eword_emb = tf.reshape(eword_emb, [self.num_words - 1, -1])

      end_att = ffnn_name(eword_emb, 2, 150, 1, 'end_att', self._dropout) + self.end_scores + context_mention_score

      end_mask = tf.reshape(tf.sequence_mask(mention_end + 1, self.num_words, dtype=tf.float32), [-1, 1])

      end_score = tf.concat([tf.zeros([1, 1]), end_att], 0) # + tf.log(end_mask)

      _, start_score = tf.split(start_score, [1, self.num_words - 1], 0)
      _, end_score = tf.split(end_score, [1, self.num_words - 1], 0)

      start_score = tf.reshape(start_score, [-1])
      end_score = tf.reshape(end_score, [-1])

      antecedent_score = tf.gather(start_score, self.mention_starts) + tf.gather(end_score, self.mention_ends)
      antecedent_score = tf.reshape(antecedent_score, [1, -1])
      new_h = tf.concat([tf.zeros([1, 1]), antecedent_score], 1)

      # ante_vec = tf.cond(tf.equal(tf.reshape(top_antecedent - 1, []), -1),
      #         lambda: tf.zeros([1, self.k]),
      #         lambda: tf.one_hot(top_antecedent - 1, self.k))
      
      '''
      span_vec = tf.transpose(tf.one_hot(word_index, self.k))

      add_A = tf.tile(ante_vec, [self.k, 1]) * span_vec

      new_A = A + add_A + tf.transpose(add_A)

      # D = A / tf.reduce_sum(new_A, axis=1)

      D = tf.sqrt(tf.diag(1 / tf.reduce_sum(new_A, axis=1)))

      DAD = tf.matmul(tf.matmul(D, new_A), D)

      new_memory = tf.reshape(tf.nn.relu(tf.matmul(DAD, sentence_emb)), [1, -1])

      new_A = tf.reshape(new_A, [1, -1])

      new_c = tf.concat([index + self.word_index_update, new_A, new_memory, padding], 1)
      new_h = tf.transpose(mention_att)
      '''

      new_c = c + self.word_index_update

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
