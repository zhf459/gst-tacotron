import tensorflow as tf


def split_heads(inputs, num_heads):
  """Splits a tensor in depth.
  Args:
    inputs: A ``tf.Tensor`` of shape :math:`[B, T, D]`.
    num_heads: The number of heads :math:`H`.
  Returns:
    A ``tf.Tensor`` of shape :math:`[B, H, T, D / H]`.
  """
  static_shape = inputs.get_shape().as_list()
  depth = static_shape[-1]
  outputs = tf.reshape(
    inputs, [tf.shape(inputs)[0], tf.shape(inputs)[1], num_heads, depth // num_heads])
  outputs = tf.transpose(outputs, perm=[0, 2, 1, 3])
  return outputs


def combine_heads(inputs):
  """Concatenates heads.
  Args:
    inputs: A ``tf.Tensor`` of shape :math:`[B, H, T, D]`.
  Returns:
    A ``tf.Tensor`` of shape :math:`[B, T, D * H]`.
  """
  static_shape = inputs.get_shape().as_list()
  depth = static_shape[-1]
  num_heads = static_shape[1]
  outputs = tf.transpose(inputs, perm=[0, 2, 1, 3])
  outputs = tf.reshape(inputs, [tf.shape(outputs)[0], tf.shape(outputs)[1], depth * num_heads])
  return outputs


def dot_product_attention(queries, keys, values):
  """Computes the dot product attention.
  Args:
    queries: The sequence of queries. A tensor of shape :math:`[B, T_1, ...]`.
    keys: The sequence use to calculate attention scores. A tensor of shape
      :math:`[B, T_2, ...]`.
    values: The sequence to attend. A tensor of shape :math:`[B, T_2, ...]`.
  Returns:
    A tuple ``(context vector, attention vector)``.
  """
  # Dot product between queries and keys.
  dot = tf.matmul(queries, keys, transpose_b=True)
  # Compute attention weights.
  attn = tf.nn.softmax(dot)
  # Compute attention context.
  context = tf.matmul(attn, values)

  return context, attn


def additive_attention(queries, keys, values):
  """Computes the additive (mlp) attention.
  Args:
    queries: The sequence of queries. A tensor of shape :math:`[B, T_1, ...]`.
    keys: The sequence use to calculate attention scores. A tensor of shape
      :math:`[B, T_2, ...]`.
    values: The sequence to attend. A tensor of shape :math:`[B, T_2, ...]`.
    mask: A ``tf.Tensor`` applied to the dot product.
    dropout: The probability to drop units from the inputs.
  Returns:
    A tuple ``(context vector, attention vector)``.
  """
  # Get the number of hidden units from the trailing dimension of keys
  num_units = queries.get_shape()[-1].value
  dtype = queries.dtype

  v = tf.get_variable("attention_v", [num_units], dtype=dtype)
  # Singlelayer multilayer perceptron.
  add = tf.reduce_sum(v * tf.tanh(keys + queries), [-1], keepdims=True)
  # Compute attention weights.
  attn = tf.nn.softmax(add)
  # Compute attention context.
  context = tf.matmul(attn, values, transpose_a=True)

  return context, attn


def multi_head_attention(num_heads,
                         queries,
                         memory,
                         num_units=None,
                         attention_type='additive',
                         scope='style_token_layer'):
  """Computes the multi-head attention as described in
  https://arxiv.org/abs/1706.03762.
  Args:
    num_heads: The number of attention heads.
    queries: The sequence of queries. A tensor of shape :math:`[B, T_1, ...]`.
    memory: The sequence to attend. A tensor of shape :math:`[B, T_2, ...]`.
      If ``None``, computes self-attention.
    num_units: The number of hidden units. If not set, it is set to the input
      dimension.
    attention_type: a string, either "dot_product", "additive".
  Returns:
    The concatenated attention context of each head.
  """
  with tf.variable_scope(scope):
    num_units = num_units or queries.get_shape().as_list()[-1]

    if num_units % num_heads != 0:
      raise ValueError("Multi head attention requires that num_units is a"
                       " multiple of {}".format(num_heads))

    queries = tf.layers.conv1d(queries, num_units, 1)
    keys = tf.layers.conv1d(memory, num_units, 1)
    values = memory

    queries = split_heads(queries, num_heads)
    keys = split_heads(keys, num_heads)
    values = split_heads(values, num_heads)

    if attention_type == 'additive':
      heads, _ = additive_attention(queries, keys, values)
    elif attention_type == 'dot_product':
      queries *= (num_units // num_heads)**-0.5
      heads, _ = dot_product_attention(queries, keys, values)
    else:
      raise ValueError('Only additive and dot_product attention are supported')

    # Concatenate all heads output.
    combined = combine_heads(heads)

    return combined
