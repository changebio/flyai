import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq

import conf
from processor import Processor

input_seq_len = conf.input_seq_len
output_seq_len = conf.output_seq_len

processor = Processor()
'''
使用tensorflow实现自己的算法

'''
# 得到训练和测试的数据
# LSTM神经元size
size = 64
# 初始学习率
init_learning_rate = 0.0003
learning_rate_decay = 0.99
# 在样本中出现频率超过这个值(包括他自己)才会进入词表
# 模型操作辅助类

# 放在全局的位置，为了动态算出num_encoder_symbols和num_decoder_symbols
num_encoder_symbols = processor.length
num_decoder_symbols = processor.length

# ——————————————————导入数据——————————————————————

encoder_inputs = []
decoder_inputs = []
target_weights = []
for i in range(input_seq_len):
    encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
for i in range(output_seq_len + 1):
    decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
for i in range(output_seq_len):
    target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))

learning_rate = tf.Variable(float(init_learning_rate), trainable=False, dtype=tf.float32)
learning_rate_decay_op = learning_rate.assign(learning_rate * 0.99)


# ——————————————————定义神经网络变量——————————————————

def get_model(feed_previous=False):
    """构造模型
    """
    # decoder_inputs左移一个时序作为targets
    targets = [decoder_inputs[i + 1] for i in range(output_seq_len)]

    cell = tf.contrib.rnn.LSTMCell(size)

    # 这里输出的状态我们不需要
    outputs, _ = seq2seq.embedding_attention_seq2seq(
        encoder_inputs,
        decoder_inputs[:output_seq_len],
        cell,
        num_encoder_symbols=num_encoder_symbols,
        num_decoder_symbols=num_decoder_symbols,
        embedding_size=size,
        output_projection=None,
        feed_previous=feed_previous,
        dtype=tf.float32)

    # 计算加权交叉熵损失
    loss = seq2seq.sequence_loss(outputs, targets, target_weights)
    # 梯度下降优化器
    update = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    # 模型持久化
    saver = tf.train.Saver(tf.global_variables())

    return encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, learning_rate_decay_op, learning_rate
