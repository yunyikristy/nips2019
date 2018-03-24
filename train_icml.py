#-*-coding:utf-8-*-

import tensorflow as tf
import compact_bilinear_pooling as cbp
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import scipy.io as sio
import scipy
import cv2


f_weights = {
    'conv_1_weights': tf.get_variable('f_conv_1_weights', [5, 5, 3, 32], initializer=tf.truncated_normal_initializer(stddev=0.02)),
    'conv_2_weights': tf.get_variable('f_conv_2_weights', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02)),
    'fc1_weights': tf.get_variable('f_fc1_weights', [7*7*128, 512], initializer=tf.truncated_normal_initializer(stddev=0.02)),
    'fc2_weights': tf.get_variable('f_fc2_weights', [512, 10], initializer=tf.truncated_normal_initializer(stddev=0.02))
}
f_biases = {
    'conv_1_biases': tf.get_variable('f_conv_1_biases', [32], initializer=tf.constant_initializer(0.1)),
    'conv_2_biases': tf.get_variable('f_conv_2_biases', [64], initializer=tf.constant_initializer(0.1)),
    'fc1_biases': tf.get_variable('f_fc1_biases', [512], initializer=tf.constant_initializer(0.1)),
    'fc2_biases': tf.get_variable('f_fc2_biases', [10], initializer=tf.constant_initializer(0.1))
}


d_weights = {
    'conv_1_weights': tf.get_variable('d_conv_1_weights', [5, 5, 3, 32], initializer=tf.truncated_normal_initializer(stddev=0.02)),
    'conv_2_weights': tf.get_variable('d_conv_2_weights', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02)),
    'fc1_weights': tf.get_variable('d_fc1_weights', [7*7*64, 512], initializer=tf.truncated_normal_initializer(stddev=0.02)),
    'fc2_weights': tf.get_variable('d_fc2_weights', [512, 3], initializer=tf.truncated_normal_initializer(stddev=0.02))
}
d_biases = {
    'conv_1_biases': tf.get_variable('d_conv_1_biases', [32], initializer=tf.constant_initializer(0.1)),
    'conv_2_biases': tf.get_variable('d_conv_2_biases', [64], initializer=tf.constant_initializer(0.1)),
    'fc1_biases': tf.get_variable('d_fc1_biases', [512], initializer=tf.constant_initializer(0.1)),
    'fc2_biases': tf.get_variable('d_fc2_biases', [3], initializer=tf.constant_initializer(0.1))
}

g_weights = {
    'conv_1_weights': tf.get_variable('g_conv_1_weights', [5, 5, 128, 64], initializer=tf.truncated_normal_initializer(stddev=0.02)),
    'conv_2_weights': tf.get_variable('g_conv_2_weights', [5, 5, 64, 3], initializer=tf.truncated_normal_initializer(stddev=0.02)),
}
g_biases = {
    'conv_1_biases': tf.get_variable('g_conv_1_biases', [8], initializer=tf.constant_initializer(0.1)),
    'conv_2_biases': tf.get_variable('g_conv_2_biases', [3], initializer=tf.constant_initializer(0.1)),
}


batch_size = 100

svhn_ori = tf.placeholder(tf.float32, [batch_size, 32, 32, 3])
svhn = tf.image.resize_images(svhn_ori, [28, 28])
mnist = tf.placeholder(tf.float32, [batch_size, 28, 28, 3])
label = tf.placeholder(tf.float32, [batch_size, 10])

with tf.variable_scope("LeNet"):
    mnist_conv1 = tf.nn.conv2d(mnist, f_weights['conv_1_weights'], strides=[1,1,1,1], padding='SAME')
    mnist_relu1 = tf.nn.relu(tf.nn.bias_add(mnist_conv1, f_biases['conv_1_biases']))
    mnist_relu1 = tf.nn.relu(mnist_conv1)
    mnist_pool1 = tf.nn.max_pool(mnist_relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    mnist_conv2 = tf.nn.conv2d(mnist_pool1, f_weights['conv_2_weights'], strides=[1,1,1,1], padding='SAME')
    mnist_relu2 = tf.nn.relu(mnist_conv2)
    mnist_relu2 = tf.nn.relu(tf.nn.bias_add(mnist_conv2, f_biases['conv_2_biases']))
    mnist_pool2 = tf.nn.max_pool(mnist_relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    mnist_gpool = tf.nn.avg_pool(mnist_pool2, ksize=[1,7,7,1], strides=[1,1,1,1], padding='VALID')
    mnist_gpool = tf.tile(mnist_gpool, [1, 7, 7, 1])
    mnist_cbp = tf.reshape(cbp.compact_bilinear_pooling_layer(mnist_pool2, mnist_gpool, 128, sum_pool=False, sequential=False), [batch_size, 7, 7, 128]) / 10000.
    mnist_flatten = tf.reshape(mnist_cbp, [batch_size, 7*7*128])
    mnist_fc1 = tf.nn.relu(tf.matmul(mnist_flatten, f_weights['fc1_weights']) + f_biases['fc1_biases'])
    mnist_fc2 = tf.nn.relu(tf.matmul(mnist_fc1, f_weights['fc2_weights']) + f_biases['fc2_biases'])

with tf.variable_scope("LeNet", reuse=True):
    svhn_conv1 = tf.nn.conv2d(svhn, f_weights['conv_1_weights'], strides=[1,1,1,1], padding='SAME')
    svhn_relu1 = tf.nn.relu(tf.nn.bias_add(svhn_conv1, f_biases['conv_1_biases']))
    svhn_pool1 = tf.nn.max_pool(svhn_relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    svhn_conv2 = tf.nn.conv2d(svhn_pool1, f_weights['conv_2_weights'], strides=[1,1,1,1], padding='SAME')
    svhn_relu2 = tf.nn.relu(tf.nn.bias_add(svhn_conv2, f_biases['conv_2_biases']))
    svhn_pool2 = tf.nn.max_pool(svhn_relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    svhn
    svhn_gpool = tf.nn.avg_pool(svhn_pool2, ksize=[1,7,7,1], strides=[1,1,1,1], padding='VALID')
    svhn_gpool = tf.tile(svhn_gpool, [1, 7, 7, 1])
    svhn_cbp = tf.reshape(cbp.compact_bilinear_pooling_layer(svhn_pool2, svhn_gpool, 128, sum_pool=False, sequential=False), [batch_size, 7, 7, 128]) / 10000.


with tf.variable_scope("g_net"):
    mnist_resize1 = tf.image.resize_nearest_neighbor(mnist_cbp, [14, 14])
    mnist_deconv1 = tf.nn.conv2d(mnist_resize1, g_weights['conv_1_weights'], strides=[1,1,1,1], padding='SAME')
    #mnist_derelu1 = tf.nn.relu(tf.nn.bias_add(mnist_deconv1, g_biases['conv_1_biases']))
    mnist_derelu1 = tf.nn.relu(mnist_deconv1)
    mnist_resize2 = tf.image.resize_nearest_neighbor(mnist_derelu1, [28, 28])
    mnist_deconv2 = tf.nn.conv2d(mnist_resize2, g_weights['conv_2_weights'], strides=[1,1,1,1], padding='SAME')
    mnist_tanh = tf.nn.tanh(tf.nn.bias_add(mnist_deconv2, g_biases['conv_2_biases']))
    fake_m2m = (mnist_tanh + 1) / 2. * 255

with tf.variable_scope("g_net", reuse=True):
    svhn_resize1 = tf.image.resize_nearest_neighbor(svhn_cbp, [14, 14])
    svhn_deconv1 = tf.nn.conv2d(svhn_resize1, g_weights['conv_1_weights'], strides=[1,1,1,1], padding='SAME')
    #svhn_derelu1 = tf.nn.relu(tf.nn.bias_add(svhn_deconv1, g_biases['conv_1_biases']))
    svhn_derelu1 = tf.nn.relu(svhn_deconv1)
    svhn_resize2 = tf.image.resize_nearest_neighbor(svhn_derelu1, [28, 28])
    svhn_deconv2 = tf.nn.conv2d(svhn_resize2, g_weights['conv_2_weights'], strides=[1,1,1,1], padding='SAME')
    svhn_tanh = tf.nn.tanh(tf.nn.bias_add(svhn_deconv2, g_biases['conv_2_biases']))
    fake_s2m = (svhn_tanh + 1) / 2. * 255

with tf.variable_scope("d_net"):
    real_conv1 = tf.nn.conv2d(mnist, d_weights['conv_1_weights'], strides=[1,1,1,1], padding='SAME')
    real_relu1 = tf.nn.relu(tf.nn.bias_add(real_conv1, d_biases['conv_1_biases']))
    real_pool1 = tf.nn.max_pool(real_relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    real_conv2 = tf.nn.conv2d(real_pool1, d_weights['conv_2_weights'], strides=[1,1,1,1], padding='SAME')
    real_relu2 = tf.nn.relu(tf.nn.bias_add(real_conv2, d_biases['conv_2_biases']))
    real_pool2 = tf.nn.max_pool(real_relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    real_flatten = tf.reshape(real_pool2, [batch_size, 7*7*64])
    real_fc1 = tf.nn.relu(tf.matmul(real_flatten, d_weights['fc1_weights']) + d_biases['fc1_biases'])
    real_fc2 = tf.nn.relu(tf.matmul(real_fc1, d_weights['fc2_weights']) + d_biases['fc2_biases'])

with tf.variable_scope("d_net", reuse=True):
    fake_m2m_d_conv1 = tf.nn.conv2d(fake_m2m, d_weights['conv_1_weights'], strides=[1,1,1,1], padding='SAME')
    fake_m2m_d_relu1 = tf.nn.relu(tf.nn.bias_add(fake_m2m_d_conv1, d_biases['conv_1_biases']))
    fake_m2m_d_pool1 = tf.nn.max_pool(fake_m2m_d_relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    fake_m2m_d_conv2 = tf.nn.conv2d(fake_m2m_d_pool1, d_weights['conv_2_weights'], strides=[1,1,1,1], padding='SAME')
    fake_m2m_d_relu2 = tf.nn.relu(tf.nn.bias_add(fake_m2m_d_conv2, d_biases['conv_2_biases']))
    fake_m2m_d_pool2 = tf.nn.max_pool(fake_m2m_d_relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    fake_m2m_d_flatten = tf.reshape(fake_m2m_d_pool2, [batch_size, 7*7*64])
    fake_m2m_d_fc1 = tf.nn.relu(tf.matmul(fake_m2m_d_flatten, d_weights['fc1_weights']) + d_biases['fc1_biases'])
    fake_m2m_d_fc2 = tf.nn.relu(tf.matmul(fake_m2m_d_fc1, d_weights['fc2_weights']) + d_biases['fc2_biases'])

with tf.variable_scope("d_net", reuse=True):
    fake_s2m_d_conv1 = tf.nn.conv2d(fake_s2m, d_weights['conv_1_weights'], strides=[1,1,1,1], padding='SAME')
    fake_s2m_d_relu1 = tf.nn.relu(tf.nn.bias_add(fake_s2m_d_conv1, d_biases['conv_1_biases']))
    fake_s2m_d_pool1 = tf.nn.max_pool(fake_s2m_d_relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    fake_s2m_d_conv2 = tf.nn.conv2d(fake_s2m_d_pool1, d_weights['conv_2_weights'], strides=[1,1,1,1], padding='SAME')
    fake_s2m_d_relu2 = tf.nn.relu(tf.nn.bias_add(fake_s2m_d_conv2, d_biases['conv_2_biases']))
    fake_s2m_d_pool2 = tf.nn.max_pool(fake_s2m_d_relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    fake_s2m_d_flatten = tf.reshape(fake_s2m_d_pool2, [batch_size, 7*7*64])
    fake_s2m_d_fc1 = tf.nn.relu(tf.matmul(fake_s2m_d_flatten, d_weights['fc1_weights']) + d_biases['fc1_biases'])
    fake_s2m_d_fc2 = tf.nn.relu(tf.matmul(fake_s2m_d_fc1, d_weights['fc2_weights']) + d_biases['fc2_biases'])

with tf.variable_scope("LeNet", reuse=True):
    fake_s2m_conv1 = tf.nn.conv2d(fake_s2m, f_weights['conv_1_weights'], strides=[1,1,1,1], padding='SAME')
    fake_s2m_relu1 = tf.nn.relu(tf.nn.bias_add(fake_s2m_conv1, f_biases['conv_1_biases']))
    fake_s2m_pool1 = tf.nn.max_pool(fake_s2m_relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    fake_s2m_conv2 = tf.nn.conv2d(fake_s2m_pool1, f_weights['conv_2_weights'], strides=[1,1,1,1], padding='SAME')
    fake_s2m_relu2 = tf.nn.relu(tf.nn.bias_add(fake_s2m_conv2, f_biases['conv_2_biases']))
    fake_s2m_pool2 = tf.nn.max_pool(fake_s2m_relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    fake_s2m_gpool = tf.nn.avg_pool(fake_s2m_pool2, ksize=[1,7,7,1], strides=[1,1,1,1], padding='VALID')
    fake_s2m_gpool = tf.tile(fake_s2m_gpool, [1, 7, 7, 1])
    fake_s2m_cbp = tf.reshape(cbp.compact_bilinear_pooling_layer(fake_s2m_pool2, svhn_gpool, 128, sum_pool=False, sequential=False), [batch_size, 7, 7, 128]) / 10000.

with tf.variable_scope("LeNet", reuse=True):
    fake_m2m_conv1 = tf.nn.conv2d(fake_m2m, f_weights['conv_1_weights'], strides=[1,1,1,1], padding='SAME')
    fake_m2m_relu1 = tf.nn.relu(tf.nn.bias_add(fake_m2m_conv1, f_biases['conv_1_biases']))
    fake_m2m_pool1 = tf.nn.max_pool(fake_m2m_relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    fake_m2m_conv2 = tf.nn.conv2d(fake_m2m_pool1, f_weights['conv_2_weights'], strides=[1,1,1,1], padding='SAME')
    fake_m2m_relu2 = tf.nn.relu(tf.nn.bias_add(fake_m2m_conv2, f_biases['conv_2_biases']))
    fake_m2m_pool2 = tf.nn.max_pool(fake_m2m_relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    fake_m2m_gpool = tf.nn.avg_pool(fake_m2m_pool2, ksize=[1,7,7,1], strides=[1,1,1,1], padding='VALID')
    fake_m2m_gpool = tf.tile(fake_m2m_gpool, [1, 7, 7, 1])
    fake_m2m_cbp = tf.reshape(cbp.compact_bilinear_pooling_layer(fake_m2m_pool2, mnist_gpool, 128, sum_pool=False, sequential=False), [batch_size, 7, 7, 128]) / 100000.
    fake_m2m_flatten = tf.reshape(fake_m2m_cbp, [batch_size, 7*7*128])
    fake_m2m_fc1 = tf.nn.relu(tf.matmul(fake_m2m_flatten, f_weights['fc1_weights']) + f_biases['fc1_biases'])
    fake_m2m_fc2 = tf.nn.relu(tf.matmul(fake_m2m_fc1, f_weights['fc2_weights']) + f_biases['fc2_biases'])


mnist_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=mnist_fc2, labels=label))
fake_m2m_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fake_m2m_fc2, labels=label))
fake_s2m_loss = tf.reduce_mean(tf.abs(fake_s2m_cbp - svhn_cbp))

real_d_loss = tf.nn.softmax_cross_entropy_with_logits(logits=real_fc2, labels=tf.constant([[1.0, 0.0, 0.0]] * batch_size))
real_d_loss = tf.reduce_mean(real_d_loss)
fake_m2m_d_loss = tf.nn.softmax_cross_entropy_with_logits(logits=fake_m2m_d_fc2, labels=tf.constant([[0.0, 1.0, 0.0]] * batch_size))
fake_m2m_d_loss = tf.reduce_mean(fake_m2m_d_loss)
fake_s2m_d_loss = tf.nn.softmax_cross_entropy_with_logits(logits=fake_s2m_d_fc2, labels=tf.constant([[0.0, 0.0, 1.0]] * batch_size))
fake_s2m_d_loss = tf.reduce_mean(fake_s2m_d_loss)

fake_m2m_g_loss = tf.nn.softmax_cross_entropy_with_logits(logits=fake_m2m_d_fc2, labels=tf.constant([[1.0, 0.0, 0.0]] * batch_size))
fake_m2m_g_loss = tf.reduce_mean(fake_m2m_g_loss)
fake_s2m_g_loss = tf.nn.softmax_cross_entropy_with_logits(logits=fake_s2m_d_fc2, labels=tf.constant([[1.0, 0.0, 0.0]] * batch_size))
fake_s2m_g_loss = tf.reduce_mean(fake_s2m_g_loss)

f_loss= (mnist_loss + fake_m2m_loss + fake_s2m_loss) / 3.
#f_loss = mnist_loss + fake_m2m_loss
g_loss = (fake_m2m_g_loss + fake_s2m_g_loss) / 2. + 5. * f_loss
#g_loss = (fake_m2m_g_loss + fake_s2m_g_loss) / 2.
d_loss = real_d_loss + (fake_s2m_d_loss  + fake_m2m_d_loss) * 2.

all_vars = tf.trainable_variables()
for key in all_vars:
    print key
g_vars = [var for var in all_vars if not var.name.startswith('d_')]
#g_vars = [var for var in all_vars if var.name.startswith('g_')]
d_vars = [var for var in all_vars if var.name.startswith('d_')]


g_lr = 0.0002
d_lr = 0.0002

g_trainer = tf.train.AdamOptimizer(g_lr, beta1=0.5).minimize(g_loss, var_list=g_vars)
d_trainer = tf.train.AdamOptimizer(d_lr, beta1=0.5).minimize(d_loss, var_list=d_vars)


mnist_data = input_data.read_data_sets("/tmp/data", one_hot=True)
mnist_data = mnist_data.train
mnist_data_images = mnist_data.images #(55000, 784) 
mnist_data_labels = mnist_data.labels
mnist_data_images = mnist_data_images * 255.
mnist_data_images = np.reshape(mnist_data_images, (-1, 28, 28, 1))
mnist_data_images = np.tile(mnist_data_images, [1, 1, 1, 3])

svhn_data = sio.loadmat("./svhn/train_32x32.mat")
svhn_data_images = svhn_data['X'].astype(np.float32)
svhn_data_images = np.transpose(svhn_data_images, (3, 0, 1, 2))

print mnist_data_images.shape, svhn_data_images.shape

mnist_ids = np.arange(mnist_data_images.shape[0])
svhn_ids = np.arange(svhn_data_images.shape[0])
loops = mnist_data_images.shape[0] / batch_size

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver(tf.all_variables())
    saver2 = tf.train.Saver(tf.trainable_variables())
    saver2.restore(sess, ('models_g/model-14'))
    for epoch in range(100):
    
        np.random.shuffle(mnist_ids)
        for i in range(loops):
            current_ids = mnist_ids[i*batch_size:(i+1)*batch_size]
            batch_mnist_image = mnist_data_images[current_ids, :, :, :]
            batch_mnist_label = mnist_data_labels[current_ids]
            random_svhn = np.random.choice(svhn_ids, batch_size)
            batch_svhn_image = svhn_data_images[random_svhn, :, :, :]

            feed = {
                mnist: batch_mnist_image,
                label: batch_mnist_label,
                svhn_ori: batch_svhn_image
            }
            _, debug_d_loss = sess.run([d_trainer, d_loss], feed)
            sess.run(g_trainer, feed)
            sess.run(g_trainer, feed)
            _, debug_g_loss, debug_mnist_loss, debug_fake_s2m_loss, debug_fake_m2m_loss = sess.run([g_trainer, g_loss, mnist_loss, fake_s2m_loss, fake_m2m_loss], feed)

            #tmp1, tmp2 = sess.run([real_fc2, fake_m2m_fc2], feed)
            #print tmp1[0], tmp2[0]

            print 'epoch %d  iter %d  d_loss %f g_loss %f f_loss %f + %f + %f'%(epoch, i, debug_d_loss, debug_g_loss, debug_mnist_loss, debug_fake_s2m_loss, debug_fake_m2m_loss)

            if i % 100 == 0:
                tmp1, tmp2 = sess.run([fake_s2m, fake_m2m], feed)
                svhn_28x28 = cv2.resize(batch_svhn_image[0], (28, 28))
                res = np.concatenate([batch_mnist_image[0], svhn_28x28, tmp1[0], tmp2[0]], axis=1)
                scipy.misc.imsave('icml_mnist/%03d_%04d.jpg'%(epoch, i), res)
        saver.save(sess, 'models/model-%s'%epoch)
