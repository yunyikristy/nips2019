from __future__ import division
from __future__ import print_function

import prettytensor as pt
import time
import tensorflow as tf
import numpy as np
import scipy.misc
import os
import sys
from six.moves import range
from app_encoder import AppEncoder

import compact_bilinear_pooling as cbp

from misc.config import cfg
from misc.utils import mkdir_p


TINY = 1e-8

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = [g for g, _ in grad_and_vars if g != None]
        grad = tf.stack(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def average_losses(loss):
    tf.add_to_collection('losses', loss)
    losses = tf.get_collection('losses')
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n(losses + regularization_losses, name="total_loss")
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name="avg")
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return total_loss

class CondGANTrainer(object):
    def __init__(self,
                 model,
                 dataset=None,
                 exp_name="model",
                 ckt_logs_dir="ckt_logs",
                 ):
        """
        :type model: RegularizedGAN
        """
        self.model = model
        self.dataset = dataset
        self.exp_name = exp_name
        self.log_dir = ckt_logs_dir
        self.checkpoint_dir = ckt_logs_dir

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        self.model_path = cfg.TRAIN.PRETRAINED_MODEL

        self.log_vars = []

    def init_opt(self):
        '''Helper function for init_opt'''


        self.app_encoder = AppEncoder()

        with tf.device('/cpu:0'):
            self.g_lr = tf.placeholder(
                tf.float32, [],
                name='generator_learning_rate'
            )
            self.d_lr = tf.placeholder(
                tf.float32, [],
                name='discriminator_learning_rate'
            )

            g_opt = tf.train.AdamOptimizer(self.g_lr, beta1=0.5)
            d_opt = tf.train.AdamOptimizer(self.d_lr, beta1=0.5)
            self.models = []
            self.num_gpu = 2

            for gpu_id in range(self.num_gpu):
                with tf.device('/gpu:%d'%gpu_id):
                    with tf.name_scope('tower_%d'%gpu_id):
                        with tf.variable_scope('cpu_variables', reuse=gpu_id>0):
                            image = tf.placeholder(
                                tf.float32, [self.batch_size, 224, 224, 3],
                                name = 'image')

                            text = tf.placeholder(tf.float32, [self.batch_size, 1024], name='text')

                            label = tf.placeholder(
                                tf.float32, [self.batch_size, 200], name="label"
                            )
       

                            with tf.variable_scope("Inception", reuse=gpu_id>0):
                                image_app_7x7, image_app =\
                                    self.app_encoder.build(image)
                            image_app = tf.tile(image_app, [1, 7, 7, 1])

                            image_app_cbp = tf.reshape(cbp.compact_bilinear_pooling_layer(image_app_7x7, image_app, 800, sum_pool=False, sequential=False), (self.batch_size, 7, 7, 800))
                            image_app_cbp /= tf.reduce_max(image_app_cbp, axis=3, keep_dims=True)

                            text_fc_weights = tf.get_variable("text_fc_weight", [1024, 7*7*800], initializer=tf.truncated_normal_initializer(stddev=0.01))
                            text_fc_bias = tf.get_variable("text_fc_bias", [7*7*800], initializer=tf.constant_initializer(0.0))
                            text_app = tf.nn.bias_add(tf.matmul(text, text_fc_weights), text_fc_bias)
                            text_app = tf.reshape(text_app, [-1, 7, 7, 800])
                            text_app /= tf.reduce_max(text_app, axis=3, keep_dims=True)
       
                            with pt.defaults_scope(phase=pt.Phase.train):
                                with tf.variable_scope("g_net", reuse=gpu_id>0):    
                                    fake_t2t = self.model.get_generator(image_app_cbp)
                                with tf.variable_scope("g_net", reuse=True):
                                    fake_s2t = self.model.get_generator(text_app)
                            
                            with tf.variable_scope("Inception", reuse=True):
                                fake_s2t_app_7x7, fake_s2t_app =\
                                        self.app_encoder.build(fake_s2t)
                                fake_t2t_app_7x7, fake_t2t_app = \
                                        self.app_encoder.build(fake_t2t)

                            fake_s2t_app = tf.tile(fake_s2t_app, [1, 7, 7, 1])
                            fake_t2t_app = tf.tile(fake_t2t_app, [1, 7, 7, 1])

                            fake_s2t_app_cbp = tf.reshape(cbp.compact_bilinear_pooling_layer(fake_s2t_app_7x7, fake_s2t_app, 800, sum_pool=False, sequential=False), (self.batch_size, 7, 7, 800))
                            fake_s2t_app_cbp /= tf.reduce_max(fake_s2t_app_cbp, axis=3, keep_dims=True)
                            fake_t2t_app_cbp = tf.reshape(cbp.compact_bilinear_pooling_layer(fake_t2t_app_7x7, fake_t2t_app, 800, sum_pool=False, sequential=False), (self.batch_size, 7, 7, 800))
                            fake_t2t_app_cbp /= tf.reduce_max(fake_t2t_app_cbp, axis=3, keep_dims=True)


                            fake_s2t_app_pool = tf.nn.avg_pool(fake_s2t_app_cbp, [1,7,7,1], [1,1,1,1], padding='VALID')

                            fc_weights = tf.get_variable("fc_weight", [800, 200], initializer=tf.truncated_normal_initializer(stddev=0.01))
                            fc_bias = tf.get_variable("fc_bias", [200], initializer=tf.constant_initializer(0.0))

                            image_fc = tf.reshape(tf.nn.avg_pool(image_app_cbp, [1, 7, 7, 1], [1,1,1,1], padding='VALID'), (-1, 800))
                            image_fc = tf.nn.bias_add(tf.matmul(image_fc, fc_weights), fc_bias)
                            fake_t2t_fc = tf.reshape(tf.nn.avg_pool(fake_t2t_app_cbp, [1,7,7,1], [1,1,1,1], padding='VALID'), (-1, 800))
                            fake_t2t_fc = tf.nn.bias_add(tf.matmul(fake_t2t_fc, fc_weights), fc_bias)

                            real_logit = self.model.get_discriminator(image)
                            fake_s2t_logit = self.model.get_discriminator(fake_s2t)
                            fake_t2t_logit = self.model.get_discriminator(fake_t2t)

                            real_d_loss =\
                                tf.nn.softmax_cross_entropy_with_logits(real_logit,
                                                                tf.constant([[1.0, 0.0, 0.0]]*self.batch_size))
                            real_d_loss = tf.reduce_mean(real_d_loss)

                            fake_s2t_d_loss =\
                                tf.nn.softmax_cross_entropy_with_logits(fake_s2t_logit,
                                                                tf.constant([[0.0, 1.0, 0.0]]*self.batch_size))
                            fake_s2t_d_loss = tf.reduce_mean(fake_s2t_d_loss)

                            fake_t2t_d_loss =\
                                tf.nn.softmax_cross_entropy_with_logits(fake_t2t_logit,
                                                                tf.constant([[0.0, 0.0, 1.0]]*self.batch_size))
                            fake_t2t_d_loss = tf.reduce_mean(fake_t2t_d_loss)

                            d_loss =\
                                real_d_loss + (fake_s2t_d_loss + fake_t2t_d_loss) / 2.


                            fake_s2t_g_loss = \
                                tf.nn.softmax_cross_entropy_with_logits(fake_s2t_logit,
                                                                tf.constant([[1.0, 0.0, 0.0]]*self.batch_size))
                            fake_s2t_g_loss = tf.reduce_mean(fake_s2t_g_loss)

                            fake_t2t_g_loss = \
                                tf.nn.softmax_cross_entropy_with_logits(fake_t2t_logit,
                                                                tf.constant([[1.0, 0.0, 0.0]]*self.batch_size))
                            fake_t2t_g_loss = tf.reduce_mean(fake_t2t_g_loss)
       
                            f_t_loss = tf.nn.softmax_cross_entropy_with_logits(image_fc, label)
                            f_t_loss = tf.reduce_mean(f_t_loss)
                            f_t2t_loss = tf.nn.softmax_cross_entropy_with_logits(fake_t2t_fc, label)

                            f_t2t_loss = tf.reduce_mean(f_t2t_loss)
                            f_s2t_loss = tf.abs(fake_s2t_app_pool - text_app)
                            f_s2t_loss = tf.reduce_mean(f_s2t_loss)

                            debug_loss1 = f_t_loss
                            debug_loss2 = f_t2t_loss
                            debug_loss3 = f_s2t_loss

                            f_loss = (f_t_loss + f_t2t_loss + f_s2t_loss) / 3.

                            g_loss =\
                                 (fake_s2t_g_loss + fake_t2t_g_loss)/2. + 10. * f_loss

                            t_vars = tf.trainable_variables()
                            g_train_vars = []
                            d_train_vars = []

                            for var in t_vars:
                                if var.name.startswith('d_'):
                                    d_train_vars.append(var)
                                else:
                                    g_train_vars.append(var)
                            
        
                            d_grad = d_opt.compute_gradients(d_loss, var_list=d_train_vars)
                            g_grad = g_opt.compute_gradients(g_loss, var_list=g_train_vars)

                            self.models.append((image, text, label, fake_s2t, fake_t2t, g_loss, d_loss, g_grad, d_grad, debug_loss1, debug_loss2, debug_loss3))

 

            print('build model on gpu tower done')
            _, _, _, _, _, tower_g_loss, tower_d_loss, tower_g_grad, tower_d_grad, loss1, loss2, loss3 = zip(*self.models)

            self.aver_d_loss = tf.reduce_mean(tower_d_loss)
            self.aver_g_loss = tf.reduce_mean(tower_g_loss)
            self.d_op = d_opt.apply_gradients(average_gradients(tower_d_grad))
            self.g_op = g_opt.apply_gradients(average_gradients(tower_g_grad))

            self.loss1 = tf.reduce_mean(loss1)
            self.loss2 = tf.reduce_mean(loss2)
            self.loss3 = tf.reduce_mean(loss3)

    def build_model(self, sess):
        self.init_opt()
        sess.run(tf.initialize_all_variables())

        tvars = tf.all_variables()
        weights = np.load('stageI_icml/pretrained/app_encoder.npy').item()
        ops = []
        for var in tvars:
            if var.name.startswith('I'):
                try:
                    ops.append(tf.assign(var, weights[var.name]))
                except:
                    print(var.name)
        sess.run(ops)

        return 0 

    def train(self):
        np.set_printoptions(threshold='nan')
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            #with tf.device("/gpu:%d" % cfg.GPU_ID):
            counter = self.build_model(sess)
            #counter = 0
            saver = tf.train.Saver(tf.all_variables(),
                                   keep_checkpoint_every_n_hours=2)
            #saver.restore(sess, "ckt_logs/birds/stageI_2018_02_05_17_17_18/model_6000.ckpt")

            g_lr = cfg.TRAIN.GENERATOR_LR
            d_lr = cfg.TRAIN.DISCRIMINATOR_LR
            num_embedding = cfg.TRAIN.NUM_EMBEDDING
            lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH
            number_example = self.dataset.train._num_examples
            updates_per_epoch = int(number_example * 1.0/ self.batch_size / self.num_gpu) + 1
            #epoch_start = int(counter / updates_per_epoch)
            epoch_start = 0
            for i in range(epoch_start):
                if i % lr_decay_step == 0 and i != 0:
                    g_lr * 0.5
                    d_lr * 0.5

            for epoch in range(epoch_start, self.max_epoch):

                if epoch % lr_decay_step == 0 and epoch != 0:
                    g_lr *= 0.5
                    d_lr *= 0.5

                for i in range(updates_per_epoch):
                    feed_dict = {
                        self.g_lr: g_lr,
                        self.d_lr: d_lr
                    }
                    for j in range(len(self.models)):
                        batch_images, batch_embedding, batch_label =\
                            self.dataset.train.next_batch(self.batch_size)

                        feed_dict[self.models[j][0]] = batch_images
                        feed_dict[self.models[j][1]] = batch_embedding
                        feed_dict[self.models[j][2]] = batch_label
                    
                    _, d_loss = sess.run([self.d_op, self.aver_d_loss], feed_dict)
                    sess.run(self.g_op, feed_dict)
                    sess.run(self.g_op, feed_dict)
                    _, g_loss, loss1, loss2, loss3 = sess.run([self.g_op, self.aver_g_loss, self.loss1, self.loss2, self.loss3], feed_dict)

                    if i % 20 == 0:
                        s2t, t2t = sess.run([self.models[0][3], self.models[0][4]], feed_dict)
                        imgs = np.concatenate([batch_images[0], s2t[0], t2t[0]], axis=1)
                        scipy.misc.imsave('icml_bird/%03d_%04d.jpg'%(epoch, i), imgs)

                    now = time.time()
                    now = time.localtime(now)
                    now = time.strftime('%H-%M-%S', now)
                    print('[%s] epoch %d iter %d g_loss %f d_loss %f f_loss %f + %f + %f'
                            %(now, epoch, i,  g_loss, d_loss, loss1, loss2, loss3))

                    # save checkpoint
                    counter += 1
                    if counter % 500 == 0:
                        snapshot_path = "%s/%s_%s.ckpt" %\
                                         (self.checkpoint_dir,
                                          self.exp_name,
                                          str(counter))
                        fn = saver.save(sess, snapshot_path)
                        print("Model saved in file: %s" % fn)
