"""Implements a voxel flow model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils.loss_utils import l1_loss, l2_loss, vae_loss, l1_regularizer
from utils.loss_utils import l1_charbonnier_loss, l1_charbonnier
from utils.geo_layer_utils import vae_gaussian_layer
from utils.geo_layer_utils import bilinear_interp
from utils.geo_layer_utils import meshgrid

'''TODO
Check regularization terms again;
- Check Charbonnier penalty function
'''


FLAGS = tf.app.flags.FLAGS

class Voxel_flow_model(object):
  def __init__(self, is_train=True):
    self.is_train = is_train

  def inference(self, input_images):
    """Inference on a set of input_images.
    Args:
    """
    return self._build_model(input_images) 

  def loss(self, predictions, flow_motion, flow_mask, targets, 
           lambda_motion, lambda_mask, epsilon):
    """Compute the necessary loss for training.
    """
    # corrected l1 regularized loss
    # self.reproduction_loss = l1_loss(predictions, targets) \
                  # + lambda_motion * l1_regularizer(flow_motion) \
                  # + lambda_mask * l1_regularizer(flow_mask)

    # Charbonnier loss
    self.reproduction_loss = l1_charbonnier_loss(predictions, targets, epsilon) \
                  + lambda_motion * l1_charbonnier(flow_motion, epsilon) \
                  + lambda_mask * l1_charbonnier(flow_mask, epsilon)
# 
    # # Charbonnier regularization with l1 loss
    # self.reproduction_loss = l1_loss(predictions, targets) \
    #               + lambda_motion * l1_charbonnier(flow_motion, epsilon) \
    #               + lambda_mask * l1_charbonnier(flow_mask, epsilon)

    return self.reproduction_loss

  def coarse_loss(self, predictions, targets, epsilon):
    """
    computes loss for coarser scales
    """
    return(l1_charbonnier_loss(predictions, targets, epsilon))

  def _build_model(self, input_images):
    """Build a VAE model.
    Args:
    """

    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.elu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)):
      
      # Define network     
      batch_norm_params = {
        'decay': 0.9997,
        'epsilon': 0.001,
        'is_training': self.is_train,
      }
      with slim.arg_scope([slim.batch_norm], is_training = self.is_train, updates_collections=None):
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
          normalizer_params=batch_norm_params):

          # scale 64 x 64
          input_images_64 = tf.image.resize_bilinear(input_images, [64, 64])
          conv_1c = slim.conv2d(input_images_64, 64, [5, 5], stride=1, scope='conv1c')
          pool_1c = slim.max_pool2d(conv_1c, [2, 2], scope='pool1c')
          conv_2c = slim.conv2d(pool_1c, 128, [5, 5], stride=1, scope='conv2c')
          pool_2c = slim.max_pool2d(conv_2c, [2, 2], scope='pool2c')
          conv_3c = slim.conv2d(pool_2c, 256, [3, 3], stride=1, scope='conv3c')
          pool_3c = slim.max_pool2d(conv_3c, [2, 2], scope='pool3c')
          bottleneck_c = slim.conv2d(pool_3c, 256, [3, 3], stride=1, scope='bottleneck_c')
          # decoders
          upsamp_1c = tf.image.resize_bilinear(bottleneck_c, [16, 16])
          deconv_1c = slim.conv2d(tf.concat([upsamp_1c, conv_3c], axis=3), 
                        256, [3, 3], stride=1, scope='deconv1c')
          upsamp_2c = tf.image.resize_bilinear(deconv_1c, [32, 32])
          deconv_2c = slim.conv2d(tf.concat([upsamp_2c, conv_2c], axis=3), 
                        128, [5, 5], stride=1, scope='deconv2c')
          # upsample to input dimensions
          upsamp_3c = tf.image.resize_bilinear(deconv_2c, [64, 64])
          deconv_3c = slim.conv2d(tf.concat([upsamp_3c, conv_1c], axis=3), 
                        64, [5, 5], stride=1, scope='deconv_3c')
          flow_64 = slim.conv2d(deconv_3c, 3, [5, 5], stride=1, scope='flow_64')


          # scale 128 x 128
          input_images_128 = tf.image.resize_bilinear(input_images, [128, 128])
          conv_1b = slim.conv2d(input_images_128, 64, [5, 5], stride=1, scope='conv1b')
          pool_1b = slim.max_pool2d(conv_1b, [2, 2], scope='pool1b')
          conv_2b = slim.conv2d(pool_1b, 128, [5, 5], stride=1, scope='conv2b')
          pool_2b = slim.max_pool2d(conv_2b, [2, 2], scope='pool2b')
          conv_3b = slim.conv2d(pool_2b, 256, [3, 3], stride=1, scope='conv3b')
          pool_3b = slim.max_pool2d(conv_3b, [2, 2], scope='pool3b')
          bottleneck_b = slim.conv2d(pool_3b, 256, [3, 3], stride=1, scope='bottleneck_b')
          # decoders
          upsamp_1b = tf.image.resize_bilinear(bottleneck_b, [32, 32])
          deconv_1b = slim.conv2d(tf.concat([upsamp_1b, conv_3b], axis=3), 
                        256, [3, 3], stride=1, scope='deconv1b')
          upsamp_2b = tf.image.resize_bilinear(deconv_1b, [64, 64])
          deconv_2b = slim.conv2d(tf.concat([upsamp_2b, conv_2b], axis=3), 
                        128, [5, 5], stride=1, scope='deconv2b')
          # upsample to input dimensions
          upsamp_3b = tf.image.resize_bilinear(deconv_2b, [128, 128])
          deconv_3b = slim.conv2d(tf.concat([upsamp_3b, conv_1b], axis=3), 
                        64, [5, 5], stride=1, scope='deconv3b')
          # concatenate w/ coarser scale
          deconv_64b = tf.image.resize_bilinear(flow_64[:, :, :, :2], [128, 128])
          deconv_64b = slim.conv2d(deconv_64b, 32, [5, 5], stride=1, scope='deconv_64b')
          flow_128 = slim.conv2d(tf.concat([deconv_64b, deconv_3b], axis=3), 3,
                                 [5, 5], stride=1, scope='flow_128')


          # scale 256 x 256
          # encoders
          conv_1a = slim.conv2d(input_images, 64, [5, 5], stride=1, scope='conv1a')
          pool_1a = slim.max_pool2d(conv_1a, [2, 2], scope='pool1a')
          conv_2a = slim.conv2d(pool_1a, 128, [5, 5], stride=1, scope='conv2a')
          pool_2a = slim.max_pool2d(conv_2a, [2, 2], scope='pool2a')
          conv_3a = slim.conv2d(pool_2a, 256, [3, 3], stride=1, scope='conv3a')
          pool_3a = slim.max_pool2d(conv_3a, [2, 2], scope='pool3a')
          bottleneck_a = slim.conv2d(pool_3a, 256, [3, 3], stride=1, scope='bottleneck_a')
          # decoders
          upsamp_1a = tf.image.resize_bilinear(bottleneck_a, [64, 64])
          deconv_1a = slim.conv2d(tf.concat([upsamp_1a, conv_3a], axis=3), 
                        256, [3, 3], stride=1, scope='deconv1a')
          upsamp_2a = tf.image.resize_bilinear(deconv_1a, [128, 128])
          deconv_2a = slim.conv2d(tf.concat([upsamp_2a, conv_2a], axis=3), 
                        128, [5, 5], stride=1, scope='deconv2a')
          # upsample to input dimensions
          upsamp_3a = tf.image.resize_bilinear(deconv_2a, [256, 256])

          deconv_256 = slim.conv2d(tf.concat([upsamp_3a, conv_1a], axis=3), 
                        32, [5, 5], stride=1, scope='deconv256')
          # concatenate w/ coarser scale
          deconv_64a = tf.image.resize_bilinear(flow_64[:, :, :, :2], [256, 256])
          deconv_64a = slim.conv2d(deconv_64a, 32, [5, 5], stride=1, scope='deconv_64a')
          deconv_128 = tf.image.resize_bilinear(flow_128[:, :, :, :2], [256, 256])
          deconv_128 = slim.conv2d(deconv_128, 32, [5, 5], stride=1, scope='deconv_128')
          conv_concat1 = slim.conv2d(tf.concat([deconv_64a, deconv_128, deconv_256], axis=3),
                                96, [5, 5], stride=1, scope='conv_concat1')
          conv_concat2 = slim.conv2d(conv_concat1, 64, [5, 5], stride=1, scope='conv_concat2')

    net = slim.conv2d(conv_concat2, 3, [5, 5], stride=1, activation_fn=tf.tanh,
                      normalizer_fn=None, scope='flow')

    net, flow_motion, flow_mask = self._synthesize_frame(net, input_images)
    net128, __, __ = self._synthesize_frame(flow_128, input_images_128)
    net64, __, __ = self._synthesize_frame(flow_64, input_images_64)

    return(net, flow_motion, flow_mask, net128, net64)

  def _synthesize_frame(self, net, input_images):
    net_copy = net
    
    flow = net[:, :, :, 0:2]
    mask = tf.expand_dims(net[:, :, :, 2], 3)

    grid_x, grid_y = meshgrid(net.shape[1], net.shape[2])
    grid_x = tf.tile(grid_x, [FLAGS.batch_size, 1, 1])
    grid_y = tf.tile(grid_y, [FLAGS.batch_size, 1, 1])

    flow = 0.5 * flow

    coor_x_1 = grid_x + flow[:, :, :, 0]
    coor_y_1 = grid_y + flow[:, :, :, 1]

    coor_x_2 = grid_x - flow[:, :, :, 0]
    coor_y_2 = grid_y - flow[:, :, :, 1]    
    
    output_1 = bilinear_interp(input_images[:, :, :, 0:3], coor_x_1, coor_y_1, 'interpolate')
    output_2 = bilinear_interp(input_images[:, :, :, 3:6], coor_x_2, coor_y_2, 'interpolate')

    mask = 0.5 * (1.0 + mask)
    mask = tf.tile(mask, [1, 1, 1, 3])
    net = tf.multiply(mask, output_1) + tf.multiply(1.0 - mask, output_2)

    # for the correct loss function
    flow_motion = net_copy[:, :, :, 0:2]
    flow_mask = tf.expand_dims(net_copy[:, :, :, 2], 3)

    return(net, flow_motion, flow_mask)