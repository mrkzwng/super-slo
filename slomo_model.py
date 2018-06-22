from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils.geo_layer_utils import vae_gaussian_layer
from utils.geo_layer_utils import bilinear_interp
from utils.geo_layer_utils import meshgrid


FLAGS = tf.app.flags.FLAGS

class SloMo_model(object):
  def __init__(self, for_interpolation, is_train=True):
    """
    for_interpolation {bool}
    is_train {bool}
    """
    self.is_train = is_train
    self.batch_norm_params = {'decay': 0.9997,
                              'epsilon': 0.001,
                              'is_training': self.is_train}
    self.for_interpolation = for_interpolation
    self.mode = 'interpolater_' if for_interpolation else 'computer_'


  def inference(self, input_images):
    """
    model's output
    """
    return self._build_model(input_images) 


  def _encode_layer(self, 
                    scope, inputs, out_channels, 
                    kernel_size, pool=True, stride=1):
    """
    an encoding layer
    """
    if pool:
      avg_pool = slim.avg_pool2d(inputs, [2, 2], 
                                 scope=scope + 'pool')
      conv = slim.conv2d(avg_pool, 
                         out_channels, 
                         kernel_size, 
                         stride, 
                         scope=scope + 'conv1')
      conv = slim.conv2d(conv,
                         out_channels,
                         kernel_size,
                         stride,
                         scope=scope + 'conv2')
    else:
      conv = slim.conv2d(inputs, 
                         out_channels, 
                         kernel_size, 
                         stride, 
                         scope=scope + 'conv1')
      conv = slim.conv2d(conv,
                         out_channels,
                         kernel_size,
                         stride,
                         scope=scope + 'conv2')

    return(conv)


  def _decode_layer(self,
                    scope, inputs, skip_inputs,
                    out_channels, kernel_size, stride=1):
    """
    a decoding layer
    """
    upsamp = tf.image.resize_bilinear(inputs, 
                  [inputs.shape[1] * 2, inputs.shape[2] * 2])
    cat_inputs = tf.concat([upsamp, skip_inputs], axis=3)
    conv = slim.conv2d(cat_inputs,
                     out_channels,
                     kernel_size,
                     stride,
                     scope=scope + 'conv1')
    conv = slim.conv2d(conv,
                     out_channels,
                     kernel_size,
                     stride,
                     scope=scope + 'conv2')

    return(conv)


  def _build_model(self, input_images):
    """
    graph definition
    """
    with slim.arg_scope(
      [slim.conv2d],
      activation_fn=tf.nn.leaky_relu,
      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
      reuse=tf.AUTO_REUSE):

      with slim.arg_scope([slim.batch_norm], 
                          is_training = self.is_train, 
                          updates_collections=None):

        with slim.arg_scope([slim.conv2d], 
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=self.batch_norm_params):
          # encode
          layers = {}
          out_channels = 32
          for layer_idx in range(6):
            if layer_idx == 0:
              kernel_size = [7, 7]
              layers['layer_' + str(layer_idx)] = \
                   self._encode_layer(scope=self.mode + 'encoder_' + str(layer_idx) + '_',
                   inputs=input_images,
                   out_channels=out_channels,
                   kernel_size=kernel_size,
                   pool=False)
            else:
              kernel_size = [5, 5] if layer_idx == 1 else [3, 3]
              layers['layer_'+str(layer_idx)] = \
                   self._encode_layer(scope=self.mode + 'encoder_' + str(layer_idx) + '_',
                   inputs=layers['layer_' + str(layer_idx - 1)],
                   out_channels=out_channels,
                   kernel_size=kernel_size)
            out_channels = out_channels * 2 if out_channels < 512 else 512

          layer = layers['layer_5']

          # decode
          for layer_idx in range(5):
            layer = self._decode_layer(scope=self.mode + 'decoder_' + str(layer_idx) + '_',
                       inputs=layer, 
                       skip_inputs=layers['layer_'+str(4 - layer_idx)],
                       out_channels=out_channels,
                       kernel_size=kernel_size)
            out_channels = out_channels / 2 

          if self.for_interpolation:
            layer = slim.conv2d(layer, 8, [3, 3], 
                                stride=1, scope='interpolation_output')
            flow_t0 = layer[:, :, :, :3]
            flow_t1 = layer[:, :, :, 3:6]
            vis_mask_0 = tf.expand_dims(layer[:, :, :, 6], axis=3)
            vis_mask_1 = tf.expand_dims(layer[:, :, :, 7], axis=3)
            outputs = (flow_t0, flow_t1, vis_mask_0, vis_mask_1)
          else:
            layer = slim.conv2d(layer, 6, [3, 3], 
                                stride=1, scope='computation_output')
            flow_01 = layer[:, :, :, :3]
            flow_10 = layer[:, :, :, 3:]
            outputs = (flow_01, flow_10)

    return(outputs)


  def warp(self, net, input_images):
    net_copy = net
    
    flow = net[:, :, :, 0:2]
    mask = tf.expand_dims(net[:, :, :, 2], 3)

    grid_x, grid_y = meshgrid(net.shape[1], net.shape[2])
    grid_x = tf.tile(grid_x, [FLAGS.batch_size, 1, 1])
    grid_y = tf.tile(grid_y, [FLAGS.batch_size, 1, 1])

    # flow = 0.5 * flow

    coor_x_1 = grid_x + flow[:, :, :, 0]
    coor_y_1 = grid_y + flow[:, :, :, 1]

    # coor_x_2 = grid_x - flow[:, :, :, 0]
    # coor_y_2 = grid_y - flow[:, :, :, 1]    
    
    output_1 = bilinear_interp(input_images[:, :, :, 0:3], coor_x_1, coor_y_1, 'interpolate')
    # output_2 = bilinear_interp(input_images[:, :, :, 3:6], coor_x_2, coor_y_2, 'interpolate')

    mask = (1.0 + mask)
    mask = tf.tile(mask, [1, 1, 1, 3])
    net = tf.multiply(mask, output_1) \
            # + tf.multiply(1.0 - mask, output_2)

    # for the correct loss function
    flow_motion = net_copy[:, :, :, 0:2]
    flow_mask = tf.expand_dims(net_copy[:, :, :, 2], 3)
    
    return(net)