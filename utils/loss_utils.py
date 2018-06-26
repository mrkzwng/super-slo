"""Implements various tensorflow loss layer.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


def l1_charbonnier(values, epsilon):
  """
  implements the generalized Charbonnier distance
  """
  loss = tf.reduce_mean(tf.sqrt(tf.square(values) + tf.square(epsilon)))
  return(loss)

def l1_charbonnier_loss(predictions, targets, epsilon):
  """
  implements loss via generalized Charbonnier
  """
  return(l1_charbonnier(predictions - targets, epsilon))

def l1_loss(predictions, targets):
  """Implements tensorflow l1 loss.
  Args:
  Returns:
  """
  return(tf.reduce_mean(tf.abs(predictions - targets)))


def l1_regularizer(flow):
  '''
  implements L1 regularization for flow in R^[n x m]
  '''
  l1_magnitude = tf.reduce_mean(tf.abs(flow))

  return(l1_magnitude)


def l2_loss(predictions, targets):
  """Implements tensorflow l2 loss, normalized by number of elements.
  Args:
  Returns:
  """
  n_elems = tf.cast(tf.size(targets), tf.float32)

  return(tf.sqrt(tf.reduce_sum(tf.square(predictions - targets))) / n_elems)


def tv_loss():
  #TODO
  pass
  
def vae_loss(z_mean, z_logvar, prior_weight=1.0):
  """Implements the VAE reguarlization loss.
  """
  total_elements = (tf.shape(z_mean)[0] * tf.shape(z_mean)[1] * tf.shape(z_mean)[2]
      * tf.shape(z_mean)[3])
  total_elements = tf.to_float(total_elements)

  vae_loss = -0.5 * tf.reduce_sum(1.0 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar))
  vae_loss = tf.div(vae_loss, total_elements)
  return vae_loss

def bilateral_loss():
  #TODO
  pass
