import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time
import os

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import graph_util
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.ops import state_ops


teacher_checkpoint = 'logs/vgg/model.ckpt-5000'
kd_checkpoint = 'logs/kd_mobilenet/model.ckpt-0'


def no_use_var(name):
  if name.split('/')[-1] == 'Momentum':
    return True
  if 'lr' in name:
    return True
  if 'step' in name:
    return True
  if 'RMSProp' in name:
      return True
  if 'ExponentialMovingAverage' in name:
      return True
  if 'Adam' in name:
    return True
  return False


def look_up_var(ckpt_path):
  var_value = {}
  reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
  var_to_shape_map = reader.get_variable_to_shape_map()
  for key in var_to_shape_map:
    try:
      tensor = reader.get_tensor(key)
    except KeyError:
      # This tensor doesn't exist in the graph (for example it's
      # 'global_step' or a similar housekeeping element) so skip it.
      continue
    #var tensor is the type of np.ndarray
    var_value[key] = tensor

  pop_key = []
  for key in var_value:
    if no_use_var(key):
      pop_key.append(key)
  for key in pop_key:
    var_value.pop(key)

  return var_value


if __name__ == '__main__':
  vars_teacher = look_up_var(teacher_checkpoint)
  vars_kd = look_up_var(kd_checkpoint)

  num = 0
  for key in vars_teacher:
    num += 1
    key_split = key.split('/')
    if 'ssd' in key_split[0]:
      key_split[0] = 'Teacher'
      kd_key = '/'.join(key_split)
    else:
      kd_key = key
    teacher_value = vars_teacher[key]
    kd_value = vars_kd[kd_key]
    if (teacher_value == kd_value).all():
      continue
    diff = teacher_value - kd_value
    print key, np.max(diff), np.min(diff)
  print num, 'vars have been checked'


  
    
    




