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


input_checkpoint = 'ssd_mobilenet_v1_coco_11_06_2017/model.ckpt'
input_checkpoint = 'mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt'
save_checkpoint = 'ssd_mobilenet_v1_coco_11_06_2017/ssd_myname_model.ckpt'
save_checkpoint = 'mobilenet_v1_1.0_224/mobilenetv1.ckpt'


def no_use(name):
  if name.split('/')[-1] == 'Momentum':
    return True
  if 'RMSProp' in name:
    return True
  if 'ExponentialMovingAverage' in name:
    return True
  if 'lr' in name:
    return True
  if 'step' in name:
    return True
#  if 'Teacher' in name:
#    return True
  return False


def get_vars(ckpt):
    """give the ckpt model, and return a dict, with the params' name as keys and params' value as value
    Input:
        ckpt: the name of ckpt model
    Output:
        var_value: the dict with params' name as key; params' value as value
    """
    var_value = {}
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt)
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
      if no_use(key):
        pop_key.append(key)
    for key in pop_key:
      var_value.pop(key)
    print "var_value DONE!"
    return var_value

def change_name(ori_name):
    if 'ssd' in input_checkpoint:
      print "changing the ssd ckpt"
      return change_name_ssd(ori_name)
    else:
      print "changing the imagenet ckpt"
      return change_name_imagenet(ori_name)

def change_name_imagenet(ori_name):
    name = None
    name_split = ori_name.split('/')
    name_split[0] = 'MobilenetV1'
    name_split.insert(0, 'ssd_300_mobilenetv1')
    tmp_name = name_split[2].split('_')
    if len(tmp_name) > 1:
      block_num = tmp_name[1]
      name_split.insert(2, 'block%s'%block_num)
    
    name = '/'.join(name_split)
    return name
 
def change_name_ssd(ori_name):
    """From the original name, we return the proper name as our model
    If we don't want the param with ori_name, we return None
    in download mobilenet_ssd:
    name: FeatureExtractor/MobilenetV1/Conv2d_#i_depthwise(pointwise)
    VS. ssd_300_mobilenetv1/MobilenetV1/block#i/Conv2d_#i_depthwise(pointwise)
      
      FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1(2)_Conv2d_#j_1*1(3*3)_#channelnum
    VS. ssd_300_mobilenetv2/MobileNetV1/block#j+12/conv1*1(3*3)
    """
    name = None
    if 'MobilenetV1' in ori_name:
      name_split = ori_name.split('/')
      name_split[0] = 'ssd_300_mobilenetv1'
      tmp_name = name_split[2].split('_')
      if len(tmp_name) > 5:
        name_split[1] = 'block%d'%(int(tmp_name[5])+12)
        name_split[2] = 'conv%s'%tmp_name[6]
      else:
        num = tmp_name[1]
        name_split.insert(2, 'block%s'%num)
      name = '/'.join(name_split)
    if 'BoxPredictor' in ori_name:
      name_split = ori_name.split('/')
      block_num = int(name_split[0].split('_')[1]) + 12
      if block_num < 13: block_num = 11
      if name_split[1] == 'ClassPredictor':
        conv_name = 'conv_cls'
      else:
        conv_name = 'conv_loc'
      name_split[0] = 'ssd_300_mobilenetv1'
      name_split[1] = 'block%d_box'%block_num
      name_split.insert(2, conv_name)
      name = '/'.join(name_split)
    return name

def change_all_names(input_ckpt, save_ckpt):
    """from the ckpt, we change the name of each params in the way of function **change_name**
    Input:
        input_ckpt: the path of input ckpt model
        output_ckpt: the path of output ckpt model
    """
    # First get the var_values from input_ckpt
    var_value = get_vars(input_ckpt)

    # Second build a new graph
    graph = tf.Graph()
    with graph.as_default():
      for key in var_value:
        new_key = change_name(key)
        if new_key:
          var = tf.Variable(var_value[key], name=new_key, trainable=True)
          print new_key, var.shape

    # Third save the new graph
    with tf.Session(graph=graph) as sess:
      init_op = tf.global_variables_initializer()
      sess.run(init_op)
      saver = tf.train.Saver()
      saver.save(sess, save_ckpt)
      print "Save new graph DONE!"



if __name__ == '__main__':

  '''
  var_value = get_vars(input_checkpoint)
  for key in var_value:
    print key
  '''
  # Change all the name in input_checkpoint
  change_all_names(input_checkpoint, save_checkpoint)

  
    
    




