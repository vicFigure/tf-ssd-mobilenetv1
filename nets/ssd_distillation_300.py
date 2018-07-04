import math
from collections import namedtuple

import numpy as np
import tensorflow as tf

import tf_extended as tfe
from nets import custom_layers
from nets import ssd_common
from nets import ssd_mobilenetv1_300
from nets import ssd_vgg_300
from nets import np_methods

import mobilenet_v1

slim = tf.contrib.slim

# =========================================================================== #
# SSD class definition.
# =========================================================================== #
SSDParams = namedtuple('SSDParameters', ['img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'
                                         ])

class SSDNet(object):
    """Implementation of the SSD MoibleNetV1-based 300 network.

    The default features layers with 300x300 image input are:
      conv4 ==> 38 x 38
      conv7 ==> 19 x 19
      conv8 ==> 10 x 10
      conv9 ==> 5 x 5
      conv10 ==> 3 x 3
      conv11 ==> 1 x 1
    The default image size used to train this network is 300x300.
    """
    default_params = SSDParams(
        img_shape=(300, 300),
        num_classes=21,
        no_annotation_label=21,
        feat_layers=['block11', 'block13', 'block14', 'block15', 'block16', 'block17'],
        feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
        anchor_size_bounds=[0.15, 0.90],
        # anchor_size_bounds=[0.20, 0.90],
        anchor_sizes=[(21., 45.),
                      (45., 99.),
                      (99., 153.),
                      (153., 207.),
                      (207., 261.),
                      (261., 315.)],
        # anchor_sizes=[(30., 60.),
        #               (60., 111.),
        #               (111., 162.),
        #               (162., 213.),
        #               (213., 264.),
        #               (264., 315.)],
        anchor_ratios=[[2, .5],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5],
                       [2, .5]],
        anchor_steps=[8, 16, 32, 64, 100, 300],
        anchor_offset=0.5,
        normalizations=[-1, -1, -1, -1, -1, -1],
        prior_scaling=[0.1, 0.1, 0.2, 0.2]
        )

    def __init__(self, params=None):
        """Init the SSD net with some parameters. Use the default ones
        if none provided.
        """
        if isinstance(params, SSDParams):
            self.params = params
        else:
            self.params = SSDNet.default_params
        self.student_model = ssd_mobilenetv1_300.SSDNet(self.params)
        self.teacher_model = ssd_vgg_300.SSDNet()
        self.teacher_kd_layer = {}
        self.student_kd_layer = {}

    # ======================================================================= #
    def net(self, inputs,
            is_training=True,
            update_feat_shapes=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='ssd_300_mobilenetv1'):
        """SSD network definition.
        """
        self.is_training = is_training
        student_r = self.student_model.net(inputs,is_training=is_training, scope=scope)
        self.student_kd_layer['logits'] = student_r[2]
        self.student_kd_layer['predictions'] = student_r[0]
        self.student_kd_layer['localisations'] = student_r[1]
        self.student_endpoints = student_r[-1]
        # Update feature shapes (try at least!)
        if update_feat_shapes:
            shapes = ssd_feat_shapes_from_net(student_r[0], self.params.feat_shapes)
            self.params = self.params._replace(feat_shapes=shapes)
        print("feat_shapes",self.params.feat_shapes)
        if is_training:
            teacher_r = self.teacher_model.net(inputs, is_training=False, scope='Teacher')
            self.teacher_kd_layer['logits'] = teacher_r[2]
            self.teacher_kd_layer['predictions'] = teacher_r[0]
            self.teacher_kd_layer['localisations'] = teacher_r[1]
            self.teacher_endpoints = teacher_r[-1]
            for item in self.teacher_kd_layer['logits']:
                tf.stop_gradient(item)
            for item in self.teacher_kd_layer['predictions']:
                tf.stop_gradient(item)
            for item in self.teacher_kd_layer['localisations']:
                tf.stop_gradient(item)
        return student_r

    def arg_scope(self, weight_decay=0.0005, data_format='NHWC'):
        """Network arg_scope.
        """
        return ssd_arg_scope(weight_decay, data_format=data_format)

    def arg_scope_caffe(self, caffe_scope):
        """Caffe arg_scope used for weights importing.
        """
        return ssd_arg_scope_caffe(caffe_scope)

    # ======================================================================= #
    def update_feature_shapes(self, predictions):
        """Update feature shapes from predictions collection (Tensor or Numpy
        array).
        """
        shapes = ssd_feat_shapes_from_net(predictions, self.params.feat_shapes)
        self.params = self.params._replace(feat_shapes=shapes)

    def anchors(self, img_shape, dtype=np.float32):
        """Compute the default anchor boxes, given an image shape.
        """
        return ssd_anchors_all_layers(img_shape,
                                      self.params.feat_shapes,
                                      self.params.anchor_sizes,
                                      self.params.anchor_ratios,
                                      self.params.anchor_steps,
                                      self.params.anchor_offset,
                                      dtype)

    def bboxes_encode(self, labels, bboxes, anchors,
                      scope=None):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_encode(
            labels, bboxes, anchors,
            self.params.num_classes,
            self.params.no_annotation_label,
            ignore_threshold=0.5,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def bboxes_decode(self, feat_localizations, anchors,
                      scope='ssd_bboxes_decode'):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_decode(
            feat_localizations, anchors,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def detected_bboxes(self, predictions, localisations,
                        select_threshold=None, nms_threshold=0.5,
                        clipping_bbox=None, top_k=400, keep_top_k=200):
        """Get the detected bounding boxes from the SSD network output.
        """
        # Select top_k bboxes from predictions, and clip
        rscores, rbboxes = \
            ssd_common.tf_ssd_bboxes_select(predictions, localisations,
                                            select_threshold=select_threshold,
                                            num_classes=self.params.num_classes)
        rscores, rbboxes = \
            tfe.bboxes_sort(rscores, rbboxes, top_k=top_k)
        # Apply NMS algorithm.
        rscores, rbboxes = \
            tfe.bboxes_nms_batch(rscores, rbboxes,
                                 nms_threshold=nms_threshold,
                                 keep_top_k=keep_top_k)
        if clipping_bbox is not None:
            rbboxes = tfe.bboxes_clip(clipping_bbox, rbboxes)
        return rscores, rbboxes



    def ssd_KD_loss(self, kd_lambda=1, scope='KD_loss'):
      # threshold for the select bbox
      threshold = 0.5
      with tf.variable_scope(scope, 'KD_loss'):
        # Get the teacher's prediction, and flatten them
        """
        we neet do the process separable to samples, 
        for example, we should  do nms in the output of one image, so we can't concat all the default box -- we should reshape the output to [batchsize, #of deault_box, 4/21],
        """
        t_logits = self.teacher_kd_layer['logits']
        t_localisations = self.teacher_kd_layer['localisations']
        t_predictions = self.teacher_kd_layer['predictions']
        batch_size = t_logits[0].shape[0]
        t_flogits = []
        t_fpredictions = []
        t_flocalisations = []
        for i in range(len(t_logits)):
            t_flogits.append(tf.reshape(t_logits[i], [batch_size, -1, self.params.num_classes]))
            t_flocalisations.append(tf.reshape(t_localisations[i], [batch_size, -1, 4]))
            t_fpredictions.append(tf.reshape(t_predictions[i], [batch_size, -1, self.params.num_classes]))

		# Now select the prior box for guidance
        mask = []
        self.anchors = self.anchors(self.params.img_shape)
        for i in range(len(t_logits)):
            mask_layer = select_layer_mask(t_fpredictions[i], t_localisations[i], self.anchors[i], threshold)
            mask.append(mask_layer)

		# Now we can define the KD-loss
        dtype = t_logits[0].dtype
        t_predictions = tf.concat(t_fpredictions, axis=1)
        t_localisations = tf.concat(t_flocalisations, axis=1)
        s_predictions = self.student_kd_layer['predictions']
        s_localisations = self.student_kd_layer['localisations']
        s_logits = self.student_kd_layer['logits']
        s_flocalisations = []
        s_fpredictions = []
        for i in range(len(s_logits)):
            s_flocalisations.append(tf.reshape(s_localisations[i], [batch_size, -1, 4]))
            s_fpredictions.append(tf.reshape(s_predictions[i], [batch_size, -1, self.params.num_classes]))
        s_predictions = tf.concat(s_fpredictions, axis=1)
        s_localisations = tf.concat(s_flocalisations, axis=1)
        mask = tf.concat(mask, axis=1)
        fmask = tf.cast(mask, dtype)

        N = tf.reduce_sum(tf.cast(mask, dtype))
        weights = kd_lambda * fmask
        with tf.name_scope('KD_mean_square_error_predictions'):
            loss = tf.reduce_mean(tf.square(t_predictions - s_predictions))
            loss = tf.div(tf.reduce_sum(loss * weights), N, name='value')
            tf.losses.add_loss(loss)
	    with tf.name_scope('KD_mean_square_error_localisations'):
			loss = tf.reduce_mean(tf.square(t_localisations - s_localisations))
            loss = tf.div(tf.reduce_sum(loss * weights), N, name='value')
            tf.losses.add_loss(loss)

    def losses(self, logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               scope='ssd_losses'):
        """Define the SSD network losses.
        """
        """
        ori_loss = self.student_model.losses(logits, localisations,
                          gclasses, glocalisations, gscores,
                          match_threshold=match_threshold,
                          negative_ratio=negative_ratio,
                          alpha=alpha,
                          label_smoothing=label_smoothing,
                          scope=scope)
        """
        if self.is_training:
          KD_loss = self.ssd_KD_loss(scope='KD_loss')


# =========================================================================== #
# SSD tools...
# =========================================================================== #
def select_layer_mask(predictions_layer, localisation_layer, anchor_layer, threshold=0.5):
    """
	1. we delete the background bbox: predict is class 0
	2. we delete the bbox whose prediction < threshold
	3. we do nms delete useless bbox. NOTICE: we do nms in one image, that is we should deal separated in batch_size dimension
	Input:
	    predictions_layer: flattened [batchsize, #default_box, 21]
   	    localisation_layer: not flattened, 5 dimensions
	    bbox: reshape to 3 dimensions 
	"""
    classes = tf.argmax(predictions_layer, axis=2)
    scores = tf.reduce_max(predictions_layer, axis=2)
    # step 1
    scores = scores * tf.cast(classes>0, scores.dtype)
    # step 2
    mask = tf.greater(scores, threshold)
    # step 3
    bbox = ssd_common.tf_ssd_bboxes_decode_layer(localisation_layer, anchor_layer)
    bbox = tf.reshape(bbox, [bbox.shape[0], -1, bbox.shape[-1]])
    nms_mask = tf.py_func(feature_nms, [predictions_layer, classes, bbox, 0.45], tf.bool)
    mask = tf.logical_and(mask, nms_mask)
    return mask

	
def feature_nms(predictions, classes, bbox, nms_threshold=0.45):
    batch_size = predictions.shape[0]
    num_defaultbox = predictions.shape[1]
    scores = np.amax(predictions, axis=2)
    mask = np.ones([batch_size, num_defaultbox], dtype=np.bool)
    
    #deal with one image
    for n in range(batch_size-1):
        score_batch = scores[n,:]
        index = np.argsort(-score_batch)
        bbox_batch = bbox[n,index,:]
        class_batch = classes[n,index]
        # do the nms
        for i in range(num_defaultbox-1):
            keep_bboxes = np.ones([num_defaultbox], dtype=np.bool)
            if keep_bboxes[i]:
                # Computer overlap with bboxes which are following.
                overlap = np_methods.bboxes_jaccard(bbox_batch[i], bbox_batch[(i+1):])
                # Overlap threshold for keeping + checking part of the same class
                keep_overlap = np.logical_or(overlap < nms_threshold, class_batch[(i+1):] != class_batch[i])
                keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)
        # update the mask
        mask[n,:] = np.logical_and(mask[n,:], keep_bboxes[index])	
	return mask
	
def ssd_size_bounds_to_values(size_bounds,
                              n_feat_layers,
                              img_shape=(300, 300)):
    """Compute the reference sizes of the anchor boxes from relative bounds.
    The absolute values are measured in pixels, based on the network
    default size (300 pixels).

    This function follows the computation performed in the original
    implementation of SSD in Caffe.

    Return:
      list of list containing the absolute sizes at each scale. For each scale,
      the ratios only apply to the first value.
    """
    assert img_shape[0] == img_shape[1]

    img_size = img_shape[0]
    min_ratio = int(size_bounds[0] * 100)
    max_ratio = int(size_bounds[1] * 100)
    step = int(math.floor((max_ratio - min_ratio) / (n_feat_layers - 2)))
    # Start with the following smallest sizes.
    sizes = [[img_size * size_bounds[0] / 2, img_size * size_bounds[0]]]
    for ratio in range(min_ratio, max_ratio + 1, step):
        sizes.append((img_size * ratio / 100.,
                      img_size * (ratio + step) / 100.))
    return sizes


def ssd_feat_shapes_from_net(predictions, default_shapes=None):
    """Try to obtain the feature shapes from the prediction layers. The latter
    can be either a Tensor or Numpy ndarray.

    Return:
      list of feature shapes. Default values if predictions shape not fully
      determined.
    """
    feat_shapes = []
    for l in predictions:
        # Get the shape, from either a np array or a tensor.
        if isinstance(l, np.ndarray):
            shape = l.shape
        else:
            shape = l.get_shape().as_list()
        shape = shape[1:4]
        # Problem: undetermined shape...
        if None in shape:
            return default_shapes
        else:
            feat_shapes.append(shape)
    return feat_shapes


def ssd_anchor_one_layer(img_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         step,
                         offset=0.5,
                         dtype=np.float32):
    """Computer SSD default anchor boxes for one feature layer.

    Determine the relative position grid of the centers, and the relative
    width and height.

    Arguments:
      feat_shape: Feature shape, used for computing relative position grids;
      size: Absolute reference sizes;
      ratios: Ratios to use on these features;
      img_shape: Image shape, used for computing height, width relatively to the
        former;
      offset: Grid offset.

    Return:
      y, x, h, w: Relative x and y grids, and height and width.
    """
    # Compute the position grid: simple way.
    # y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    # y = (y.astype(dtype) + offset) / feat_shape[0]
    # x = (x.astype(dtype) + offset) / feat_shape[1]
    # Weird SSD-Caffe computation using steps values...
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = len(sizes) + len(ratios)
    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)
    # Add first anchor boxes with ratio=1.
    h[0] = sizes[0] / img_shape[0]
    w[0] = sizes[0] / img_shape[1]
    di = 1
    if len(sizes) > 1:
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        di += 1
    for i, r in enumerate(ratios):
        h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
    return y, x, h, w


def ssd_anchors_all_layers(img_shape,
                           layers_shape,
                           anchor_sizes,
                           anchor_ratios,
                           anchor_steps,
                           offset=0.5,
                           dtype=np.float32):
    """Compute anchor boxes for all feature layers.
    """
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = ssd_anchor_one_layer(img_shape, s,
                                             anchor_sizes[i],
                                             anchor_ratios[i],
                                             anchor_steps[i],
                                             offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors


# =========================================================================== #
# Functional definition of MobileNetV1-based SSD 300.
# =========================================================================== #
def tensor_shape(x, rank=3):
    """Returns the dimensions of a tensor.
    Args:
      image: A N-D Tensor of shape.
    Returns:
      A list of dimensions. Dimensions that are statically known are python
        integers,otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]


def ssd_multibox_layer(inputs,
                       num_classes,
                       sizes,
                       ratios=[1],
                       normalization=-1,
                       bn_normalization=False):
    """Construct a multibox layer, return a class and localization predictions.
    """
    net = inputs
    if normalization > 0:
        net = custom_layers.l2_normalization(net, scaling=True)
    # Number of anchors.
    num_anchors = len(sizes) + len(ratios)

    # Location.
    num_loc_pred = num_anchors * 4
    loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], activation_fn=None,
                           scope='conv_loc')
    loc_pred = custom_layers.channel_to_last(loc_pred)
    loc_pred = tf.reshape(loc_pred,
                          tensor_shape(loc_pred, 4)[:-1]+[num_anchors, 4])
    # Class prediction.
    num_cls_pred = num_anchors * num_classes
    cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], activation_fn=None,
                           scope='conv_cls')
    cls_pred = custom_layers.channel_to_last(cls_pred)
    cls_pred = tf.reshape(cls_pred,
                          tensor_shape(cls_pred, 4)[:-1]+[num_anchors, num_classes])
    return cls_pred, loc_pred


def ssd_arg_scope(weight_decay=0.0005, data_format='NHWC'):
    """Defines the MobileNetV1 arg scope.

    Args:
      weight_decay: The l2 regularization coefficient.

    Returns:
      An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            data_format=data_format):
            with slim.arg_scope([custom_layers.pad2d,
                                 custom_layers.l2_normalization,
                                 custom_layers.channel_to_last],
                                data_format=data_format) as sc:
                return sc


# =========================================================================== #
# Caffe scope: importing weights at initialization.
# =========================================================================== #
def ssd_arg_scope_caffe(caffe_scope):
    """Caffe scope definition.

    Args:
      caffe_scope: Caffe scope object with loaded weights.

    Returns:
      An arg_scope.
    """
    # Default network arg scope.
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=caffe_scope.conv_weights_init(),
                        biases_initializer=caffe_scope.conv_biases_init()):
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu):
            with slim.arg_scope([custom_layers.l2_normalization],
                                scale_initializer=caffe_scope.l2_norm_scale_init()):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    padding='SAME') as sc:
                    return sc


def ssd_net(inputs,
            num_classes=SSDNet.default_params.num_classes,
            feat_layers=SSDNet.default_params.feat_layers,
            anchor_sizes=SSDNet.default_params.anchor_sizes,
            anchor_ratios=SSDNet.default_params.anchor_ratios,
            normalizations=SSDNet.default_params.normalizations,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='ssd_300_mobilenetv1'):
    return SSDNet.net(inputs)
