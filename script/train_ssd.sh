#!/bin/bash

set -e

GPU_ID=$1
DATASET=$2
NET=$3
# DATASET should be pascalvoc_2007 / pascalvoc_2012 / pascalvoc_0712
# NET should vgg / mobilenetv1 / teacher-student

case ${DATASET} in
  pascalvoc_2007)
    TRAIN_DATASET="pascalvoc_2007"
    TEST_DATASET="pascalvoc_2007"
    DATASET_DIR=tfrecords/voc
	;;
  pascalvoc_2012)
    TRAIN_DATASET="pascalvoc_2012"
    TEST_DATASET="pascalvoc_2007"
    DATASET_DIR=tfrecords/voc
	;;
  pascalvoc_0712)
    TRAIN_DATASET="pascalvoc_0712"
    TEST_DATASET="pascalvoc_2007"
    DATASET_DIR=tfrecords/voc
	;;
  *)
    echo "No dataset given"
	exit
	;;
esac

TRAIN_DIR=logs/${NET}
CHECKPOINT_PATH=checkpoints/ssd_300_${NET}/ssd_300_${NET}.ckpt
CHECKPOINT_PATH=checkpoints/${NET}/${NET}.ckpt
MODEL_NAME=ssd_300_${NET}

#    --checkpoint_exclude_scopes=ssd_300_mobilenetv1/block11_box/conv_cls,ssd_300_mobilenetv1/block13_box/conv_cls,ssd_300_mobilenetv1/block14_box,ssd_300_mobilenetv1/block15_box/conv_cls,ssd_300_mobilenetv1/block16_box/conv_cls,ssd_300_mobilenetv1/block17_box/conv_cls \
#    --trainable_scopes=ssd_300_mobilenetv1/block11_box/conv_cls,ssd_300_mobilenetv1/block13_box/conv_cls,ssd_300_mobilenetv1/block14_box/conv_cls,ssd_300_mobilenetv1/block15_box/conv_cls,ssd_300_mobilenetv1/block16_box/conv_cls,ssd_300_mobilenetv1/block17_box/conv_cls \
CUDA_VISIBLE_DEVICES=${GPU_ID} python train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=${TRAIN_DATASET} \
    --dataset_split_name=train \
    --model_name=${MODEL_NAME} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=ssd_300_mobilenetv1/block11_box,ssd_300_mobilenetv1/block13_box,ssd_300_mobilenetv1/block14,ssd_300_mobilenetv1/block15,ssd_300_mobilenetv1/block16,ssd_300_mobilenetv1/block17,ssd_300_mobilenetv1/block14_box,ssd_300_mobilenetv1/block15_box,ssd_300_mobilenetv1/block16_box,ssd_300_mobilenetv1/block17_box\
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.001 \
    --learning_rate_decay_type=polynomial \
    --end_learning_rate=0.0001 \
    --max_number_of_steps=300000 \
    --batch_size=16

EVAL_DIR=${TRAIN_DIR}/eval
CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_ssd_network.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=${TEST_DATASET} \
    --dataset_split_name=test \
    --model_name=${MODEL_NAME} \
    --checkpoint_path=${TRAIN_DIR} \
    --wait_for_checkpoints=False \
    --batch_size=1 \
    --max_num_batches=4952
