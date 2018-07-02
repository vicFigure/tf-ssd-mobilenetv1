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

CHECKPOINT_PATH=checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt
CHECKPOINT_PATH=checkpoints/ssd_300_${NET}/ssd_300_${NET}.ckpt
CHECKPOINT_PATH=logs/${NET}/model.ckpt-42203
TRAIN_DIR=logs/${NET}
MODEL_NAME=ssd_300_${NET}

EVAL_DIR=${TRAIN_DIR}/eval
CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_ssd_network.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=${TEST_DATASET} \
    --dataset_split_name=test \
    --model_name=${MODEL_NAME} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --wait_for_checkpoints=False \
    --batch_size=1 \
    --max_num_batches=4952

