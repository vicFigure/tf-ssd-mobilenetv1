#!/bin/bash
set -e

DATASET_DIR=../datasets/pascal_voc/VOC2012/
OUT_DIR=tfrecords/voc

python tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=${DATASET_DIR} \
    --output_name=voc_2012_train \
    --output_dir=${OUT_DIR}
