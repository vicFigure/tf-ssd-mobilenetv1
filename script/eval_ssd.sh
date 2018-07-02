EVAL_DIR=logs/
DATASET_DIR=tfrecords/voc_2007/
CHECKPOINT_PATH=checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt

CUDA_VISIBLE_DEVICES='' python eval_ssd_network.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=train \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --batch_size=1 \
    --max_num_batches=50
