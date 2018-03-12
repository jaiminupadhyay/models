#!/usr/bin/env bash

WORKSPACE_DIR=/home/rcf-40/jupadhya/staging/Workspace
TRAINING_DIR=${WORKSPACE_DIR}/training
PRETRAINED_MODELS=${WORKSPACE_DIR}/pre-trained-models/faster_rcnn_resnet101_coco_2018_01_28
RESEARCH_DIR=/home/rcf-40/jupadhya/staging/Repos/models/research

PATH_TO_YOUR_PIPELINE_CONFIG=${TRAINING_DIR}/models/faster_rcnn_resnet101_df.config
PATH_TO_TRAIN_DIR=${WORKSPACE_DIR}/training/models/train
PATH_TO_EVAL_DIR=${WORKSPACE_DIR}/training/models/eval

CHECKPOINT_NUMBER=000000

# From tensorflow/models/research/
cd ${RESEARCH_DIR}

# Edit the faster_rcnn_resnet101_pets.config template. Please note that there
# are multiple places where PATH_TO_BE_CONFIGURED needs to be set.
sed -i "s|PATH_TO_BE_CONFIGURED|"${TRAINING_DIR}"/data|g" \
    object_detection/samples/configs/faster_rcnn_resnet101_df.config

# Copy edited template to cloud.
cp object_detection/samples/configs/faster_rcnn_resnet101_df.config \
    ${PATH_TO_YOUR_PIPELINE_CONFIG}

cp ${RESEARCH_DIR}/object_detection/data/deep-fashion/df_label_map.pbtxt ${TRAINING_DIR}/data

cp ${RESEARCH_DIR}/object_detection/data/deep-fashion/df_*.record ${TRAINING_DIR}/data

cp ${PRETRAINED_MODELS}/model.ckpt.* ${TRAINING_DIR}/models/model/

# From the tensorflow/models/research/ directory
python object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --train_dir=${PATH_TO_TRAIN_DIR}

# From the tensorflow/models/research/ directory
python object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --checkpoint_dir=${PATH_TO_TRAIN_DIR} \
    --eval_dir=${PATH_TO_EVAL_DIR}

# From tensorflow/models/research/
gsutil cp ${PATH_TO_TRAIN_DIR}/model.ckpt-${CHECKPOINT_NUMBER}.* .
python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --trained_checkpoint_prefix model.ckpt-${CHECKPOINT_NUMBER} \
    --output_directory exported_graphs