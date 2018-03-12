#!/usr/bin/env bash

source activate tensorflow

WORKSPACE_DIR=/home/rcf-40/jupadhya/staging/Workspace
TRAINING_DIR=${WORKSPACE_DIR}/training
PRETRAINED_MODELS=${WORKSPACE_DIR}/pre-trained-models/faster_rcnn_resnet101_coco_2018_01_28
RESEARCH_DIR=/home/rcf-40/jupadhya/staging/Repos/models/research

PATH_TO_YOUR_PIPELINE_CONFIG=${TRAINING_DIR}/models/model/faster_rcnn_resnet101_df.config
PATH_TO_TRAIN_DIR=${WORKSPACE_DIR}/training/models/model/train
PATH_TO_EVAL_DIR=${WORKSPACE_DIR}/training/models/model/eval

# From tensorflow/models/research/
cd ${RESEARCH_DIR}

python object_detection/data/deep-fashion/extract_df_bboxes.py

# ./create_tf_record.sh

cp object_detection/data/deep-fashion/df_label_map.pbtxt ${TRAINING_DIR}/data

# cp object_detection/data/deep-fashion/df_*.record ${TRAINING_DIR}/data

# cp ${PRETRAINED_MODELS}/model.ckpt.* ${TRAINING_DIR}/models/model/

# Edit the faster_rcnn_resnet101_pets.config template. Please note that there
# are multiple places where PATH_TO_BE_CONFIGURED needs to be set.
sed -e "s|PATH_TO_BE_CONFIGURED|"${TRAINING_DIR}"|g" \
    object_detection/samples/configs/faster_rcnn_resnet101_df.config > ${PATH_TO_YOUR_PIPELINE_CONFIG}
echo "Config file"

sed -e "s|PATH_TO_TRAIN_DIR|"${PATH_TO_TRAIN_DIR}"|g" \
    -e "s|RESEARCH_DIR|"${RESEARCH_DIR}"|g" \
    -e "s|PATH_TO_YOUR_PIPELINE_CONFIG|"${PATH_TO_YOUR_PIPELINE_CONFIG}"|g" \
    hpc_training_job.pbs > ${TRAINING_DIR}/training_job.pbs
echo "Training job"

sed -e "s|PATH_TO_TRAIN_DIR|"${PATH_TO_TRAIN_DIR}"|g" \
    -e "s|RESEARCH_DIR|"${RESEARCH_DIR}"|g" \
    -e "s|PATH_TO_YOUR_PIPELINE_CONFIG|"${PATH_TO_YOUR_PIPELINE_CONFIG}"|g" \
    -e "s|PATH_TO_EVAL_DIR|"${PATH_TO_EVAL_DIR}"|g" \
    hpc_evaluation_job.pbs > ${TRAINING_DIR}/evaluation_job.pbs
echo "Eval job"

sed -e "s|PATH_TO_TRAIN_DIR|"${PATH_TO_TRAIN_DIR}"|g" \
    -e "s|RESEARCH_DIR|"${RESEARCH_DIR}"|g" \
    -e "s|PATH_TO_YOUR_PIPELINE_CONFIG|"${PATH_TO_YOUR_PIPELINE_CONFIG}"|g" \
    hpc_export_job.pbs > ${TRAINING_DIR}/export_job.pbs
echo "Export job"

tree ${TRAINING_DIR}
