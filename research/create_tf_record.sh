#!/usr/bin/env bash

WORKSPACE_DIR=/home/rcf-40/jupadhya/staging/Workspace
RESEARCH_DIR=/home/rcf-40/jupadhya/staging/Repos/models/research
DEEP_FASHION_DIR=/home/rcf-40/jupadhya/staging/Workspace/deep-fashion
ANNO_DIR=${DEEP_FASHION_DIR}/anno
TRAINING_DIR=${WORKSPACE_DIR}/training/overfit

cd ${RESEARCH_DIR}

python object_detection/dataset_tools/create_df_tf_record.py \
--logtostderr \
--data_dir="${DEEP_FASHION_DIR}" \
--train_annotations_file="${ANNO_DIR}/df_train.npy" \
--val_annotations_file="${ANNO_DIR}/df_val.npy" \
--testdev_annotations_file="${ANNO_DIR}/df_test.npy" \
--labels_file="${ANNO_DIR}/df_labels.npy" \
--output_dir="${TRAINING_DIR}/data"
