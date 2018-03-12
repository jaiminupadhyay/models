# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert Deep Fashion dataset to TFRecord for object_detection.

Example usage:
    python create_df_tf_record.py --logtostderr \
      --data_dir="${DEEP_FASHION_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
      --labels_file="${LABEL_FILE}" \
      --output_dir="${OUTPUT_DIR}"
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import json
import os
import numpy as np
import PIL.Image

import tensorflow as tf

from object_detection.utils import dataset_util

flags = tf.app.flags
tf.flags.DEFINE_string('data_dir', '',
                       'Data directory.')
tf.flags.DEFINE_string('train_annotations_file', '',
                       'Training annotations NPY file.')
tf.flags.DEFINE_string('val_annotations_file', '',
                       'Validation annotations NPY file.')
tf.flags.DEFINE_string('testdev_annotations_file', '',
                       'Test-dev annotations NPY file.')
tf.flags.DEFINE_string('labels_file', '',
                       'Labels NPY file.')
tf.flags.DEFINE_string('output_dir', '/tmp/', 'Output data directory.')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def create_tf_example(image,
                      annotations_list,
                      data_dir,
                      labels_index):
  """Converts image and annotations to a tf.Example proto.

  Args:
    image: dict with keys:
      [u'file_name', u'height', u'width', u'id']
    annotations_list:
      list of dicts with keys:
      [u'image_id', u'bbox', u'label_id', u'id']
      This function converts bounding box coordinates
      to the format expected by the Tensorflow Object Detection API (which is
      which is [ymin, xmin, ymax, xmax] with coordinates normalized relative
      to image size).
    data_dir: directory containing the image files.
    labels_index: a dict containing Deep Fashion label information keyed
      by the 'id' field of each label.
  Returns:
    example: The converted tf.Example
    num_annotations_skipped: Number of (invalid) annotations that were ignored.

  Raises:
    ValueError: if the image pointed to by filename is not a valid JPEG
  """
  filename = image[1]
  image_id = image[0]

  full_path = os.path.join(data_dir, filename)
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  image_width, image_height = image.size
  key = hashlib.sha256(encoded_jpg).hexdigest()

  xmin = []
  xmax = []
  ymin = []
  ymax = []
  label_names = []
  label_ids = []
  num_annotations_skipped = 0
  for object_annotations in annotations_list:
    (x1, y1, x2, y2) = tuple(map(int, object_annotations[:4]))
    if x2-x1 <= 0 or y2-y1 <= 0:
      num_annotations_skipped += 1
      continue
    if x2 > image_width or y2 > image_height:
      num_annotations_skipped += 1
      continue
    xmin.append(float(x1) / image_width)
    xmax.append(float(x2) / image_width)
    ymin.append(float(y1) / image_height)
    ymax.append(float(y2) / image_height)
    label_id = int(object_annotations[4])
    label_ids.append(label_id)
    label_names.append(labels_index[label_id].encode('utf8'))

  feature_dict = {
    'image/height':
      dataset_util.int64_feature(image_height),
    'image/width':
      dataset_util.int64_feature(image_width),
    'image/filename':
      dataset_util.bytes_feature(filename.encode('utf8')),
    'image/source_id':
      dataset_util.bytes_feature(str(image_id).encode('utf8')),
    'image/key/sha256':
      dataset_util.bytes_feature(key.encode('utf8')),
    'image/encoded':
      dataset_util.bytes_feature(encoded_jpg),
    'image/format':
      dataset_util.bytes_feature('jpeg'.encode('utf8')),
    'image/object/bbox/xmin':
      dataset_util.float_list_feature(xmin),
    'image/object/bbox/xmax':
      dataset_util.float_list_feature(xmax),
    'image/object/bbox/ymin':
      dataset_util.float_list_feature(ymin),
    'image/object/bbox/ymax':
      dataset_util.float_list_feature(ymax),
    'image/object/class/label':
      dataset_util.int64_list_feature(label_ids),
  }
  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return key, example, num_annotations_skipped


def _create_tf_record_from_df_annotations(annotations_file, labels_file, data_dir, output_path):
  """Loads Deep Fashion annotation numpy files and converts to tf.Record format.

  Args:
    annotations_file: NPY file containing bounding box annotations.
    data_dir: Directory containing the image files.
    output_path: Path to output tf.Record file.
  """
  groundtruth_data = np.load(annotations_file)
  labels = np.load(labels_file)
  labels_index = {int(label[0]): label[1] for label in labels}

  images = groundtruth_data[:, [0, 1]]

  annotations_index = {}
  tf.logging.info(
    'Found groundtruth annotations. Building annotations index.')
  for annotation in groundtruth_data[:, [0, 2, 3, 4, 5, 6]]:
    image_id = annotation[0]
    if image_id not in annotations_index:
      annotations_index[image_id] = []
    annotations_index[image_id].append(annotation[1:])
  missing_annotation_count = 0
  for image in images:
    image_id = image[0]
    if image_id not in annotations_index:
      missing_annotation_count += 1
      annotations_index[image_id] = []
  tf.logging.info('%d images are missing annotations.',
                  missing_annotation_count)

  tf.logging.info('writing to output path: %s', output_path)
  writer = tf.python_io.TFRecordWriter(output_path)
  total_num_annotations_skipped = 0
  for idx, image in enumerate(images):
    if idx % 100 == 0:
      tf.logging.info('On image %d of %d', idx, len(images))
    annotations_list = annotations_index[image[0]]
    _, tf_example, num_annotations_skipped = create_tf_example(
      image, annotations_list, data_dir, labels_index)
    total_num_annotations_skipped += num_annotations_skipped
    writer.write(tf_example.SerializeToString())
  writer.close()
  tf.logging.info('Finished writing, skipped %d annotations.',
                  total_num_annotations_skipped)


def main(_):
  assert FLAGS.data_dir, '`data_dir` missing.'
  assert FLAGS.train_annotations_file, '`train_annotations_file` missing.'
  assert FLAGS.val_annotations_file, '`val_annotations_file` missing.'
  assert FLAGS.testdev_annotations_file, '`testdev_annotations_file` missing.'
  assert FLAGS.labels_file, '`labels_file` missing.'

  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)
  train_output_path = os.path.join(FLAGS.output_dir, 'df_train.record')
  val_output_path = os.path.join(FLAGS.output_dir, 'df_val.record')
  testdev_output_path = os.path.join(FLAGS.output_dir, 'df_testdev.record')

  _create_tf_record_from_df_annotations(
    FLAGS.train_annotations_file,
    FLAGS.labels_file,
    FLAGS.data_dir,
    train_output_path)
  _create_tf_record_from_df_annotations(
    FLAGS.val_annotations_file,
    FLAGS.labels_file,
    FLAGS.data_dir,
    val_output_path)
  _create_tf_record_from_df_annotations(
    FLAGS.testdev_annotations_file,
    FLAGS.labels_file,
    FLAGS.data_dir,
    testdev_output_path)


if __name__ == '__main__':
  tf.app.run()
