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
"""Test for create_df_tf_record.py."""

import io
import os

import numpy as np
import PIL.Image
import tensorflow as tf

from object_detection.dataset_tools import create_df_tf_record


class CreateDFTFRecordTest(tf.test.TestCase):

  def _assertProtoEqual(self, proto_field, expectation):
    """Helper function to assert if a proto field equals some value.

    Args:
      proto_field: The protobuf field to compare.
      expectation: The expected value of the protobuf field.
    """
    proto_list = [p for p in proto_field]
    self.assertListEqual(proto_list, expectation)

  def test_create_tf_example(self):
    image_file_name = 'tmp_image.jpg'
    image_data = np.random.rand(256, 256, 3)
    tmp_dir = self.get_temp_dir()
    save_path = os.path.join(tmp_dir, image_file_name)
    image = PIL.Image.fromarray(image_data, 'RGB')
    image.save(save_path)

    image = [11, image_file_name]

    annotations_list = [[64, 64, 192, 192, 2]]

    data_dir = tmp_dir
    label_index = {
        1: 'dog',
        2: 'cat',
        3: 'human'
    }

    (_, example,
     num_annotations_skipped) = create_df_tf_record.create_tf_example(
         image, annotations_list, data_dir, label_index)

    self.assertEqual(num_annotations_skipped, 0)
    self._assertProtoEqual(
        example.features.feature['image/height'].int64_list.value, [256])
    self._assertProtoEqual(
        example.features.feature['image/width'].int64_list.value, [256])
    self._assertProtoEqual(
        example.features.feature['image/filename'].bytes_list.value,
        [image_file_name])
    self._assertProtoEqual(
        example.features.feature['image/source_id'].bytes_list.value,
        [str(image[0])])
    self._assertProtoEqual(
        example.features.feature['image/format'].bytes_list.value, ['jpeg'])
    self._assertProtoEqual(
        example.features.feature['image/object/bbox/xmin'].float_list.value,
        [0.25])
    self._assertProtoEqual(
        example.features.feature['image/object/bbox/ymin'].float_list.value,
        [0.25])
    self._assertProtoEqual(
        example.features.feature['image/object/bbox/xmax'].float_list.value,
        [0.75])
    self._assertProtoEqual(
        example.features.feature['image/object/bbox/ymax'].float_list.value,
        [0.75])


if __name__ == '__main__':
  tf.test.main()
