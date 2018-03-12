from __future__ import print_function
import argparse
import numpy as np
import os

DEEP_FASHION_DIR = '/home/jaimin/Workspace/deep-fashion'
ANNO_DIR = os.path.join(DEEP_FASHION_DIR, 'anno')
TRAINING_DIR = os.path.join(DEEP_FASHION_DIR, 'training')
LABELS_NPY = os.path.join(ANNO_DIR, 'df_labels.npy')
LABELS_PBTXT = os.path.join(TRAINING_DIR, 'df_label_map.pbtxt')
BBOX_LIST = os.path.join(DEEP_FASHION_DIR, 'Category and Attribute Prediction Benchmark/Anno/list_bbox.txt')
EVAL_LIST = os.path.join(DEEP_FASHION_DIR, 'Category and Attribute Prediction Benchmark/Eval/list_eval_partition.txt')
CATEGORY_LIST = os.path.join(DEEP_FASHION_DIR,
                             'Category and Attribute Prediction Benchmark/Anno/list_category_cloth.txt')


def read_anno_map(list_path, anno_map):
  with open(list_path, 'r') as f:
    anno_len = int(f.readline())
    f.readline()
    for i in range(anno_len):
      anno = f.readline().split()
      image = anno[0]
      anno = anno[1:]
      if image not in anno_map:
        anno_map[image] = []
      anno_map[image].extend(anno)


def read_category_list(list_path):
  category_list = []
  with open(list_path, 'r') as f:
    anno_len = int(f.readline())
    f.readline()
    for i in range(anno_len):
      category_line = f.readline().split()

      category_label = category_line[0]
      id = int(category_line[1])
      category_list.append(category_label)
  return category_list


def create_label_pbtxt():
  category_list = read_category_list(CATEGORY_LIST)

  labels = []
  for idx, ele in enumerate(category_list, 1):
    labels.append([idx,ele])
  labels = np.asarray(labels)

  np.save(LABELS_NPY, labels)
  print('Saved ' + LABELS_NPY)

  item = '''item {{\n  name: "{1}"\n  id: {0}\n  display_name: "{2}"\n}}\n\n'''

  with open(LABELS_PBTXT, 'w') as pbfile:
    for label in labels:
      print(item.format(label[0], label[1].lower(), label[1].replace('_', ' '))[:-1])

  print('Saved ' + LABELS_PBTXT)
  return labels


def main():
  anno_map = {}

  labels = create_label_pbtxt()

  read_anno_map(BBOX_LIST, anno_map)

  for image_path in anno_map:
    image_group_name = os.path.basename(os.path.dirname(image_path))
    image_group_name = image_group_name.split('_')
    for x in reversed(image_group_name):
      label = labels[labels[:, 1] == x]
      if label.shape[0] > 0:
        label_id = label[0][0]
        anno_map[image_path] = map(int, anno_map[image_path])
        anno_map[image_path].append(label_id)
        break

  read_anno_map(EVAL_LIST, anno_map)

  train = []
  val = []
  test = []

  image_id = 1
  for k, v in anno_map.iteritems():
    section = v[-1]
    image = [image_id, k]
    image_id += 1
    image.extend(v[:-1])
    if section == 'train':
      train.append(image)
    elif section == 'val':
      val.append(image)
    else:
      test.append(image)

    print(image)

  print("[id, 'image_path', x_1, y_1, x_2, y_2, label_id]")

  train = np.asarray(train)
  val = np.asarray(val)
  test = np.asarray(test)

  train_file = os.path.join(ANNO_DIR, 'df_train.npy')
  val_file = os.path.join(ANNO_DIR, 'df_val.npy')
  test_file = os.path.join(ANNO_DIR, 'df_test.npy')

  print('Train shape: ', train.shape)
  print('Val shape: ', val.shape)
  print('Test shape: ', test.shape)

  np.save(train_file, train)
  print('Saved ' + train_file)

  np.save(val_file, val)
  print('Saved ' + val_file)

  np.save(test_file, test)
  print('Saved ' + test_file)


if __name__ == '__main__':
  main()
