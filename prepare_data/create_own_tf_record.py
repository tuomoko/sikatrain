# Modified from TensorFlow object detection code in
# https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pascal_tf_record.py
# 
# Changed input data structure from XML to JSON
# Added randomizing to split the data into training and validation data 
#
# Original license:
# 
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

r"""Convert raw JSON + JPEG dataset to TFRecord for object_detection.

Example usage:
    python create_own_tf_record.py \
        --input_file=/home/user/data.json \
        --input_path=/home/user \
        --output_path=/home/user/train \
        --label_map_path=/home/user/train/pig_label_map.pbtxt
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

import json
import random

from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags
flags.DEFINE_string('input_file', '', 'Input directory to raw JSON dataset.')
flags.DEFINE_string('input_path', '', 'Input directory where the images are.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/pig_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_float('div_traindata', 0.8, 'Portion of data used for training. Rest is for validation data')
FLAGS = flags.FLAGS



def dict_to_tf_example(data,
                       label_map_dict,
                       input_path,
                       ignore_difficult_instances=False):
    """Convert dict to tf.Example proto.

    Notice that this function does NOT normalize the bounding box coordinates provided
    by the raw data.

    Args:
    data: dict holding necessary fields for a single image
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).

    Returns:
    example: The converted tf.Example.

    Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    img_path = data['filename']
    full_path = train_file = os.path.join(input_path, img_path)
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(data['width'])
    height = int(data['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    #for obj in data['object']:
    obj = data['object']
    difficult = bool(int(obj['difficult']))
    #if ignore_difficult_instances and difficult:
    #  continue

    difficult_obj.append(int(difficult))

    xmin.append(float(obj['bbox']['xmin']))
    ymin.append(float(obj['bbox']['ymin']))
    xmax.append(float(obj['bbox']['xmax']))
    ymax.append(float(obj['bbox']['ymax']))
    classes_text.append(obj['name'].encode('utf8'))
    classes.append(label_map_dict[obj['name']])
    truncated.append(int(obj['truncated']))
    poses.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example


def main(_):
    input_file = FLAGS.input_file
    div_traindata = FLAGS.div_traindata

    full_dataset = json.load(open(input_file))
    
    train_file = os.path.join(FLAGS.output_path, 'train.tfrecord')
    val_file = os.path.join(FLAGS.output_path, 'val.tfrecord')
    
    random.seed(42)
    random.shuffle(full_dataset)
    num_examples = len(full_dataset)
    if div_traindata > 0 and div_traindata < 1:
        num_train = int(div_traindata * num_examples)
    else:
        num_train = num_examples
    train_dataset = full_dataset[:num_train]
    val_dataset = full_dataset[num_train:]

    writer = tf.python_io.TFRecordWriter(train_file)

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    
    logging.info('Reading training data from %s dataset.', input_file)
    
    for idx, example in enumerate(train_dataset):
      if idx % 100 == 0:
        logging.info('On image %d of %d', idx, len(train_dataset))
      tf_example = dict_to_tf_example(example, label_map_dict, FLAGS.input_path)
      writer.write(tf_example.SerializeToString())
    
    writer.close()
    
    if len(val_dataset) > 0:
        writer = tf.python_io.TFRecordWriter(val_file)
        logging.info('Reading validation data from %s dataset.', input_file)
        
        for idx, example in enumerate(val_dataset):
          if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(val_dataset))
          tf_example = dict_to_tf_example(example, label_map_dict, FLAGS.input_path)
          writer.write(tf_example.SerializeToString())
    
        writer.close()


if __name__ == '__main__':
  tf.app.run()
