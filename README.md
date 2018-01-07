# sikatrain
Tensorflow based object classifier for [Pass the Pigs](https://en.wikipedia.org/wiki/Pass_the_Pigs) dices

## Prerequisites
* Tensorflow and all it's dependencies. Tested with tensorflow-gpu 1.4.1, cuda-8.0, cudnn-6.0.21
* Tensorflow object detection API https://github.com/tensorflow/models/tree/master/research/object_detection
* Note that paths tensorflow/models/research/ and slim subdirectory should be appended to PYTHONPATH (like the object detection API installation instructions mention)

## Generating training data
The training data is generated from video files taken from the dices. This makes it easier to get large number of training pictures. The files to prepare the data are located in **prepare_data/** folder.
Note that each video file should only contain one result (e.g. leaning jowler)
From the video files, every 10th frame is taken and a very simple object detection algorithm is used to find the pig and draw a bounding box around it.
This data is then converted to TFRecord files, that can be crunched by the Tensorflow

First, the video files are converted to JSON data and JPEG images with a script **convert_train_data.py**
The script will need to be modified to change the input files. The script can be run by
```
python convert_train_data.py
```
This script saves the raw jpg files to jpg/ subdirectory and a JSON file containing all the classification data to json_data.txt. Additionally debug images are saved to debug/ subdirectory. Here, one should verify that the bounding boxes are ok. If they are not ok, the detection threshold may need adjustments.

After creating JSON and JPEG data, they can be transformed to TFRecord file. It can be done with the tool:
```
python create_own_tf_record.py \
    --input_file=json_data.txt \
    --input_path=/path/to/sikatrain/prepare_data \
    --output_path=/path/to/sikatrain/model/inputs \
    --label_map_path=/path/to/sikatrain/model/pig_label_map.pbtxt
```

Now you should have all the data needed for training the model! Notice that if you run these straight from the GitHub, you will overwrite the TFRecord files created with a larger dataset.

## Training
I used faster_rcnn_resnet101_coco model from object detection API. I took a fully trained model to my model checkpoint to make the learning faster. The trained models can be found from the [object detection API documentation](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

The files to train the model are located in **model/** folder. In addition to the files in GitHub, I recommend you to download my pretrained model from http://vider.kapsi.fi/sikatrain/sikatrain_graph.tar.gz This model should be unpacked to /path/to/sikatrain/model/graph . When this model is unpacked, you can also directly run the webserver without any training.

The training is quite heavy process. I noticed that with my desktop system I couldn't have a webbrowser with many tabs open when starting the training, since it would not get enough memory. Maybe it's best to do the training without X window system open at all.

I was not able to run the evaluation in parallel with training, but apparently it can be done.

The configuration file **faster_rcnn_resnet101_coco.config** can be edited for improved performance. Probably the following items could be tweaked:
* train_config/batch_size
* train_input_reader/queue_capacity
* train_input_reader/min_after_dequeue

The training can be run with the following command. It should be run in the **/path/to/sikatrain/model** path and the path in the command should refer to the object detection API path.
```
python /path/to/models/research/object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=faster_rcnn_resnet101_coco.config \
    --train_dir=train
```

The evaluation should be run with the following command, but I couldn't run it properly.
```
python /path/to/models/research/object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=faster_rcnn_resnet101_coco.config \
    --checkpoint_dir=train \
    --eval_dir=eval
```

You may run the Tensorboard to see how the model is improving. Of course this requires also some resources, so depending on the machine you are running this, it may not be feasible to run it simultaneously.
```
tensorboard --logdir=/path/to/sikatrain/model
```

After sufficient training, you can export the model with the following command. Replace the XXX with the latest model checkpoint.
```
python /path/to/models/research/object_detection/export_inference_graph.py \
    --input_type=image_tensor \
    --pipeline_config_path=faster_rcnn_resnet101_coco.config \
    --trained_checkpoint_prefix=train/model.ckpt-XXX \
    --output_directory=graph
```

## Webserver + API
There is a flask webserver that is used to serve a webpage where a user can take a picture and it's automatically classified.
The webserver and web API is located in **api/** folder. Note that you will need a graph generated from a trained model. See the instructions in the **Training** section to seee where to download the data.

The picture taking part is implemented using javascript and WebRTC and the API is running with flask in python and uses Tensorflow and Tensorflow object detection API to classify objects in images.

Since many browsers require that WebRTC page which uses camera is served via HTTPS, the webserver (Flask) is configured for that. You need to create SSL keys.
For example the following command may be used to create self signed key. Note that CN should be the same as your public IP or hostname.
```
openssl req \
       -newkey rsa:2048 -nodes -keyout sika_api.key \
       -x509 -days 365 -out sika_api.crt
```

You can run the webserver with the command:
```
python sika_api.py \
    --frozen_model_filename=/path/to/sikatrain/model/graph/frozen_inference_graph.pb \
    --label_map_path=/path/to/sikatrain/model/pig_label_map.pbtxt \
    --num_labels=6
```


