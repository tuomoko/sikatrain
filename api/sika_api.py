import json, argparse, time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
import tensorflow as tf

from flask import Flask, request, url_for, render_template, send_file, redirect, jsonify, send_from_directory, g

from collections import defaultdict
from io import StringIO, BytesIO
from PIL import Image
import numpy as np
import uuid

#from OpenSSL import SSL
import ssl

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

#TODO some better way to store the location of the model and label map

frozen_model_filename=os.environ['FROZEN_MODEL'] # Path to frozen model
label_map_path=os.environ['LABEL_MAP'] # Path to label map
num_labels=6 # Number of labels
gpu_memory=.5 # GPU memory per process

# Load label maps
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_labels, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Function that is called once (per worker?) when the application is loaded
def start_tf(app):
    # Load detection graph
    print('Loading the model')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(frozen_model_filename, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    #graph = load_graph(frozen_model_filename)
    #x = graph.get_tensor_by_name('prefix/Placeholder/inputs_placeholder:0')
    #y = graph.get_tensor_by_name('prefix/Accuracy/predictions:0')
    
    # Tensors
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    fetches = [detection_boxes, detection_scores, detection_classes, num_detections]

    print('Starting Session, setting the GPU memory usage to %f' % gpu_memory)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    persistent_sess = tf.Session(graph=detection_graph, config=sess_config)

    app.config['tf_sess'] = persistent_sess;
    app.config['tf_fetches'] = fetches;
    app.config['tf_image_tensor'] = image_tensor;

# Helper function
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

# General application configurations
app = Flask(__name__)
IM_SCALED_SIZE = 320, 320
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
SCORE_THR = 0.5
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

start_tf(app)

##################################################
# API part
##################################################

@app.route("/api/predict", methods=['POST'])
def predict():
    start = time.time()
    return_data = []
    # check if the post request has the file part
    if 'file' not in request.files:
        print('No file part')
        print(str(request.files))
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        print('No selected file')
        return redirect(request.url)
    if file:
        #filename = file.filename
        name = str(uuid.uuid4())
        upload_filename = name + '.png'
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], upload_filename)
        file.save(full_filename)
        print("serving: "+full_filename)
        image = Image.open(full_filename)
        image = image.convert('RGB')
        image.thumbnail(IM_SCALED_SIZE)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        
        ##################################################
        # Tensorflow part
        ##################################################
        sess = app.config['tf_sess']
        fetches = app.config['tf_fetches'];
        image_tensor = app.config['tf_image_tensor'];
        (boxes, scores, classes, num) = sess.run(fetches,
            feed_dict={image_tensor: image_np_expanded})
        ##################################################
        # END Tensorflow part
        ##################################################
        
        class_squeeze = np.squeeze(classes).astype(np.int32)
        boxes_squeeze = np.squeeze(boxes)
        scores_squeeze = np.squeeze(scores)
        
        if scores_squeeze[1] > SCORE_THR:
            print "Two pigs found"
            pig1 = category_index[class_squeeze[0]]['name']
            pig2 = category_index[class_squeeze[1]]['name']
        elif scores_squeeze[0] > SCORE_THR:
            print "One pig found"
            pig1 = 'bacon'
            pig2 = 'bacon'
        else:
            print "No pigs found"
            pig1 = ""
            pig2 = ""
        
        print "Pig 1 = " + pig1 + " Pig 2 = " + pig2
        
        # Reopen the full image after classification
        image = Image.open(full_filename)
        image = image.convert('RGB')
        image_np = load_image_into_numpy_array(image)
        
        #Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
             image_np,
             np.squeeze(boxes),
             np.squeeze(classes).astype(np.int32),
             np.squeeze(scores),
             category_index,
             use_normalized_coordinates=True,
             line_thickness=2,
             max_boxes_to_draw=2,
             min_score_thresh=SCORE_THR)
        
        processed_image = Image.fromarray(image_np)
        processed_filename = name + '.jpg'
        full_filename = os.path.join(PROCESSED_FOLDER, processed_filename)
        processed_image.save(full_filename, 'JPEG', quality=90)
        
        return_data = dict()
        return_data['pig1'] = str(pig1)
        return_data['pig2'] = str(pig2)
        return_data['img_url'] = os.path.join('processed', processed_filename)
        
        #retvalue = json.dumps(data)
        #retvalue = serve_pil_image(processed_image)
        #plt.figure(figsize=IMAGE_SIZE)
        #plt.imshow(image_np)
        #plt.show()
        
    print("Time spent handling the request: %f" % (time.time() - start))
    
    print "Returning "+str(return_data)
    return jsonify(return_data)
##################################################
# END API part
##################################################

##################################################
# Serving the images
##################################################
@app.route('/processed/<path:filename>')
def download_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename, as_attachment=True)
##################################################
# END Serving the images
##################################################

##################################################
# Index and other stuff
##################################################
@app.route('/')
def index():
    return render_template('index.html')
##################################################
# END Index and other stuff
##################################################




# Main function
if __name__ == "__main__":
    #context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    #context.load_cert_chain('sika_api.crt', 'sika_api.key')
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0 # Prevent caching js files

    print('Starting the API')
    app.run(host= '0.0.0.0') #, ssl_context=context)
    
    #Serve javascript and CSS files
    url_for('static', filename='app.js')
    url_for('static', filename='style.css')
