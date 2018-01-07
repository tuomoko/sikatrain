import json, argparse, time

import tensorflow as tf

from flask import Flask, request, url_for, render_template, send_file, redirect
from flask_cors import CORS

from collections import defaultdict
from io import StringIO, BytesIO
from PIL import Image
import os
import numpy as np

#from OpenSSL import SSL
import ssl

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

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
IM_SCALED_SIZE = 720, 405
UPLOAD_FOLDER = 'uploads'
app.config['IM_SCALED_SIZE'] = IM_SCALED_SIZE
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SCORE_THR'] = 0.5
cors = CORS(app)

##################################################
# API part
##################################################

@app.route("/api/predict", methods=['POST'])
def predict():
    start = time.time()
    retvalue = ""
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
        filename = file.filename
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(full_filename)
        print("serving: "+full_filename)
        image = Image.open(full_filename)
        image = image.convert('RGB')
        image.thumbnail(app.config['IM_SCALED_SIZE'])
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        
        ##################################################
        # Tensorflow part
        ##################################################
        (boxes, scores, classes, num) = persistent_sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        ##################################################
        # END Tensorflow part
        ##################################################
        
        class_squeeze = np.squeeze(classes).astype(np.int32)
        boxes_squeeze = np.squeeze(boxes)
        scores_squeeze = np.squeeze(scores)
        
        indexes = list()
        for idx, val in enumerate(scores_squeeze):
            if val > app.config['SCORE_THR']:
                indexes.append(idx)
            else:
                break
        
        retvalue += "Pigs found: "+str(len(indexes))+"\n"
        for idx in indexes:
            retvalue += "Pig: "+str(idx)+" is "+category_index[class_squeeze[idx]]['name']+"\n"
        print(retvalue)
        
        
        
        #Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
             image_np,
             np.squeeze(boxes),
             np.squeeze(classes).astype(np.int32),
             np.squeeze(scores),
             category_index,
             use_normalized_coordinates=True,
             line_thickness=4)
        
        return_image = Image.fromarray(image_np)
        retvalue = serve_pil_image(return_image)
        #plt.figure(figsize=IMAGE_SIZE)
        #plt.imshow(image_np)
        #plt.show()
        
    print("Time spent handling the request: %f" % (time.time() - start))
    
    return retvalue
##################################################
# END API part
##################################################

##################################################
# Index and other stuff
##################################################
@app.route('/')
def index():
    return render_template('index.html')
##################################################
# END API part
##################################################




# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="results/frozen_model.pb", type=str, help="Frozen model file to import")
    parser.add_argument("--label_map_path", default="label_map.pbtxt", type=str, help="Label file")
    parser.add_argument("--num_labels", default=6, type=int, help="Number of labels")
    parser.add_argument("--gpu_memory", default=.2, type=float, help="GPU memory per process")
    args = parser.parse_args()
    
    ##################################################
    # Tensorflow part
    ##################################################
    
    # Load detection graph
    print('Loading the model')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(args.frozen_model_filename, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    # Load label map
    label_map = label_map_util.load_labelmap(args.label_map_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=args.num_labels, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    
    
    #graph = load_graph(args.frozen_model_filename)
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

    print('Starting Session, setting the GPU memory usage to %f' % args.gpu_memory)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    persistent_sess = tf.Session(graph=detection_graph, config=sess_config)
    ##################################################
    # END Tensorflow part
    ##################################################
    
    context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    context.load_cert_chain('sika_api.crt', 'sika_api.key')
    
    print('Starting the API')
    app.run(host= '0.0.0.0', ssl_context=context)
    
    #Serve javascript and CSS files
    url_for('static', filename='app.js')
    url_for('static', filename='style.css')
