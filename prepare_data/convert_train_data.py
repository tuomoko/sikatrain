# This is the file which prepares videos. Generates still images in jpg/ folder and classification data in json_data.txt

import cv2
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import json
import os

# List all videos we want to use
# Tuple contains (video name without extension, class of the pig in the video, detection threshold)
# You may need to adjust the detection threshold based on lighting of the video. 
# The threshold is a difference between red pixels to blue and green pixels to detect a pig.

# This is the video available in GitHub as an example
#videos = [('IMG_4855','leaning_jowler',30)]

# This is the full dataset used to train the model
videos = [('IMG_4941','side',30),('IMG_4942','side_spot',30),('IMG_4943','trotter',30), \
    ('IMG_4944','razorback',30),('IMG_4946','snouter',30),('IMG_4947','leaning_jowler',30),('IMG_4948','leaning_jowler',45), \
    ('IMG_4949','snouter',45),('IMG_4950','razorback',45),('IMG_4951','trotter',45),('IMG_4952','side_spot',45), \
    ('IMG_4953','side',45)]

# Extension of the video files
videoext = '.MOV'

# These are the classes to use for training. This needs to match the numbering in pig_label_map.pbtxt
# Note that first class should never be used. It is there to convert from python 0 based arrays to 
# the 1 based label map that is required by Tensorflow
classes = [[], 'side','side_spot','razorback','trotter','snouter','leaning_jowler']

# Maximum image size
im_size = 480, 480

# Paths that are used for input and output files
videopath = 'video/'
imagepath = 'jpg/'
debugpath = 'debug/'


# Non-machine learning based find the pig function
def find_pig(image,th):
    mask = list()
    for px in image.getdata():
        if px[0] > (px[1]+th) and px[0] > (px[2]+th):
            mask.append(1)
        else:
            mask.append(0)
    mask_img = image.convert('1')
    mask_img.putdata(mask)
    boundary = mask_img.getbbox()
    if boundary:
        bound_large = (boundary[0]-10,boundary[1]-10,boundary[2]+10,boundary[3]+10)
    else:
        bound_large = (0,0,0,0)
    return bound_large

# Function to check whether the bounding box is limited to the image edge
# This is used to remove those frames that may contain only a part of the object
def check_bounding(box, size):
    if box[0] <= 0: #left
        return 0
    if box[1] <= 0: #upper
        return 0
    if box[2] >= size[0]: #right
        return 0
    if box[3] >= size[1]: #lower
        return 0
    return 1

# Function to save a debug image to help checking the result
def save_debug_image(image,boundbox,classidx,videoname,count):
    if debugpath:
        new_image = image.copy()
        actual_size = im.size
        draw = ImageDraw.Draw(new_image)
        draw.rectangle(boundbox, outline="red")
        text_x = boundbox[0]
        if boundbox[3] < (actual_size[1] - 15):
            text_y = boundbox[3] + 2
        elif boundbox[1] > 15:
            text_y = boundbox[1] - 15
        else:
            text_y = 0
        draw.text((text_x, text_y), classes[classidx], fill="red", font=ImageFont.truetype("arial.ttf", 12))
        debugname = debugpath+videoname+"_frame%d.jpg" % count
        new_image.save(debugname)

# Main loop
image_data = list()

for videotuple in videos:
    videoname = videotuple[0]
    videoclass = classes.index(videotuple[1])
    th = videotuple[2]
    vidcap = cv2.VideoCapture(videopath+videoname+videoext)
    success,image = vidcap.read()
    count = 0
    if success:
        print('Working on '+videoname+videoext+' with a class of '+classes[videoclass])
    while success:
      success,image = vidcap.read()
      if success and (count%10 == 0):
          imname = imagepath+videoname+"_frame%d.jpg" % count
          cv2.imwrite(imname, image)     # save frame as JPEG file
          im = Image.open(imname)
          im.thumbnail(im_size)
          actual_size = im.size
          im.save(imname)
          boundbox = find_pig(im,th)
          if check_bounding(boundbox, actual_size):
              save_debug_image(im,boundbox,videoclass,videoname,count)
              im_dict = dict()
              im_dict["filename"] = imname
              im_dict["height"] = actual_size[1]
              im_dict["width"] = actual_size[0]
              im_dict["object"] = dict()
              # Bounding boxes are scaled to 0...1 for Tensorflow
              im_dict["object"]["bbox"] = dict()
              im_dict["object"]["bbox"]["xmin"] = (1.0*boundbox[0])/actual_size[0]
              im_dict["object"]["bbox"]["ymin"] = (1.0*boundbox[1])/actual_size[1]
              im_dict["object"]["bbox"]["xmax"] = (1.0*boundbox[2])/actual_size[0]
              im_dict["object"]["bbox"]["ymax"] = (1.0*boundbox[3])/actual_size[1]
              im_dict["object"]["name"] = videotuple[1]
              im_dict["object"]['difficult'] = 0
              im_dict["object"]['truncated'] = 0
              im_dict["object"]['pose'] = 'Unspecified'
              image_data.append(im_dict)
      count += 1

# Write out the JSON data
with open('json_data.txt', 'w') as outfile:
    json.dump(image_data, outfile)
