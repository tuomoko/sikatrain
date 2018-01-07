/*
 *  Copyright (c) 2015 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree.
 */

'use strict';

var errorElement = document.querySelector('#errorMsg');
var video = document.querySelector('video');
var snapbutton = document.getElementById('snap');
var canvas = document.getElementById('canvas');
var photo = document.getElementById('photo');

// Put variables in global scope to make them available to the browser console.
var constraints = window.constraints = {
  audio: false,
  video: {facingMode: "environment", width: 640}
};

function handleSuccess(stream) {
  var videoTracks = stream.getVideoTracks();
  console.log('Got stream with constraints:', constraints);
  console.log('Using video device: ' + videoTracks[0].label);
  stream.oninactive = function() {
    console.log('Stream inactive');
  };
  window.stream = stream; // make variable available to browser console
  video.srcObject = stream;
}

function handleError(error) {
  if (error.name === 'ConstraintNotSatisfiedError') {
    errorMsg('The resolution ' + constraints.video.width.exact + 'x' +
        constraints.video.width.exact + ' px is not supported by your device.');
  } else if (error.name === 'PermissionDeniedError') {
    errorMsg('Permissions have not been granted to use your camera and ' +
      'microphone, you need to allow the page access to your devices in ' +
      'order for the demo to work.');
  }
  errorMsg('getUserMedia error: ' + error.name, error);
}

function errorMsg(msg, error) {
  errorElement.innerHTML += '<p>' + msg + '</p>';
  if (typeof error !== 'undefined') {
    console.error(error);
  }
}

//**dataURL to blob**
function dataURLtoBlob(dataurl) {
    var arr = dataurl.split(','), mime = arr[0].match(/:(.*?);/)[1],
        bstr = atob(arr[1]), n = bstr.length, u8arr = new Uint8Array(n);
    while(n--){
        u8arr[n] = bstr.charCodeAt(n);
    }
    return new Blob([u8arr], {type:mime});
}

//**blob to dataURL**
function blobToDataURL(blob, callback) {
    var a = new FileReader();
    a.onload = function(e) {callback(e.target.result);}
    a.readAsDataURL(blob);
}


function takepicture() {
    var context = canvas.getContext('2d');
    
    var width = video.videoWidth;
    var height = video.videoHeight;
    if (width == 0) {
        width = 10;
        height = 10;
    }
    canvas.width = width;
    canvas.height = height;
    context.drawImage(video, 0, 0, width, height);
    
    
    var data = canvas.toDataURL('image/png');
    photo.setAttribute('src', data);
    //video.hide();
    
    console.log('Data:', data);
    
    // Create a new FormData object.
    var formData = new FormData();
    
    // Add the file to the request.
    formData.append('file', dataURLtoBlob(data), 'filename.png');
    
    // Set up the AJAX request.
    var xhr = new XMLHttpRequest();
    xhr.responseType    = "blob";
    
    // Open the connection.
    xhr.open('POST', '/api/predict', true);
    
    errorMsg('Starting file upload', 'fileupload');
        
    // Set up a handler for when the request finishes.
    xhr.onload = function () {
      if (xhr.status === 200) {
        errorMsg('The file uploaded successfully', 'fileupload');
        blobToDataURL(
            xhr.response,
            function(dataurl){
                console.log(dataurl);
                photo.setAttribute('src', dataurl);
            }
        );
      } else {
        errorMsg('An error occurred while uploading the file. Try again', 'fileupload');
      }
    };

    // Send the Data.
    xhr.send(formData);
    
  }

snapbutton.addEventListener('click', function(ev){
      takepicture();
      ev.preventDefault();
    }, false);


navigator.mediaDevices.getUserMedia(constraints).
    then(handleSuccess).catch(handleError);

