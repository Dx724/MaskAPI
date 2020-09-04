import flask
from flask import jsonify, request, render_template
import base64
import re
from PIL import Image
from io import BytesIO
import sys, traceback
import numpy as np
import tensorflow as tf
import cv2
import requests
from functools import wraps

model = tf.keras.models.load_model("mask_model")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_cascade_2 = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

app = flask.Flask(__name__)
app.config["DEBUG"] = False

def crop_to_face(img): #Returns (faceFound, croppedImage)
  cvImg = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) #OpenCV Color Representation
  cvGray = cv2.cvtColor(cvImg, cv2.COLOR_BGR2GRAY) #OpenCV Grayscale Representation
  faces = face_cascade.detectMultiScale(cvGray, 1.25, 5)
  if (len(faces) < 1):
    faces.extend(face_cascade_2.detectMultiScale(cvGray, 1.25, 5))
    if (len(faces) < 1):
      return (False, None)
  largestIdx = 0
  largestSize = 0
  fIdx = 0
  for face in faces:
    fSize = face[2] * face[3]
    if fSize > largestSize:
      largestSize = 0
      largestIdx = fIdx
    fIdx += 1
  targetFace = faces[largestIdx]
  print("Target Face", targetFace)
  return (True, Image.fromarray(cv2.cvtColor(cvImg[targetFace[1]:targetFace[1]+targetFace[3], targetFace[0]:targetFace[0]+targetFace[2]], cv2.COLOR_BGR2RGB)))
  
def counter_hit(keyName):
  c = requests.get("https://api.countapi.xyz/hit/dx724.maskdetect/" + keyName)
  print("Counter", keyName, c.json()["value"])

def error_to_500(f):
  @wraps(f)
  def wrap500(*args, **kwargs):
    try:
      return f(*args, **kwargs)
    except:
      print("Arbitrary Error")
      #print(sys.exc_info()[0])
      traceback.print_exc(file=sys.stdout)
      print("When Calling:", f.__name__)
      counter_hit("maskapi_arbitraryerror")
      return flask.jsonify({"error": "Unknown Server Error"}), 500
  return wrap500
    

@app.route("/", methods=["GET"])
@error_to_500
def home():
  return "<h1>MaskAPI</h1>"

@app.route("/api/test", methods=["GET"])
@error_to_500
def api_test():
  testVal = 0
  if "t" in request.args:
    testVal = int(request.args["t"])
  return jsonify({'a': 3, 'b': 5, 't': testVal})

@app.route("/api/imgtest", methods=["POST"])
@error_to_500
def api_imgtest():
  testVal = 0
  if "img" in request.json:
    img_data = re.sub("^data:image/.+;base64,", "", request.json["img"])
    im = Image.open(BytesIO(base64.b64decode(img_data)))
    im = im.convert("RGB")
    im = im.resize((224, 244)).convert("RGB")
    outputBuffer = BytesIO()
    im.save(outputBuffer, format="JPEG")
    b64new = bytes("data:image/jpeg;base64,", encoding="utf-8") + base64.b64encode(outputBuffer.getvalue())
    return render_template("imageview.html", imgData = b64new.decode("utf-8")), 200
  else:
    return jsonify({"result": "error"}), 400 #400: Bad Input

@app.route("/api/detect_mask", methods=["POST"])
@error_to_500
def api_detect_mask():
  if "img" in request.json:
    try:
      counter_hit("maskapicall")
      img_data = re.sub("^data:image/.+;base64,", "", request.json["img"])
      im = Image.open(BytesIO(base64.b64decode(img_data)))
      im = im.convert("RGB")
      doFindFace = ("find_face" in request.json) and request.json["find_face"] == True
      print("DoFindFace?", doFindFace)
      if doFindFace:
        counter_hit("maskapi_findface")
        faceFound, imNew = crop_to_face(im)
        print("Face Search Result:", faceFound)
        if faceFound:
          counter_hit("maskapi_findface_success")
          im = imNew
          #return jsonify({"error": "No Face Detected"}), 400 --> Sometimes an up-close masked face isn't found by the haar cascade
      im = im.resize((224, 244))
      iData = np.asarray(im)/255
      print(iData)
      predResult = model(iData[np.newaxis],training=False).numpy()[0][0]
      print("Prediction Result", str(predResult))
      return jsonify({"is_wearing_mask": bool(round(predResult)),
        **({"face_found": faceFound} if doFindFace else {})
      }), 200
    except:
      print("TensorFlow Predict Error")
      traceback.print_exc(file=sys.stdout)#print(sys.exc_info()[0])
      print("Image: ", request.json["img"])
      counter_hit("maskapi_error")
      return jsonify({"error": "Service Error"}), 500 #500: Internal Server Error
  else:
    return jsonify({"error": "No Image Provided"}), 400 #400: Bad Input

@app.route("/frontend/imgtest", methods=["GET"])
@error_to_500
def frontend_imgtest():
  return render_template("imageview_fe.html"), 200

@app.route("/frontend/detectmask", methods=["GET"])
@error_to_500
def frontend_detectmask():
  return render_template("detectmask_fe.html"), 200

app.run(host="0.0.0.0", port="8080")