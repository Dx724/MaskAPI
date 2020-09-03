import flask
from flask import jsonify, request
import base64
import re
from PIL import Image
from io import BytesIO

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route("/", methods=["GET"])
def home():
  return "<h1>MaskAPI</h1>"

@app.route("/api/test", methods=["GET"])
def api_test():
  testVal = 0
  if "t" in request.args:
    testVal = int(request.args["t"])
  return jsonify({'a': 3, 'b': 5, 't': testVal})

@app.route("/api/imgtest", methods=["POST"])
def api_imgtest():
  testVal = 0
  if "img" in request.form:
    img_data = re.sub("^data:image/.+;base64,", "", request.form["img"])
    im = Image.open(BytesIO(base64.b64decode(img_data)))
    return jsonify({"result": "success"}), 200
  else:
    return jsonify({"result": "error"}), 400 #400: Bad Input

app.run(host="0.0.0.0", port="8080")