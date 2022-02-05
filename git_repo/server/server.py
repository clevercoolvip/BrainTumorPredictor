from flask import *
from werkzeug.utils import secure_filename
import os
from tensorflow import keras
import numpy as np
import cv2


model = keras.models.load_model("../saved_model/model1")
class_names = ["negative", "positive"]


app = Flask(__name__)


@app.route("/")
def ping():
    return render_template("index.html")


@app.route("/", methods=["POST"])
def imageUpload():
    if request.method == "POST":
        if request.files:
            f = request.files["file"]
            f.save(os.path.join("D:/sector16/MRI SCAN/server/static", secure_filename(f.filename)))
            img = cv2.imread(f"static/{f.filename}")
            prediction = model.predict(np.expand_dims(img, axis=0))
            confidence = f"{prediction[0][0]*100}%"
            result = round(prediction[0][0])
            return render_template("index.html", data=class_names[result], imageSrc=f"static/{f.filename}", conf=confidence)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
