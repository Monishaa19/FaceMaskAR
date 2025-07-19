from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import io
from PIL import Image
from tensorflow.keras.models import load_model
from detect_mask_video import detect_and_predict_mask

# Load models once when server starts
faceNet = cv2.dnn.readNet("face_detector/deploy.prototxt",
                          "face_detector/res10_300x300_ssd_iter_140000.caffemodel")
maskNet = load_model("mask_detector.model")

app = Flask(__name__)

@app.route("/detect_mask", methods=["POST"])
def detect_mask():
    data = request.json["image"]
    image_bytes = base64.b64decode(data.split(",")[1])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    results = []
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        label = "Mask" if mask > withoutMask else "No Mask"
        probability = float(max(mask, withoutMask))
        results.append({
            "box": [int(startX), int(startY), int(endX), int(endY)],
            "label": label,
            "probability": probability
        })

    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
