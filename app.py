from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load("gesture_model.pkl")

# All 4 classes
classes = ['anomalies', 'open_palms', 'self_referencing', 'steepled']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    keypoints = data.get('keypoints')  # a list of 33 * 4 = 132 keypoints [x1, y1, z1, vis1, ..., x33, y33, z33, vis33]

    if not keypoints or len(keypoints) != 132:
        return jsonify({"error": "Expected 132 keypoints"}), 400

    np_keypoints = np.array(keypoints).reshape(1, -1)

    pred = model.predict(np_keypoints)[0]
    proba = model.predict_proba(np_keypoints)[0]

    confidence_dict = {cls: float(np.round(score * 100, 2)) for cls, score in zip(classes, proba)}
    
    return jsonify({
        "prediction": pred,
        "confidence_scores": confidence_dict
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
