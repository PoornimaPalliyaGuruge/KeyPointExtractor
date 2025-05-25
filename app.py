# from flask import Flask, request, jsonify
# import joblib
# import numpy as np

# app = Flask(__name__)

# # Load the model
# model = joblib.load("gesture_model_datao.pkl")

# # All 4 classes
# classes = ['anomalies', 'open_palms', 'self_referencing', 'steepled']

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     keypoints = data.get('keypoints')

#     if not keypoints or len(keypoints) != 132:
#         return jsonify({"error": "Expected 132 keypoints"}), 400

#     np_keypoints = np.array(keypoints).reshape(1, -1)

#     # Predict class
#     pred = model.predict(np_keypoints)[0]

#     # Get probability scores
#     proba = model.predict_proba(np_keypoints)[0]

#     # Map class names to scores
#     confidence_dict = {cls: float(np.round(score * 100, 2)) for cls, score in zip(classes, proba)}

#     # Final score is the confidence of the predicted class
#     final_score = float(np.round(confidence_dict.get(pred, 0.0), 2))

#     return jsonify({
#         "prediction": pred,
#         "confidence_scores": confidence_dict,
#         "final_score": final_score  # Add this line
#     })

# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=5000, debug=True)


from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load model
model = joblib.load("gesture_model_datao.pkl")  # Replace with your path
classes = ['anomalies', 'open_palms', 'self_referencing', 'steepled']

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def extract_keypoints(image):
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
        return np.array(keypoints)
    return None

@app.route('/analyze-interview', methods=['POST'])
def analyze_interview():
    # Expecting a list of images in base64 or paths (adjust as per client)
    images = request.json.get('image_paths', [])

    count = {"anomalies": 0, "open_palms": 0, "self_referencing": 0, "steepled": 0}
    total_valid = 0

    for path in images:
        image = cv2.imread(path)
        if image is None:
            continue
        keypoints = extract_keypoints(image)
        if keypoints is not None and len(keypoints) == 132:
            pred = model.predict([keypoints])[0]
            count[pred] += 1
            total_valid += 1

    if total_valid == 0:
        return jsonify({"error": "No valid frames detected"}), 400

    # Calculate percentages
    summary = {cls: round((count[cls] / total_valid) * 100, 2) for cls in classes}

    # Example logic for final score:
    # Positive = 1.0 * positive % + 0.5 * neutral % - 1.0 * anomaly %
    score = (summary['open_palms'] + summary['steepled']) * 1.0 \
          + summary['self_referencing'] * 0.5 \
          - summary['anomalies'] * 1.0
    score = round(score / 2.0, 2)  # Scale to 0-100 (adjust based on formula logic)

    return jsonify({
        "summary": summary,
        "final_score": score,
        "frame_count": total_valid
    })

if __name__ == '__main__':
    app.run(debug=True)