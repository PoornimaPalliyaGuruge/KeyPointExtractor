from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load("gesture_model_datao.pkl")

# All 4 classes
classes = ['anomalies', 'open_palms', 'self_referencing', 'steepled']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    keypoints = data.get('keypoints')

    if not keypoints or len(keypoints) != 132:
        return jsonify({"error": "Expected 132 keypoints"}), 400

    np_keypoints = np.array(keypoints).reshape(1, -1)

    # Predict class
    pred = model.predict(np_keypoints)[0]

    # Get probability scores
    proba = model.predict_proba(np_keypoints)[0]

    # Map class names to scores
    confidence_dict = {cls: float(np.round(score * 100, 2)) for cls, score in zip(classes, proba)}

    # Final score is the confidence of the predicted class
    final_score = float(np.round(confidence_dict.get(pred, 0.0), 2))

    return jsonify({
        "prediction": pred,
        "confidence_scores": confidence_dict,
        "final_score": final_score  # Add this line
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
