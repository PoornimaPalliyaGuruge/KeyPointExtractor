from flask import Flask, request, jsonify
import joblib
import numpy as np
import requests
import tempfile
import cv2
import mediapipe as mp

app = Flask(__name__)

# Load the model
model = joblib.load("gesture_model_datao.pkl")

# All 4 classes
classes = ['anomalies', 'open_palms', 'self_referencing', 'steepled']

@app.route('/', methods=['GET'])
def root_url():
    return jsonify({"message": "Welcome to the Gesture Recognition API"})

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

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    data = request.get_json()
    frames = data.get('frames')  

    if not frames or not isinstance(frames, list):
        return jsonify({"error": "Expected list of keypoints per frame"}), 400

    count = {cls: 0 for cls in classes}
    total_frames = 0
    final_score_accumulator = 0.0

    for keypoints in frames:
        if len(keypoints) != 132:
            continue  # Skip invalid frames
        np_keypoints = np.array(keypoints).reshape(1, -1)
        pred = model.predict(np_keypoints)[0]
        proba = model.predict_proba(np_keypoints)[0]

        # Count predicted class
        count[pred] += 1
        total_frames += 1

        # Apply your scoring formula
        score = (
            proba[classes.index('open_palms')] * 1.0 +
            proba[classes.index('steepled')] * 1.0 +
            proba[classes.index('self_referencing')] * 0.5
        )
        final_score_accumulator += score

    if total_frames == 0:
        return jsonify({"error": "No valid frames processed"}), 400

    # Summary percentages
    summary = {cls: round((count[cls] / total_frames) * 100, 2) for cls in classes}
    average_final_score = round((final_score_accumulator / total_frames) * 100 / 2.0, 2)

    return jsonify({
        "summary": summary,
        "average_final_score": average_final_score,
        "frame_count": total_frames
    })

@app.route('/predict-video', methods=['POST'])
def predict_video():
    data = request.get_json()
    video_url = data.get('video_url')

    if not video_url:
        return jsonify({"error": "Missing video URL"}), 400

    try:
        # Step 1: Download video from Firebase to temp file
        response = requests.get(video_url, stream=True)
        if response.status_code != 200:
            return jsonify({"error": "Failed to download video"}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                temp_video.write(chunk)
            video_path = temp_video.name

        # Step 2: Extract frames and keypoints
        cap = cv2.VideoCapture(video_path)
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False)

        frames_keypoints = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                keypoints = []
                for lm in landmarks:
                    keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
                if len(keypoints) == 132:
                    frames_keypoints.append(keypoints)

        pose.close()
        cap.release()

        if len(frames_keypoints) == 0:
            return jsonify({"error": "No valid keypoints detected in video"}), 400

        # Step 3: Predict using model
        count = {cls: 0 for cls in classes}
        total_frames = 0
        final_score_accumulator = 0.0

        for keypoints in frames_keypoints:
            np_keypoints = np.array(keypoints).reshape(1, -1)
            pred = model.predict(np_keypoints)[0]
            proba = model.predict_proba(np_keypoints)[0]

            count[pred] += 1
            total_frames += 1

            score = (
                proba[classes.index('open_palms')] * 1.0 +
                proba[classes.index('steepled')] * 1.0 +
                proba[classes.index('self_referencing')] * 0.5
            )
            final_score_accumulator += score

        summary = {cls: round((count[cls] / total_frames) * 100, 2) for cls in classes}
        average_final_score = round((final_score_accumulator / total_frames) * 100 / 2.0, 2)

        return jsonify({
            "summary": summary,
            "average_final_score": average_final_score,
            "frame_count": total_frames
        })
    except Exception as e:
        return jsonify({"error": f"Error processing video: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)