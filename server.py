from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

app = Flask(__name__)

# 모델 로드
MODEL_PATH = 'captcha_model_v2.h5'
model = load_model(MODEL_PATH)

# 숫자 라벨을 알파벳과 숫자로 변환하는 함수
def label_to_char(label):
    if label < 10:
        return chr(label + ord('0'))  # 숫자 '0'~'9'
    elif label < 36:
        return chr(label - 10 + ord('a'))  # 소문자 'a'~'z'
    else:
        return chr(label - 36 + ord('A'))  # 대문자 'A'~'Z'

# 이미지 예측 후 알파벳/숫자 변환
def predict_image(image_path, model):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (150, 40))  # 이미지 크기 변경
    image = image / 255.0  # 정규화
    image = image.reshape(1, 40, 150, 1)  # 배치 차원 추가

    # 예측하기
    predictions = model.predict(image)

    predicted_labels = []
    for i in range(5):
        predicted_class = np.argmax(predictions[i][0])  # 첫 번째 문자 예측
        predicted_labels.append(predicted_class)

    # 숫자 라벨을 문자로 변환
    predicted_chars = [label_to_char(label) for label in predicted_labels]

    return ''.join(predicted_chars)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # 업로드된 파일 저장
    uploads_folder = 'uploads'
    if not os.path.exists(uploads_folder):
        os.makedirs(uploads_folder)

    file_path = os.path.join(uploads_folder, file.filename)
    file.save(file_path)

    try:
        # 예측 실행
        predicted_text = predict_image(file_path, model)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(file_path)  # 임시 파일 삭제

    return jsonify({'prediction': predicted_text})

if __name__ == '__main__':
    app.run(debug=True)
