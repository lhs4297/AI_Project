from flask import Flask, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import cv2
import torch
import numpy as np
from unet_model import UNet  # assuming your U-Net model class is named UNet and is in unet_model.py

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

# PyTorch 모델 로드
model = UNet()
model.load_state_dict(torch.load("unet_carvana_scale1.0_epoch2.pth", map_location=torch.device('cpu')))
model.eval()

# Assuming this function processes the image and returns the segmented image and a fixed value
def predict(image_path, model):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = np.transpose(image, (2, 0, 1))  # Convert to (C, H, W) format
    image = image / 255.0  # Normalize to [0, 1]
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        output = torch.sigmoid(output)
        output = output.squeeze().cpu().numpy()

    segmented_image = (output > 0.5).astype(np.uint8) * 255
    fixed_value = np.sum(segmented_image) / 255  # Example fixed value computation

    return segmented_image, fixed_value

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        input_image_path = file_path
        segmented_image, fixed_value = predict(input_image_path, model)
        
        # 파일명에서 확장자 제거
        file_base_name = os.path.splitext(filename)[0]
        
        # Ensure the processed file has the correct extension
        processed_image_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{file_base_name}.jpg")
        processed_value_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{file_base_name}_value.txt")
        
        if segmented_image is not None:
            cv2.imwrite(processed_image_path, segmented_image)
            with open(processed_value_path, 'w') as f:
                f.write(str(fixed_value))
            
            response = {
                "weight": f"{fixed_value:.3f}"
            }
            return jsonify(response)
        
        return jsonify({"error": "Failed to process image"}), 500
    
    return jsonify({"error": "Failed to upload image"}), 400

@app.route('/processed/weight/<filename>', methods=['GET'])
def get_weight(filename):
    value_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{filename}_value.txt")
    
    if not os.path.isfile(value_path):
        app.logger.error(f"File not found: {value_path}")
        return jsonify({"error": "File not found"}), 404
    
    try:
        with open(value_path, 'r') as f:
            fixed_value = float(f.read())
        response = {
            "weight": f"{fixed_value:.3f}"
        }
        return jsonify(response)
    except ValueError:
        app.logger.error(f"Invalid value in weight file: {value_path}")
        return jsonify({"error": "Invalid value in weight file"}), 500

@app.route('/processed/latest', methods=['GET'])
def get_latest_files():
    files = os.listdir(app.config['PROCESSED_FOLDER'])
    files.sort(key=lambda x: os.path.getctime(os.path.join(app.config['PROCESSED_FOLDER'], x)), reverse=True)
    latest_files = [file for file in files if file.endswith(('.jpg', '.jpeg', '.png'))][:10]
    return jsonify({"latest_files": latest_files})

@app.route('/processed/<filename>', methods=['GET'])
def get_processed_image(filename):
    # 파일 확장자가 있는 경우와 없는 경우를 모두 처리
    image_path_jpeg = os.path.join(app.config['PROCESSED_FOLDER'], f"{filename}.jpeg")
    image_path_jpg = os.path.join(app.config['PROCESSED_FOLDER'], f"{filename}.jpg")
    
    if os.path.isfile(image_path_jpeg):
        return send_from_directory(app.config['PROCESSED_FOLDER'], f"{filename}.jpeg")
    elif os.path.isfile(image_path_jpg):
        return send_from_directory(app.config['PROCESSED_FOLDER'], f"{filename}.jpg")
    else:
        app.logger.error(f"File not found: {filename}")
        return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    app.run(host='192.168.0.220', port=5000, debug=True)
