import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import tkinter
from tkinter import filedialog
import numpy as np
from flask import Flask,send_file
from flask_restful import Api, Resource, request
import os
from werkzeug.utils import secure_filename
from io import BytesIO



#---------------- Flask API Settup-------------------
app = Flask(__name__)
api = Api(app)
base_dir = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(base_dir, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
#---------------- End of Flask API Settup-------------


#--------------------Model Settup stuff----------------------------------
class DroneClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(DroneClassifier, self).__init__()
        self.model = models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.model(x)

model = DroneClassifier(num_classes=2)
model.load_state_dict(torch.load('best_drone_model_weights.pth', map_location=torch.device('cpu')))
model.eval()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the object detection model (Faster R-CNN)
detection_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
detection_model.eval()
detection_model.to(device)

# Function to make a prediction on a single image
def predict_image(image, model):
    # Preprocess the image
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Make the prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()

#---------------------------------End Of Model---------------------




#---------------- Displaying Returns---------------------------------

def display_prediction(image_path, model, detection_model,display):
    class_names = ['Drone', 'Not Drone']
    
    # Load and preprocess the image for detection
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

    # Get detections
    with torch.no_grad():
        detections = detection_model(image_tensor)

    # Draw bounding boxes and predictions
    plt.figure(figsize=(12, 12))
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image_np)
    
    for i in range(len(detections[0]['boxes'])):
        box = detections[0]['boxes'][i].cpu().numpy().astype(int)
        score = detections[0]['scores'][i].cpu().numpy()
        
        if score > 0.5:  # Threshold for displaying boxes
            xmin, ymin, xmax, ymax = box
            crop_img = image.crop((xmin, ymin, xmax, ymax))
            predicted_class = predict_image(crop_img, model)
            label = f'{class_names[predicted_class]}: {score:.2f}'

            # Draw rectangle
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(xmin, ymin, label, fontsize=15, color='white', backgroundcolor='red')

    ax.set_title(f'Predictions')
    plt.axis('off')
    if display == True:
        plt.show()
    print(label)
    print(class_names[predicted_class], f'{score:.2f}')
    return class_names[predicted_class], f'{score:.2f}'


def display_prediction_maian(image_path, model, detection_model,display):
    class_names = ['Drone', 'Not Drone']
    
    # Load and preprocess the image for detection
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

    # Get detections
    with torch.no_grad():
        detections = detection_model(image_tensor)

    # Draw bounding boxes and predictions
    plt.figure(figsize=(12, 12))
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image_np)
    
    for i in range(len(detections[0]['boxes'])):
        box = detections[0]['boxes'][i].cpu().numpy().astype(int)
        score = detections[0]['scores'][i].cpu().numpy()
        
        if score > 0.5:  # Threshold for displaying boxes
            xmin, ymin, xmax, ymax = box
            crop_img = image.crop((xmin, ymin, xmax, ymax))
            predicted_class = predict_image(crop_img, model)
            label = f'{class_names[predicted_class]}: {score:.2f}'

            # Draw rectangle
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(xmin, ymin, label, fontsize=15, color='white', backgroundcolor='red')

    ax.set_title(f'Predictions')
    plt.axis('off')
    plt.show()






def display_prediction_image(image_path, model, detection_model, display):
    class_names = ['Drone', 'Not Drone']

    # Load and preprocess the image for detection
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

    # Get detections
    with torch.no_grad():
        detections = detection_model(image_tensor)

    # Draw bounding boxes and predictions
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image_np)
    
    for i in range(len(detections[0]['boxes'])):
        box = detections[0]['boxes'][i].cpu().numpy().astype(int)
        score = detections[0]['scores'][i].cpu().numpy()
        
        if score > 0.5:  # Threshold for displaying boxes
            xmin, ymin, xmax, ymax = box
            crop_img = image.crop((xmin, ymin, xmax, ymax))
            predicted_class = predict_image(crop_img, model)
            label = f'{class_names[predicted_class]}: {score:.2f}'

            # Draw rectangle
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(xmin, ymin, label, fontsize=15, color='white', backgroundcolor='red')

    ax.set_title(f'Predictions')
    plt.axis('off')
    img_bytes = BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    return img_bytes

#--------------------------------------End of Displaying results-------------------------------------









#---------------------------------------Flask Endpoints---------------------------------------
class prediction_only(Resource):
    def post(self):
        if 'file' not in request.files:
            return {"message": "No file part"}, 400

        file = request.files['file']
        if file.filename == '':
            return {"message": "No selected file"}, 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the image and get predictions
        class_name, confidence = display_prediction(filepath, model, detection_model, False)

        return {"prediction": class_name, "confidence": confidence}


class prediction_with_images(Resource):
    def post(self):
        if 'file' not in request.files:
            return {"message": "No file part"}, 400

        file = request.files['file']
        if file.filename == '':
            return {"message": "No selected file"}, 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get image with annotations
        img_bytes = display_prediction_image(filepath, model, detection_model, False)

        return send_file(img_bytes, mimetype='image/png', as_attachment=True, download_name='prediction.png')
        
    
api.add_resource(prediction_only, "/predictionNumbers")
api.add_resource(prediction_with_images, "/predictionImages")
#--------------------------------------- End of Flask Endpoints---------------------------------------


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, threaded=True)


