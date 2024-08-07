import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import tkinter
from tkinter import filedialog
import numpy as np

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

# Load the trained model weights
model = DroneClassifier(num_classes=2)
model.load_state_dict(torch.load('best_drone_model_weights.pth'))
model.eval()

# Set device
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

# Function to display the image with the prediction and bounding box
def display_prediction(image_path, model, detection_model):
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
    print(label)
    return label

if __name__ == "__main__":
    while(True):
        tkinter.Tk().withdraw()
        file_path = tkinter.filedialog.askopenfilename()
        # Example usage: Display prediction for a single image
        image_path = file_path  # Replace with your image path
        display_prediction(image_path, model, detection_model)
        
