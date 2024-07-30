import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, models
from PIL import Image
import tkinter
from tkinter import filedialog

"""
Use this file for live guessing.




"""



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
model.load_state_dict(torch.load('best_model_weights.pth'))
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

# Function to make a prediction on a single image
def predict_image(image_path, model):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Make the prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()

# Function to display the image with the prediction
def display_prediction(image_path, model):
    class_names = ['Drone', 'Not Drone']
    predicted_class = predict_image(image_path, model)
    image = Image.open(image_path).convert("RGB")

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(f'Prediction: {class_names[predicted_class]}')
    plt.axis('off')
    plt.show()



if __name__ == "__main__":
    while(True):
        tkinter.Tk().withdraw()
        file_path = tkinter.filedialog.askopenfilename()
        # Example usage: Display prediction for a single image
        image_path = file_path  # Replace with your image path
        display_prediction(image_path, model)
        
