import liveguesser
import tkinter
from tkinter import filedialog
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import tkinter
from tkinter import filedialog
import os 

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


base_dir = os.path.abspath(os.path.dirname(__file__))
path_weights = os.path.join(base_dir, 'best_drone_model_weights.pth')
model = DroneClassifier(num_classes=2)
model.load_state_dict(torch.load(path_weights))
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




if __name__ == "__main__":
     while(True):
            tkinter.Tk().withdraw()
            file_path = tkinter.filedialog.askopenfilename()
            image_path = file_path
            liveguesser.display_prediction_maian(image_path, model, detection_model,True)
