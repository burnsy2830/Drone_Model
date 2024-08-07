import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self._load_images_and_labels()

    def _load_images_and_labels(self):
        for label, class_dir in enumerate(['drones', 'not_drones']):
            class_path = os.path.join(self.root_dir, class_dir)
            if not os.path.isdir(class_path):
                raise FileNotFoundError(f"Directory '{class_path}' does not exist.")
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.images.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Data transformations
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])




working_directory = Path(os.path.realpath(sys.argv[0])).parent


train_path = working_directory / 'dataset' / 'train'
val_path = working_directory / 'dataset' / 'val'


print(f"Train path: {train_path}")
print(f"Validation path: {val_path}")

# Load datasets
train_dataset = CustomDataset(root_dir=train_path, transform=transform_train)
val_dataset = CustomDataset(root_dir=val_path, transform=transform_val)


train_loader = DataLoader(train_dataset, batch_size=9, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=9, shuffle=False)

# Define the drone classifier model
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

# Initialize the drone classifier model
drone_model = DroneClassifier(num_classes=2)

# Define the object detection model
detection_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 2
in_features = detection_model.roi_heads.box_predictor.cls_score.in_features


detection_model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Define loss function and optimizer for both models
criterion = nn.CrossEntropyLoss()
drone_optimizer = optim.AdamW(drone_model.parameters(), lr=0.0001)
drone_scheduler = optim.lr_scheduler.StepLR(drone_optimizer, step_size=7, gamma=0.1)

detection_optimizer = optim.AdamW(detection_model.parameters(), lr=0.0001)
detection_scheduler = optim.lr_scheduler.StepLR(detection_optimizer, step_size=7, gamma=0.1)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
drone_model.to(device)
detection_model.to(device)

# Training and validation
num_epochs = 5
best_val_accuracy = 0.0

for epoch in range(num_epochs):
    drone_model.train()
    detection_model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Train the drone classifier
        drone_optimizer.zero_grad()
        outputs = drone_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        drone_optimizer.step()
        
        running_loss += loss.item()

        # Train the detection model (assumes you have detection labels)
        detection_targets = []  # Add your target labels for object detection
        for i in range(images.size(0)):
            detection_targets.append({
                'boxes': torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32, device=device),
                'labels': torch.tensor([1], dtype=torch.int64, device=device)
            })
        detection_optimizer.zero_grad()
        detection_outputs = detection_model(images, detection_targets)
        detection_loss = sum(loss for loss in detection_outputs.values())
        detection_loss.backward()
        detection_optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")
    
    drone_model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = drone_model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_accuracy = 100 * correct / total
    print(f"Validation Loss: {val_loss/len(val_loader)}, Accuracy: {val_accuracy}%")
    
    # Save the model weights if the validation accuracy is improved
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(drone_model.state_dict(), 'best_drone_model_weights.pth')
        #torch.save(detection_model.state_dict(), 'best_detection_model_weights.pth')
        print(f"Model weights saved at epoch {epoch+1} with validation accuracy: {val_accuracy}%")
    
    # Step the learning rate scheduler
    drone_scheduler.step()
    detection_scheduler.step()

