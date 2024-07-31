import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import matplotlib.pyplot as plt

"""
Use this file for training model

"""


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

# Load datasets
train_dataset = CustomDataset(root_dir="C:\\Users\\lburns\Desktop\\Drone_Model\dataset\\train", transform=transform_train)
val_dataset = CustomDataset(root_dir="C:\\Users\\lburns\\Desktop\\Drone_Model\\dataset\\val", transform=transform_val)

# DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

# Define the model
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

# Initialize the model
model = DroneClassifier(num_classes=2)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training and validation
num_epochs = 5
best_val_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")
    
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
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
        torch.save(model.state_dict(), 'best_model_weights.pth')
        print(f"Model weights saved at epoch {epoch+1} with validation accuracy: {val_accuracy}%")
    # Step the learning rate scheduler
    scheduler.step()

# Function to display images with predictions
def display_images_with_predictions(model, data_loader):
    class_names = ['Drone', 'Not Drone']
    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            plt.figure(figsize=(10, 10))
            for i in range(len(images)):
                image = images[i].cpu().permute(1, 2, 0).numpy()
                image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # Unnormalize
                image = image.clip(0, 1)

                plt.subplot(4, 4, i + 1)
                plt.imshow(image)
                plt.title(f'Pred: {class_names[preds[i]]}\n(True: {class_names[labels[i]]})')
                plt.axis('off')

            plt.show()

            conv1_weights = model.model.conv1.weight.data.cpu().numpy()

            plt.figure(figsize=(15, 10))
            for i in range(10):
                plt.subplot(1, 10, i + 1)
                plt.imshow(conv1_weights[i, 0, :, :], cmap='gray')
                plt.axis('off')
            plt.show()
            break  # Display only the first batch for brevity

# Display images with predictions
display_images_with_predictions(model, val_loader)
