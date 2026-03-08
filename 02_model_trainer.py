import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

# ================= 1. Dataset Preparation =================
data_dir = 'chess_dataset'

if not os.path.exists(data_dir):
    print(f"Error: Directory '{data_dir}' not found. Please verify the dataset path.")
    exit()

# Define image preprocessing pipeline (Resize to 224x224 and convert to tensor)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    # Apply random color and brightness jitter to enhance model robustness against lighting variations and glare
    transforms.ColorJitter(brightness=0.2, contrast=0.2), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the categorized image directory
dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

# Output the inferred class labels based on directory structure
class_names = dataset.classes
print(f"Initializing training for {len(class_names)} classes: \n{class_names}")

# ================= 2. Model Initialization (ResNet-18) =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUtilizing {device} for model training...")

# Load a pre-trained ResNet-18 model to leverage transfer learning
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Modify the final fully connected layer to output the specific number of target classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)

# Define the loss function and optimizer (Learning rate: 0.001)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ================= 3. Training Loop =================
num_epochs = 10  # Number of training iterations over the entire dataset

print("\nInitiating training process...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    corrects = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad() # Clear gradients from the previous step
        outputs = model(inputs) # Forward pass
        loss = criterion(outputs, labels) # Compute loss
        
        loss.backward() # Backpropagation
        optimizer.step() # Update model weights
        
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        corrects += torch.sum(preds == labels.data)
        total += inputs.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = corrects.double() / total
    
    print(f"Epoch {epoch+1}/{num_epochs} completed -> Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

# ================= 4. Model Serialization =================
save_path = 'chess_ai_model.pth'
torch.save(model.state_dict(), save_path)
print(f"\nTraining successfully concluded. Model weights serialized and saved to '{save_path}'")
print("The model is now ready for deployment in the automated extraction pipeline.")