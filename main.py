import os
import pandas as pd
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision.models import EfficientNet_B4_Weights
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns

# Caminhos do dataset
BASE_PATH = 'DDR-dataset/DR_grading'
TRAIN_PATH = os.path.join(BASE_PATH, 'train')
VALID_PATH = os.path.join(BASE_PATH, 'valid')
TEST_PATH = os.path.join(BASE_PATH, 'test')

# Função para carregar os labels dos arquivos .txt
def load_labels(file_path):
    labels = pd.read_csv(file_path, sep=' ', header=None, names=['filename', 'label'])
    return labels

train_labels = load_labels(os.path.join(BASE_PATH, 'train.txt'))
valid_labels = load_labels(os.path.join(BASE_PATH, 'valid.txt'))
test_labels = load_labels(os.path.join(BASE_PATH, 'test.txt'))

# Custom Dataset para carregar imagens e labels
class CustomDataset(Dataset):
    def __init__(self, labels, dir_path, transform=None):
        self.labels = labels
        self.dir_path = dir_path
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.dir_path, self.labels.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = int(self.labels.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        return image, label

# Definindo as transformações
train_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(384, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Criando datasets e dataloaders
BATCH_SIZE = 16  # Tamanho do batch

train_dataset = CustomDataset(train_labels, TRAIN_PATH, transform=train_transform)
valid_dataset = CustomDataset(valid_labels, VALID_PATH, transform=test_transform)
test_dataset = CustomDataset(test_labels, TEST_PATH, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Carregando o modelo EfficientNet-B4 pré-treinado
weights = EfficientNet_B4_Weights.DEFAULT
model = models.efficientnet_b4(weights=weights)
num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 6)
)

# Verificando se a GPU está disponível
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Definindo a função de perda e o otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Função de treino
def train_model(model, criterion, optimizer, train_loader, valid_loader, num_epochs=20):
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = valid_loader
                
            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []
            
            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            torch.save(model.state_dict(), f'efficientnet_b4_dr_model_epoch{epoch+1}.pth')
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                torch.save(best_model_wts, 'best_model.pth')
    
    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model

# Treinamento do modelo
model = train_model(model, criterion, optimizer, train_loader, valid_loader, num_epochs=20)

# Avaliação do modelo
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Função para calcular métricas
def calculate_metrics(loader, model, device):
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(6)], output_dict=True)
    kappa = cohen_kappa_score(all_labels, all_preds)
    
    per_class_accuracy = [report[str(i)]['precision'] for i in range(6)]
    mean_accuracy = sum(per_class_accuracy) / len(per_class_accuracy)

    return accuracy, mean_accuracy, kappa, classification_report(all_labels, all_preds, target_names=[str(i) for i in range(6)]), confusion_matrix(all_labels, all_preds)

# Função para plotar e salvar a matriz de confusão
def plot_confusion_matrix(cm, title, filename):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[str(i) for i in range(6)], yticklabels=[str(i) for i in range(6)])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.savefig(filename)
    plt.close()

# Calculando métricas para o conjunto de validação
print("Validation Set Metrics:")
valid_accuracy, valid_mean_accuracy, valid_kappa, valid_report, valid_conf_matrix = calculate_metrics(valid_loader, model, device)
print(f"Validation OA: {valid_accuracy:.4f}")
print(f"Validation AA: {valid_mean_accuracy:.4f}")
print(f"Validation Kappa: {valid_kappa:.4f}")
print("Validation Classification Report:")
print(valid_report)
print("Validation Confusion Matrix:")
print(valid_conf_matrix)

# Plotando e salvando a matriz de confusão para o conjunto de validação
plot_confusion_matrix(valid_conf_matrix, 'Validation Confusion Matrix', 'validation_confusion_matrix_efficientnetB4.png')

# Calculando métricas para o conjunto de teste
print("Test Set Metrics:")
test_accuracy, test_mean_accuracy, test_kappa, test_report, test_conf_matrix = calculate_metrics(test_loader, model, device)
print(f"Test OA: {test_accuracy:.4f}")
print(f"Test AA: {test_mean_accuracy:.4f}")
print(f"Test Kappa: {test_kappa:.4f}")
print("Test Classification Report:")
print(test_report)
print("Test Confusion Matrix:")
print(test_conf_matrix)

# Plotando e salvando a matriz de confusão para o conjunto de teste
plot_confusion_matrix(test_conf_matrix, 'Test Confusion Matrix', 'test_confusion_matrix_efficientnetB4.png')
