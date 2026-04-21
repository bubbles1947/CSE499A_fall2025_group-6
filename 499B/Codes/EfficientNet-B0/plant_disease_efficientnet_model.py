import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from PIL import Image
from pathlib import Path
import json
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from datetime import datetime
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for EfficientNet plant disease detection"""
    
    # Dataset
    DATASET_PATH = "./plantvillage_dataset"
    NUM_CLASSES = 38
    IMG_SIZE = 256
    BATCH_SIZE = 128
    NUM_WORKERS = 4
    
    # Model
    MODEL_NAME = "efficientnet_b0"
    PRETRAINED = True
    
    # Training
    EPOCHS = 60
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 5
    EARLY_STOPPING_PATIENCE = 12
    
    # Quantization
    QUANTIZATION_TYPE = "INT8"
    
    # Paths
    OUTPUT_DIR = "./outputs_efficientnet"
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "efficientnet_model.pt")
    ONNX_MODEL_PATH = os.path.join(OUTPUT_DIR, "efficientnet_model.onnx")
    ONNX_QUANTIZED_PATH = os.path.join(OUTPUT_DIR, "efficientnet_model_int8.onnx")
    METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __init__(self):
        Path(self.OUTPUT_DIR).mkdir(exist_ok=True)


# ============================================================================
# DATASET LOADER (Same as before)
# ============================================================================

class PlantVillageDataset(Dataset):
    """Custom Dataset for PlantVillage"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_plantvillage_dataset(dataset_path, config):
    """Load PlantVillage dataset"""
    
    print("[INFO] Loading PlantVillage dataset...")
    
    classes = sorted([d for d in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, d))])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    
    print(f"[INFO] Found {len(classes)} classes")
    
    image_paths = []
    labels = []
    
    for class_name, class_idx in class_to_idx.items():
        class_dir = os.path.join(dataset_path, class_name)
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(class_dir, img_name))
                labels.append(class_idx)
    
    print(f"[INFO] Loaded {len(image_paths)} images")
    
    # Split
    np.random.seed(42)
    indices = np.random.permutation(len(image_paths))
    
    train_idx = int(0.7 * len(indices))
    val_idx = int(0.85 * len(indices))
    
    train_indices = indices[:train_idx]
    val_indices = indices[train_idx:val_idx]
    test_indices = indices[val_idx:]
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    train_dataset = PlantVillageDataset(
        [image_paths[i] for i in train_indices],
        [labels[i] for i in train_indices],
        transform=train_transform
    )
    
    val_dataset = PlantVillageDataset(
        [image_paths[i] for i in val_indices],
        [labels[i] for i in val_indices],
        transform=val_transform
    )
    
    test_dataset = PlantVillageDataset(
        [image_paths[i] for i in test_indices],
        [labels[i] for i in test_indices],
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    return train_loader, val_loader, test_loader, class_to_idx


# ============================================================================
# EFFICIENTNET MODEL
# ============================================================================

class EfficientNetPlantClassifier(nn.Module):
    """EfficientNet-B0 for plant disease classification"""
    
    def __init__(self, num_classes, config):
        super().__init__()
        
        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(pretrained=config.PRETRAINED)
        
        # Get number of input features for the classifier
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier with custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    """Trainer class"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.DEVICE
        self.model.to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        self.scheduler = self._get_scheduler()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'best_val_acc': 0,
            'early_stop_counter': 0
        }
    
    def _get_scheduler(self):
        """Learning rate scheduler with warmup"""
        def lr_lambda(epoch):
            if epoch < self.config.WARMUP_EPOCHS:
                return float(epoch) / float(max(1, self.config.WARMUP_EPOCHS))
            return 0.5 * (1.0 + np.cos(np.pi * (epoch - self.config.WARMUP_EPOCHS) / 
                                      (self.config.EPOCHS - self.config.WARMUP_EPOCHS)))
        
        from torch.optim.lr_scheduler import LambdaLR
        return LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': total_loss / (total + 1), 'acc': 100 * correct / total})
        
        return total_loss / len(train_loader), 100 * correct / total
    
    def validate(self, val_loader):
        """Validate"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating", leave=False)
            
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({'loss': total_loss / (total + 1), 'acc': 100 * correct / total})
        
        return total_loss / len(val_loader), 100 * correct / total, all_preds, all_labels
    
    def train(self, train_loader, val_loader):
        """Full training loop"""
        print("[INFO] Starting training...")
        
        for epoch in range(self.config.EPOCHS):
            print(f"\n[Epoch {epoch+1}/{self.config.EPOCHS}]")
            
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, _, _ = self.validate(val_loader)
            
            self.scheduler.step()
            
            self.metrics['train_loss'].append(train_loss)
            self.metrics['val_loss'].append(val_loss)
            self.metrics['train_acc'].append(train_acc)
            self.metrics['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            if val_acc > self.metrics['best_val_acc']:
                self.metrics['best_val_acc'] = val_acc
                self.metrics['early_stop_counter'] = 0
                self._save_model()
                print(f"[INFO] Model saved! Best Val Acc: {val_acc:.2f}%")
            else:
                self.metrics['early_stop_counter'] += 1
                if self.metrics['early_stop_counter'] >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"[INFO] Early stopping at epoch {epoch+1}")
                    break
        
        print("[INFO] Training completed!")
    
    def _save_model(self):
        """Save model"""
        torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)
    
    def evaluate(self, test_loader, class_to_idx):
        """Evaluate on test set"""
        print("\n[INFO] Evaluating on test set...")
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        print(f"\n[TEST RESULTS]")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        test_metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.config.METRICS_PATH, 'w') as f:
            json.dump(test_metrics, f, indent=4)
        
        self._plot_confusion_matrix(all_labels, all_preds, class_to_idx)
        
        return test_metrics
    
    def _plot_confusion_matrix(self, all_labels, all_preds, class_to_idx):
        """Plot confusion matrix"""
        cm = confusion_matrix(all_labels, all_preds)
        
        plt.figure(figsize=(16, 14))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix - EfficientNet-B0')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.OUTPUT_DIR, 'confusion_matrix.png'), dpi=100)
        plt.close()
        
        print("[INFO] Confusion matrix saved!")


# ============================================================================
# EXPORT & QUANTIZATION
# ============================================================================

class ModelExporter:
    """Export and quantize model"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.DEVICE
    
    def export_to_onnx(self, dummy_input_shape=(1, 3, 256, 256)):
        """Export to ONNX"""
        print("\n[INFO] Exporting model to ONNX...")
        
        dummy_input = torch.randn(dummy_input_shape).to(self.device)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            self.config.ONNX_MODEL_PATH,
            input_names=['images'],
            output_names=['logits'],
            dynamic_axes={
                'images': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            },
            opset_version=14,
            do_constant_folding=True,
            verbose=False
        )
        
        print(f"[INFO] ONNX model saved to {self.config.ONNX_MODEL_PATH}")
        self._print_model_size(self.config.ONNX_MODEL_PATH)
    
    def quantize_model(self):
        """Quantize to INT8"""
        print("\n[INFO] Quantizing model to INT8...")
        
        try:
            quantize_dynamic(
                model_input=self.config.ONNX_MODEL_PATH,
                model_output=self.config.ONNX_QUANTIZED_PATH,
                weight_type=QuantType.QUInt8,
                optimize_model=True
            )
            
            print(f"[INFO] Quantized model saved to {self.config.ONNX_QUANTIZED_PATH}")
            
            original_size = os.path.getsize(self.config.ONNX_MODEL_PATH) / (1024 * 1024)
            quantized_size = os.path.getsize(self.config.ONNX_QUANTIZED_PATH) / (1024 * 1024)
            reduction = ((original_size - quantized_size) / original_size) * 100
            
            print(f"\n[QUANTIZATION RESULTS]")
            print(f"Original size  : {original_size:.2f} MB")
            print(f"Quantized size : {quantized_size:.2f} MB")
            print(f"Reduction      : {reduction:.1f}%")
            
        except Exception as e:
            print(f"[ERROR] Quantization failed: {e}")
    
    def _print_model_size(self, model_path):
        """Print model size"""
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Model size: {size_mb:.2f} MB")
    
    def verify_onnx_model(self):
        """Verify ONNX model"""
        print("\n[INFO] Verifying ONNX model...")
        
        try:
            model = onnx.load(self.config.ONNX_MODEL_PATH)
            onnx.checker.check_model(model)
            print("[INFO] ONNX model is valid!")
        except Exception as e:
            print(f"[ERROR] ONNX model verification failed: {e}")


# ============================================================================
# INFERENCE ENGINE
# ============================================================================

class InferenceEngine:
    """Inference using ONNX Runtime"""
    
    def __init__(self, onnx_model_path, class_to_idx, config):
        self.config = config
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        
        self.session = ort.InferenceSession(
            onnx_model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"[INFO] Inference engine loaded from {onnx_model_path}")
    
    def preprocess_image(self, image_path):
        """Preprocess image"""
        image = Image.open(image_path).convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((self.config.IMG_SIZE, self.config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        image_tensor = transform(image)
        return image_tensor.unsqueeze(0).numpy().astype(np.float32)
    
    def predict(self, image_path, top_k=5):
        """Make prediction"""
        import time
        
        input_data = self.preprocess_image(image_path)
        
        start_time = time.time()
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        latency = (time.time() - start_time) * 1000
        
        logits = outputs[0][0]
        probs = np.exp(logits) / np.sum(np.exp(logits))
        
        top_indices = np.argsort(probs)[::-1][:top_k]
        
        predictions = []
        for idx in top_indices:
            predictions.append({
                'class': self.idx_to_class[idx],
                'confidence': float(probs[idx]),
                'probability': float(probs[idx])
            })
        
        return {
            'top_predictions': predictions,
            'latency_ms': latency,
            'all_logits': logits.tolist()
        }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def _plot_training_history(metrics, config):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(metrics['train_loss'], label='Train Loss')
    ax1.plot(metrics['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss - EfficientNet-B0')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(metrics['train_acc'], label='Train Acc')
    ax2.plot(metrics['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training Accuracy - EfficientNet-B0')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'training_history.png'), dpi=100)
    plt.close()
    print(f"[INFO] Training history saved!")


def main():
    """Main pipeline"""
    
    print("=" * 80)
    print("PLANT DISEASE DETECTION - EFFICIENTNET-B0 WITH QUANTIZATION")
    print("=" * 80)
    
    config = Config()
    print(f"\n[CONFIG]")
    print(f"Device: {config.DEVICE}")
    print(f"Dataset Path: {config.DATASET_PATH}")
    print(f"Model: {config.MODEL_NAME}")
    print(f"Image Size: {config.IMG_SIZE}")
    print(f"Num Classes: {config.NUM_CLASSES}")
    
    # Load dataset
    print(f"\n[DATASET LOADING]")
    train_loader, val_loader, test_loader, class_to_idx = load_plantvillage_dataset(
        config.DATASET_PATH, 
        config
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print(f"\n[MODEL CREATION]")
    model = EfficientNetPlantClassifier(config.NUM_CLASSES, config)
    print(f"Model created: {config.MODEL_NAME}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    trainer = Trainer(model, config)
    trainer.train(train_loader, val_loader)
    
    # Evaluate
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    test_metrics = trainer.evaluate(test_loader, class_to_idx)
    
    # Export to ONNX
    exporter = ModelExporter(model, config)
    exporter.export_to_onnx()
    exporter.verify_onnx_model()
    
    # Quantize
    exporter.quantize_model()
    
    # Test inference
    print(f"\n[TESTING INFERENCE ENGINE]")
    engine = InferenceEngine(config.ONNX_QUANTIZED_PATH, class_to_idx, config)
    
    sample_image_path = list(Path(config.DATASET_PATH).glob('*/*.jpg'))[0]
    result = engine.predict(str(sample_image_path), top_k=5)
    
    print(f"\nSample Prediction:")
    print(f"Image: {sample_image_path.name}")
    for i, pred in enumerate(result['top_predictions'], 1):
        print(f"{i}. {pred['class']}: {pred['confidence']:.4f}")
    print(f"Latency: {result['latency_ms']:.1f}ms")
    
    # Plot training history
    _plot_training_history(trainer.metrics, config)
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()