
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from xcit.xcit import XCiT
import numpy as np
import os
import time

class ModelTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, criterion, optimizer, device):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_accuracy = 0
        self.no_improvement_count = 0
        self.early_stopping_patience = 5
        self.checkpoint_path = 'best_model_checkpoint.pth'

    def train_model(self):
        self.model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        progress_bar = tqdm(self.train_dataloader, desc='Training', position=0, leave=True)

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(self.device, dtype=torch.float32), labels.to(self.device, dtype=torch.long)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            if isinstance(self.model, XCiT):
             outputs = outputs.squeeze(0)

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            progress_bar.set_postfix({
                'loss': f'{running_loss / (progress_bar.n + 1):.4f}',
                'accuracy': f'{100 * total_correct / total_samples:.2f}%'
            })
        progress_bar.close()
        epoch_loss = running_loss / len(self.train_dataloader)
        epoch_acc = 100 * total_correct / total_samples
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)
        return epoch_loss, epoch_acc


    def validate_model(self):
        self.model.eval()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        progress_bar = tqdm(self.val_dataloader, desc='Validating', position=0, leave=False)

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(self.device, dtype=torch.float32), labels.to(self.device, dtype=torch.long)

            outputs = self.model(inputs)


            if isinstance(self.model, XCiT):
                outputs = outputs.squeeze(0)


            loss = self.criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            progress_bar.set_postfix({
                'val_loss': f'{running_loss / (progress_bar.n + 1):.4f}',
                'val_accuracy': f'{100 * total_correct / total_samples:.2f}%'
            })
        progress_bar.close()
        epoch_loss = running_loss / len(self.val_dataloader)
        epoch_acc = 100 * total_correct / total_samples
        self.val_losses.append(epoch_loss)
        self.val_accuracies.append(epoch_acc)
        return epoch_loss, epoch_acc


    def load_best_model(self):
        if os.path.exists(self.checkpoint_path):
            self.model.load_state_dict(torch.load(self.checkpoint_path))
            print(f"Loaded best model from {self.checkpoint_path}")


    def run_training_loop(self, num_epochs):
        for epoch in range(num_epochs):

            if epoch == 0 and os.path.exists(self.checkpoint_path):
              os.remove(self.checkpoint_path)

            print(f"Epoch {epoch + 1}/{num_epochs}")
            train_loss, train_acc = self.train_model()
            val_loss, val_acc = self.validate_model()

            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.no_improvement_count = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                self.no_improvement_count += 1

            print(f"Epoch {epoch + 1}: Train Loss {train_loss:.4f}, Acc {train_acc:.2f}%, Val Loss {val_loss:.4f}, Val Acc {val_acc:.2f}%")

            if self.no_improvement_count >= self.early_stopping_patience:
                print("Stopping early due to no improvement in validation accuracy.")
                break

        self.load_best_model()



def evaluate_model(model, dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    model.to(device)  # Move the model to the specified device
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_labels = []
    total_time = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            start_time = time.time()  # Start timing for performance

            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(inputs)  # Compute model outputs
            if isinstance(model, XCiT):
                outputs = outputs.squeeze(0)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)  # Determine predicted classes
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            all_predictions.extend(predicted.cpu().numpy())  # Save predictions for further analysis
            all_labels.extend(labels.cpu().numpy())

            batch_time = time.time() - start_time  # Calculate time for this batch
            total_time += batch_time  # Accumulate total evaluation time

    average_batch_time = total_time / len(dataloader)  # Average time per batch
    epoch_loss = running_loss / len(dataloader)  # Average loss per batch
    epoch_accuracy = 100 * total_correct / total_samples  # Calculate accuracy in percent

    # Logging results
    print(f"Validation Loss: {epoch_loss:.4f}")
    print(f"Validation Accuracy: {epoch_accuracy:.2f}%")
    print(f"Average Inference Time per Batch: {average_batch_time:.4f} seconds")

    # Returning more detailed results for further analysis
    return {
        'loss': epoch_loss,
        'accuracy': epoch_accuracy,
        'all_labels': np.array(all_labels),
        'all_predictions': np.array(all_predictions),
        'average_batch_time': average_batch_time
    }



def plot_metrics(trainer, labels, predictions):
    # Plotting training and validation losses and accuracies
    epochs = range(1, len(trainer.train_losses) + 1)
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, trainer.train_losses, label='Train Loss')
    plt.plot(epochs, trainer.val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, trainer.train_accuracies, label='Train Accuracy')
    plt.plot(epochs, trainer.val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()

def plot_conf_matrix(labels, predictions, class_names):
    # Plot confusion matrix with class names
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(15,15))


    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title('Confusion Matrix')
    plt.show()


def evaluate_pretrained(model, dataloader, device, class_list):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device).float()
            labels = labels.to(device)
            outputs = model(data)
            if isinstance(model, XCiT):
                outputs = outputs.squeeze(0)
            _, predicted = torch.max(outputs.data, 1)
            print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predictions.extend(predicted.view(-1).cpu().numpy())
            all_labels.extend(labels.view(-1).cpu().numpy())

    accuracy = correct / total

    cm = confusion_matrix(all_labels, all_predictions)

    plt.figure(figsize=(16, 9))
    ax = plt.gca()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(cax)

    ax.set_xticks(np.arange(len(class_list)))
    ax.set_yticks(np.arange(len(class_list)))
    ax.set_xticklabels(class_list, rotation=90, fontsize=8)
    ax.set_yticklabels(class_list, fontsize=8)

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix\nAccuracy: {:.2f}%'.format(accuracy * 100))
    plt.show()

    return all_predictions, all_labels


def evaluate_family_accuracy(all_predictions, all_labels, class_list, class_family_dict):
    # Initialize counters for correct predictions and totals per family
    family_correct = {}
    family_total = {}

    # Set up the data structure for families
    for family in set(class_family_dict.values()):
        family_correct[family] = 0
        family_total[family] = 0

    # Count correct predictions and total predictions per family
    for pred, true in zip(all_predictions, all_labels):
        pred_family = class_family_dict[class_list[pred]]
        true_family = class_family_dict[class_list[true]]

        if pred_family == true_family:
            family_total[true_family] += 1
            if pred == true:
                family_correct[true_family] += 1

    # Calculate accuracy for each family
    family_accuracy = {}
    for family in family_correct:
        if family_total[family] > 0:
            family_accuracy[family] = (family_correct[family] / family_total[family]) * 100
        else:
            family_accuracy[family] = 0

    # Sort families by accuracy in descending order
    sorted_families = sorted(family_accuracy.items(), key=lambda x: x[1], reverse=True)
    sorted_family_names = [item[0] for item in sorted_families]
    sorted_accuracies = [item[1] for item in sorted_families]

    # Plotting the results
    plt.figure(figsize=(12, 6))
    bars = plt.bar(sorted_family_names, sorted_accuracies, color=np.random.rand(len(sorted_family_names), 3))
    plt.xlabel('Family')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Within Family Classes')
    plt.ylim([0, 100])
    plt.xticks(rotation=45)

    # Add labels above bars
    for bar, accuracy in zip(bars, sorted_accuracies):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{int(accuracy)}%', ha='center', va='bottom')

    plt.show()