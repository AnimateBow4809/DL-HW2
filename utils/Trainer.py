import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchinfo import summary

from utils.visualization import plot_learning_curves, plot_confusion_matrix, plot_image_predictions


class Trainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        summary(model)

    def train(self, train_loader: DataLoader, val_loader: DataLoader = None, epochs=50,
              use_early_stopping: bool = False, patience: int = 3, min_delta: float = 1e-4, device="cpu"):
        num_train_samples = len(train_loader)
        best_val_loss = float('inf')
        epochs_no_improve = 0
        for epoch in range(epochs):
            epoch_train_loss = 0
            epoch_train_correct = 0
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")

            for x_batch, y_batch in train_pbar:
                output = self.model(x_batch)
                loss = self.loss_fn(output, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                with torch.no_grad():
                    epoch_train_loss += loss * len(x_batch)
                    preds = np.argmax(output, axis=1)
                    epoch_train_correct += np.sum(preds == y_batch)

            avg_train_loss = epoch_train_loss / num_train_samples
            avg_train_acc = epoch_train_correct / num_train_samples
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_acc'].append(avg_train_acc)

            if val_loader:
                val_loss, val_acc = self.evaluate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                print(
                    f"Epoch {epoch + 1}/{self.epochs} | Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                if use_early_stopping:
                    if val_loss < best_val_loss - min_delta:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        print(f"EarlyStopping counter: {epochs_no_improve} out of {patience}")
                        if epochs_no_improve >= patience:
                            print(f"\nEarly stopping triggered at epoch {epoch + 1}. Training halted.")
                            break
            else:
                print(
                    f"Epoch {epoch + 1}/{self.epochs} | Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
                if use_early_stopping:
                    print(
                        "Warning: Early stopping is enabled but no validation loader was provided. It will be ignored.")

    def evaluate(self, dataLoader: DataLoader):
        num_samples = len(dataLoader)
        total_loss = 0
        total_correct = 0
        eval_pbar = tqdm(dataLoader, desc="[Evaluate]", leave=False)

        for x_batch, y_batch in eval_pbar:
            output = x_batch
            for layer in self.layers:
                output = layer.forward(output)

            total_loss += self.loss_fn.forward(output, y_batch) * len(x_batch)
            preds = np.argmax(self.loss_fn.y_pred, axis=1)
            total_correct += np.sum(preds == y_batch)

        return total_loss / num_samples, total_correct / num_samples

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        print(f"Model loaded from {filepath}")

    def plot_misclassified_predictions(self, dataLoader: DataLoader, num_images: int = 5,
                                       image_shape: tuple = (28, 28)):
        misclassified_imgs = []
        misclassified_trues = []
        misclassified_preds = []

        for x_batch, y_batch in dataLoader:
            output = self.model(x_batch)
            preds = np.argmax(output, axis=1)
            wrong_indices = np.where(preds != y_batch)[0]
            for idx in wrong_indices:
                misclassified_imgs.append(x_batch[idx])
                misclassified_trues.append(y_batch[idx])
                misclassified_preds.append(preds[idx])
                if len(misclassified_imgs) == num_images:
                    break

            if len(misclassified_imgs) == num_images:
                break
        if len(misclassified_imgs) == 0:
            print("No misclassified images found! The model is 100% accurate on this data.")
            return
        plot_image_predictions(misclassified_imgs, misclassified_trues, misclassified_preds, image_shape)

    def get_per_class_accuracy(self, dataLoader: DataLoader):
        all_labels, all_preds = self._predict(dataLoader)
        per_class_accuracy = {}
        for i in range(10):
            class_mask = (all_labels == i)
            total_samples = np.sum(class_mask)
            if total_samples > 0:
                correct_predictions = np.sum((all_preds == i) & class_mask)
                accuracy = correct_predictions / total_samples
            else:
                accuracy = 0.0
            per_class_accuracy[i] = {"accuracy": accuracy, "total_samples": total_samples}
        return per_class_accuracy

    def plot_learning_curves(self):
        plot_learning_curves(self.history)

    def plot_confusion_matrix(self, dataLoader: DataLoader):
        y_true, y_pred = self._predict(dataLoader)
        plot_confusion_matrix(y_true, y_pred, class_names=[i for i in range(10)])

    def _predict(self, dataLoader: DataLoader):
        all_preds = []
        all_labels = []

        eval_pbar = tqdm(dataLoader, desc="[Predicting]", leave=False)
        for x_batch, y_batch in eval_pbar:
            output = self.model(x_batch)
            preds = np.argmax(output, axis=1)
            all_preds.extend(preds)
            all_labels.extend(y_batch)

        return np.array(all_labels), np.array(all_preds)
