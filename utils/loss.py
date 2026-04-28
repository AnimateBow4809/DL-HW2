from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):

    @abstractmethod
    def forward(self, y_pred, y_true):
        pass

    @abstractmethod
    def backward(self):
        pass


class SoftmaxCrossEntropyLoss(Loss):
    def __init__(self):
        self.y_pred = None
        self.y_true = None
        self.input_shape = None

    def forward(self, logits, y_true):
        self.input_shape = logits.shape
        self.y_true = y_true

        logits_stable = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(logits_stable)
        self.y_pred = exps / np.sum(exps, axis=1, keepdims=True)

        num_samples = self.input_shape[0]

        y_pred_clipped = np.clip(self.y_pred, 1e-15, 1 - 1e-15)

        correct_class_probs = y_pred_clipped[np.arange(num_samples), y_true]

        sample_losses = -np.log(correct_class_probs)

        return np.mean(sample_losses)

    def backward(self):
        num_samples, num_classes = self.input_shape
        y_true_one_hot = np.zeros_like(self.y_pred)
        y_true_one_hot[np.arange(num_samples), self.y_true] = 1
        gradient = self.y_pred - y_true_one_hot
        gradient = gradient / num_samples
        return gradient



class CrossEntropyLoss(Loss):
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        self.y_true = y_true
        num_samples = y_pred.shape[0]
        self.y_pred = np.clip(y_pred, 1e-12, 1.0 - 1e-12)
        correct_class_probs = self.y_pred[np.arange(num_samples), y_true]
        sample_losses = -np.log(correct_class_probs)
        return np.mean(sample_losses)

    def backward(self):
        num_samples = self.y_pred.shape[0]

        y_true_one_hot = np.zeros_like(self.y_pred)
        y_true_one_hot[np.arange(num_samples), self.y_true] = 1.0

        gradient = -y_true_one_hot / self.y_pred

        gradient = gradient / num_samples

        return gradient
