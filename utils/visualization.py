import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('default')


def plot_random_samples(x, y, samples_per_class=5):
    num_classes = 10
    fig, axes = plt.subplots(nrows=num_classes,
                             ncols=samples_per_class,
                             figsize=(samples_per_class * 1.2, num_classes * 1.2))

    for class_idx in range(num_classes):
        class_indices = np.where(y == class_idx)[0]
        selected_indices = np.random.choice(class_indices, samples_per_class, replace=False)

        for col_idx, img_idx in enumerate(selected_indices):
            ax = axes[class_idx, col_idx]

            ax.imshow(x[img_idx], cmap='gray', interpolation='nearest', vmin=0.0, vmax=1.0)

            ax.set_xticks([0, 14, 27])
            ax.set_yticks([0, 14, 27])
            ax.tick_params(axis='both', labelsize=7)

            if col_idx > 0:
                ax.set_yticklabels([])
            if class_idx < num_classes - 1:
                ax.set_xticklabels([])
            if col_idx == 0:
                ax.set_ylabel(f"Class {class_idx}", fontsize=10, weight='bold', labelpad=5)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout()
    plt.show()


def plot_class_distribution(y_train, y_val, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    datasets = [
        ("Training Set", y_train, "#4C72B0"),
        ("Validation Set", y_val, "#55A868"),
        ("Test Set", y_test, "#C44E52")
    ]

    classes = np.arange(10)

    for ax, (title, y_data, color) in zip(axes, datasets):
        counts = np.bincount(y_data, minlength=10)
        bars = ax.bar(classes, counts, color=color, edgecolor='black', alpha=0.8)
        ax.set_title(f"{title} (N={len(y_data)})", fontsize=12, weight='bold')
        ax.set_xlabel("Digit Class", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_xticks(classes)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        max_height = max(counts)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + (max_height * 0.02),
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=8, rotation=45)

    plt.tight_layout()
    plt.show()


def plot_learning_curves(history):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    if history['val_loss']:
        ax1.plot(epochs, history['val_loss'], label='Validation Loss', marker='o')
    ax1.set_title('Loss over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history['train_acc'], label='Train Accuracy', marker='o')
    if history['val_acc']:
        ax2.plot(epochs, history['val_acc'], label='Validation Accuracy', marker='o')
    ax2.set_title('Accuracy over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    num_classes = max(np.max(y_true), np.max(y_pred)) + 1
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names else 'auto',
                yticklabels=class_names if class_names else 'auto')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()


def plot_image_predictions(images, true_labels, pred_labels, image_shape=(28, 28)):
    num_images = len(images)
    if num_images == 0:
        print("No images to display.")
        return

    # Adjust figure size based on the academic layout proportions
    fig, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(num_images * 1.5, 2.0))
    if num_images == 1:
        axes = [axes]

    for i in range(num_images):
        ax = axes[i]
        img = images[i].reshape(image_shape)
        true_label = true_labels[i]
        pred_label = pred_labels[i]

        # Use nearest interpolation to match plot_random_samples
        ax.imshow(img, cmap='gray', interpolation='nearest')

        # Dynamically set ticks based on the image shape (e.g., [0, 14, 27] for 28x28)
        mid_y, mid_x = image_shape[0] // 2, image_shape[1] // 2
        ax.set_xticks([0, mid_x, image_shape[1] - 1])
        ax.set_yticks([0, mid_y, image_shape[0] - 1])
        ax.tick_params(axis='both', labelsize=7)

        # Remove y-axis tick labels for all but the first image to save space
        if i > 0:
            ax.set_yticklabels([])

        # Use the specific hex colors from your plot_class_distribution
        color = '#55A868' if true_label == pred_label else '#C44E52'

        ax.set_title(f"T:{true_label} | P:{pred_label}",
                     color=color,
                     fontsize=10,
                     weight='bold',
                     pad=8)

    plt.subplots_adjust(wspace=0.1)
    plt.tight_layout()
    plt.show()
