import matplotlib
import matplotlib.pyplot as plt
import csv
import os 

def plot_from_csv(csv_path):
    epochs, train_loss, val_loss, train_acc, val_acc = [], [], [], [], []
    savepath = os.path.split(csv_path)[0]
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['split'] == 'train':
                train_loss.append(float(row['loss']))
                train_acc.append(float(row['acc']))
            elif row['split'] == 'eval':
                val_loss.append(float(row['loss']))
                val_acc.append(float(row['acc']))
                epochs.append(int(row['epoch']))
    visualize_loss(train_loss, val_loss, save_path=os.path.join(savepath,'figures','loss.jpg'))
    visualize_acc(train_acc, val_acc, save_path=os.path.join(savepath,'figures','acc.jpg'))
    
def visualize_loss(train_loss, val_loss, save_path=None):
    plt.figure(figsize=(7, 4))
    plt.plot(train_loss, label="Train Loss", linewidth=2)
    plt.plot(val_loss, label="Validation Loss", linewidth=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    if(save_path):
        plt.savefig(save_path)
    plt.show()


def visualize_acc(train_acc, val_acc, save_path=None):
    plt.figure(figsize=(7, 4))
    plt.plot(train_acc, label="Train Accuracy", linewidth=2)
    plt.plot(val_acc, label="Validation Accuracy", linewidth=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    if(save_path):
        plt.savefig(save_path)
    plt.show()
