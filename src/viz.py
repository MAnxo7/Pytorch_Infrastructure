import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import csv
import os 

def plot_from_csv(csv_path,step_mode=False):
    x_train, x_val, train_loss, val_loss, train_acc, val_acc = [], [] ,[], [], [], []
    savepath = os.path.split(csv_path)[0]
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = int(row['step'] if step_mode else row['epoch'])
            if row['split'] == 'train':
                x_train.append(x)
                train_loss.append(float(row['loss']))
                train_acc.append(float(row['acc']))
            elif row['split'] == 'eval':
                x_val.append(x)
                val_loss.append(float(row['loss']))
                val_acc.append(float(row['acc']))
    visualize_loss(x_train,x_val, train_loss, val_loss, save_path=os.path.join(savepath,'figures','loss.jpg'), step_mode=step_mode)
    visualize_acc(x_train,x_val, train_acc, val_acc, save_path=os.path.join(savepath,'figures','acc.jpg'), step_mode=step_mode)
    
def visualize_loss(x_train,x_val,train_loss, val_loss, save_path=None, step_mode=False):
    
    plt.figure(figsize=(7, 4))
    plt.plot(x_train,train_loss, label="Train Loss", linewidth=2)
    plt.plot(x_val,val_loss, label="Validation Loss", linewidth=2, linestyle="--")
    plt.xlabel("Step") if step_mode else plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    if(save_path):
        plt.savefig(save_path)
    plt.show()


def visualize_acc(x_train,x_val,train_acc, val_acc, save_path=None, step_mode=False):
    plt.figure(figsize=(7, 4))
    plt.plot(x_train,train_acc, label="Train Accuracy", linewidth=2)
    plt.plot(x_val,val_acc, label="Validation Accuracy", linewidth=2, linestyle="--")
    plt.xlabel("Step") if step_mode else plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    if(save_path):
        plt.savefig(save_path)
    plt.show()
