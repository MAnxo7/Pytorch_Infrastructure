import torch,argparse
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from src import utils,models,train

#ARGS 
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--eval-only", action="store_true")
parser.add_argument("--device", type=str, default=utils.get_device())
parser.add_argument("--ckpt-path", type=str, default=None)
args = parser.parse_args()

#BASIC CONFIG
utils.set_seed(0,deterministic=True)

epochs = args.epochs
batch = args.batch_size
lr = args.lr
device = args.device

# DATALOADERS CREATION
# First position pattern
X_train = torch.tensor([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0]],dtype=torch.float32).to(device)
Y_train = torch.tensor([[0],[1],[0],[1],[0]],dtype=torch.float32).to(device)

X_eval = torch.tensor([[1,0,1],[1,1,0],[1,1,1]],dtype=torch.float32).to(device)
Y_eval = torch.tensor([[1],[0],[1]],dtype=torch.float32).to(device)

dataset_train = TensorDataset(X_train,Y_train)
dataset_eval = TensorDataset(X_eval,Y_eval)


dataloader_train = DataLoader(dataset_train,shuffle=False,batch_size=batch)
dataloader_eval = DataLoader(dataset_eval,shuffle=False,batch_size=batch)

#SPECS
model = models.BasicNN(30,X_train.shape[1],1,device)

opt = torch.optim.Adam(params=model.parameters(),lr=lr)
loss_fn = torch.nn.BCEWithLogitsLoss().to(device)

#TRAIN OR EVAL
if args.eval_only:
    if args.ckpt_path is None:
        raise ValueError("You should specific --ckpt-path when use --eval-only")
    utils.load_checkpoint(args.ckpt_path,model,opt)
    val_metrics = train.evaluate(model,dataloader_eval,loss_fn,device)
    print(f"Eval - Loss: {val_metrics['eval_loss']:.4f}, Acc: {val_metrics['eval_acc']:.4f}")
    
else:
    train.fit(model,device,dataloader_train,dataloader_eval,opt,loss_fn,epochs,early_stopping=500)



