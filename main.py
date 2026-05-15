import torch,argparse
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from src import utils,models,train

#ARGS 
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--max-steps", type=int, default=None)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--eval-only", action="store_true")
parser.add_argument("--eval-metrics", action="store_true")
parser.add_argument("--device", type=str, default=utils.get_device())
parser.add_argument("--ckpt-path", type=str, default=None)
parser.add_argument("--weight-decay", type=float, default=0)

args = parser.parse_args()

#BASIC CONFIG
utils.set_seed(0,deterministic=True)

ckpt_path = args.ckpt_path

epochs = args.epochs
batch = args.batch_size
max_steps = args.max_steps
lr = args.lr
weight_decay = args.weight_decay
device = args.device

# DATALOADERS CREATION
# Third position pattern
X_train = torch.tensor([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0]],dtype=torch.float32).to(device)
Y_train = torch.tensor([[0],[1],[0],[1],[0]],dtype=torch.float32).to(device)

X_eval = torch.tensor([[1,0,1],[1,1,0],[1,1,1]],dtype=torch.float32).to(device)
Y_eval = torch.tensor([[1],[0],[1]],dtype=torch.float32).to(device)

dataset_train = TensorDataset(X_train,Y_train)
dataset_eval = TensorDataset(X_eval,Y_eval)


dataloader_train = DataLoader(dataset_train,shuffle=False,batch_size=batch)
dataloader_eval = DataLoader(dataset_eval,shuffle=False,batch_size=batch)

#SPECS
if (ckpt_path is None):
    model = models.BasicNN(30,X_train.shape[1],1)
else:
    ckpt, model = utils.load_checkpoint(path=ckpt_path)
model = model.to(device)
opt = torch.optim.AdamW(params=model.parameters(),lr=lr,weight_decay=weight_decay)
if ckpt_path is not None and ckpt["optimizer"] is not None: opt.load_state_dict(ckpt["optimizer"])
loss_fn = torch.nn.BCEWithLogitsLoss().to(device)

#TRAIN OR EVAL
print(f"Used device: {device}")
if args.eval_only:
    if ckpt_path is None:
        raise ValueError("You should specify --ckpt-path when using --eval-only") 
    val_metrics = train.evaluate(model,dataloader_eval,loss_fn,device) 
    print("EVALUATION")
    print(f"Eval - Loss: {val_metrics['eval_loss']:.4f}, Acc: {val_metrics['eval_acc']:.4f}")
else:
    train.fit(model,device,dataloader_train,dataloader_eval,opt,loss_fn,epochs,max_steps=max_steps)


