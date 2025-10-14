def set_seed(seed: int, deterministic: bool = False):
    import os, random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)              # If there is GPU
    torch.cuda.manual_seed_all(seed)          # multi-GPU

    # 3) cuDNN flags (solo si tienes CUDA/cuDNN)
    if deterministic:
        torch.backends.cudnn.deterministic = True   # use determinist kernels
        torch.backends.cudnn.benchmark = False      
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True 
        
def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def binary_accuracy_from_logits(logits, y_true, thr=0.5):
    import torch
    probs = torch.sigmoid(logits)
    preds = (probs>thr).int()
    correct = torch.sum((preds==y_true).int()).item()
    nelems = torch.numel(preds)
    return correct/nelems

def save_checkpoint(model, optimizer, epoch, path, extra: dict | None = None):
    import torch, os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "extra": extra or {},
    }
    torch.save(payload, path)
    
def load_checkpoint(path, model=None, optimizer=None, map_location="cpu"):
    import torch
    ckpt = torch.load(path, map_location=map_location)
    if model is not None:
        model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt