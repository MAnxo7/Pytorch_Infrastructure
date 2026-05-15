import torch
from typing import Any

def set_seed(seed: int, deterministic: bool = False):
    import os, random, numpy as np
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
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def binary_accuracy_from_logits(logits, y_true, thr=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs>thr).int()
    correct = torch.sum((preds==y_true).int()).item()
    nelems = torch.numel(preds)
    return correct/nelems

def save_checkpoint(model : torch.nn.Module, optimizer : torch.optim.Optimizer, epoch_step : int, path : str, steps_mode: bool = False, extra: dict | None = None):
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "model_type": model.__class__.__name__,
        "model_config": model.get_config(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch_step": int(epoch_step),
        "steps_mode": bool(steps_mode),
        "extra": extra or {},
    }
    torch.save(payload, path)
    
def load_checkpoint(path : str, map_location: str="cpu") -> tuple[dict[str, Any], torch.nn.Module]:
    from src import models
    print("Loading checkpoint from:", path)

    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    # Checkpoint validation
    if "model_type" not in ckpt or ckpt["model_type"] is None:
        raise ValueError("Checkpoint has no 'model_type' or 'model_type' is None. Cannot reconstruct model automatically.")
    if "model_config" not in ckpt or ckpt["model_config"] is None:
        raise ValueError("Checkpoint has no 'model_config' or 'model_config' is None. Cannot reconstruct model automatically.")
    if "model" not in ckpt or ckpt["model"] is None:
        raise ValueError("Checkpoint has no 'model' state_dict or 'model' is None.")
    model_type = ckpt["model_type"]
    model_config = ckpt["model_config"]
    print(model_config)
    # Does the model exists?
    try:
        model_cls = getattr(models, model_type)
    except AttributeError as e:
        raise AttributeError(f"There is no model class named '{model_type}' in models.py.") from e
    # Are the model attributes correct?
    try:
        model = model_cls(**model_config)
    except TypeError as e:
        raise TypeError(f"The stored config for '{model_type}' does not match its constructor.") from e

    model.load_state_dict(ckpt["model"], strict=True)

    return ckpt, model

def create_run_features(model : torch.nn.Module, path: str, run_date : str = None ,lr : float = None , batch_size : int = None , wd : float = None):
    """Generates a features.file in the given path

    Parameters
    ----------
    model : torch.nn.Module
        The model that yo want to save its features.
    path : str
        The wished path for the file creation.
    run_date,lr,batch_size,wd : str,float,int,float
        The possible extra features to record.
    """
    model_dic = vars(model)
    with open(path, 'a') as featuresfile:
        featuresfile.write("## ---- MODEL ----\n")
        for key in model_dic.keys():
            value = model_dic[key]
            if (isinstance(value,bool)):
                continue
            if (isinstance(value,int) or isinstance(value,float) or isinstance(value,str)):
                txt = str(key) + ": " + str(value) + "\n" 
                featuresfile.write(txt)
        featuresfile.write("## ---- TRAINING ----\n")    
        featuresfile.write("date: " +  str(run_date) + "\n")
        featuresfile.write("lr: " +  str(lr) + "\n")
        featuresfile.write("batch_size: " +  str(batch_size) + "\n")
        featuresfile.write("weight_decay: " +  str(wd) + "\n")