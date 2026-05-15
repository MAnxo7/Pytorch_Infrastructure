import torch
import os,csv,datetime, time
from . import utils,viz
#CSV: epoch, split, loss, acc, lr, time.
class warmup():
    def __init__(self,opt,lr_target,warmup_steps):
        self.lr_target = lr_target
        self.warmup_steps = warmup_steps
        self.opt = opt
        self.act_steps = 0
    
    def is_finished(self):
        return self.act_steps >= self.warmup_steps
    
    def step(self):
        if(self.is_finished()):
            raise RuntimeError("Step try after the warmup is finished")
        self.opt.param_groups[0]['lr'] = self.lr_target * ((self.act_steps+1) / self.warmup_steps)
        self.act_steps+=1 
        
# -------------------------- TRAIN WITH STEPS -----------------------------------------

def fit(
    model: torch.nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    max_steps: int | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    warmuper: warmup | None = None,
    early_stopping: int | None = None,
    run_dir: str = os.path.join(".", "runs"),
    ) -> None:
    """
    Train a model using a step-based training loop.

    This function trains the model using the provided training and validation
    dataloaders. It periodically evaluates the model, logs metrics to CSV, saves
    checkpoints, and generates loss/accuracy plots at the end of training.

    Training stops when one of the following conditions is met:
    - the maximum number of epochs is reached;
    - the maximum number of steps is reached;
    - early stopping patience is exceeded.

    Metrics are logged every fixed number of steps (N_STEPS). The reported training metrics
    are averaged over the steps since the previous evaluation point.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.

    device : torch.device
        Device where input batches are moved before the forward pass.

    train_loader : torch.utils.data.DataLoader
        Dataloader used for training batches.

    val_loader : torch.utils.data.DataLoader
        Dataloader used for validation/evaluation.

    optimizer : torch.optim.Optimizer
        Optimizer used to update the model parameters.

    loss_fn : torch.nn.Module
        Loss function used to compare model logits against target tokens.

    epochs : int
        Maximum number of epochs to train.

    max_steps : int or None, default=None
        Maximum number of optimizer steps. If None, training is only limited by
        the number of epochs.

    scheduler : torch.optim.lr_scheduler.LRScheduler or None, default=None
        Learning-rate scheduler stepped after each optimizer update, once warmup
        has finished.

    warmuper : warmup or None, default=None
        Optional warmup object used to increase the learning rate during the first
        training steps.

    early_stopping : int or None, default=None
        Number of consecutive evaluations without validation loss improvement
        before stopping training. If None, early stopping is disabled.

    run_dir : str, default="./runs"
        Directory where the run folder, metrics, checkpoints and plots are saved.

    Returns
    -------
    None
        The function trains the model in-place and saves artifacts to disk.

    Artifacts
    ---------
    Each run creates a timestamped directory containing:
    - metrics.csv
    - features.md
    - best.pt
    - last.pt
    - figures/loss.jpg
    - figures/acc.jpg
    """
    if epochs is None and max_steps is None:
        raise ValueError("You must specify either --epochs or --max-steps")
    if epochs and epochs <= 0:
        raise ValueError("Epochs can't be 0 or negative. Try increasing --epoch or using --eval-only")
    if max_steps and max_steps <= 0:
        raise ValueError("Max_steps can't be 0 or negative. Try increasing --steps or using --eval-only")
    N_STEPS = 1 # Each N_STEPS the model is evaluated and the metrics saved
    STEP_MODE = True # This makes the x-axis of the accuracy and loss graphics created by matplot be in range of N_STEPS instead of range of epochs
    act_step,act_epoch,last_improve= 0,0,0
    train_time = 0
    train_metrics = {"train_loss":0,"train_acc":0}
    steps_pre_eval = 0
    pre_eval_loss = None
    vpatience = early_stopping if early_stopping is not None else float("inf") 
    max_steps = max_steps if max_steps is not None else float("inf") 
    epochs = epochs if epochs is not None else float("inf")
    run_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    thisrun_path = os.path.join(run_dir,run_date)
    os.makedirs(thisrun_path,exist_ok=True)
    os.makedirs(os.path.join(thisrun_path,"figures"),exist_ok=True)
    csv_path = os.path.join(thisrun_path,"metrics.csv")
    features_path = os.path.join(thisrun_path,"features.md")
    last_ckpt_path = os.path.join(thisrun_path,"last.pt")
    best_ckpt_path = os.path.join(thisrun_path,"best.pt")
    best_eval_loss, best_eval_acc, best_train_loss, best_train_acc = float("inf"), 0.0, float("inf"), 0.0
    epoch_time_list = []
    # Features 
    utils.create_run_features(model,features_path,run_date,optimizer.param_groups[0]['lr'],train_loader.batch_size,optimizer.param_groups[0]['weight_decay'])
    # CSV Head
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["step","split","loss","acc","lr","duration_s"])
    # TRAIN
    while(act_epoch < epochs and act_step < max_steps and last_improve < vpatience):

        print("#### epoch nº ", act_epoch, " ####")
        #TRAIN
        for xn,yn in train_loader:
            t0 = time.time()
            train_metrics_act = train_one_step(model,xn,yn,optimizer,loss_fn,device,scheduler=scheduler, warmuper=warmuper)
            train_metrics["train_loss"] += train_metrics_act["train_loss"]
            train_metrics["train_acc"] += train_metrics_act["train_acc"]
            steps_pre_eval+=1
            train_time += time.time() - t0
            #EVALUATE 
            if act_step == 0 or act_step%N_STEPS == 0 or act_step >= max_steps - 1:

                print("-- step nº",act_step," --")
                t0 = time.time()
                train_metrics["train_loss"] /= steps_pre_eval
                train_metrics["train_acc"] /= steps_pre_eval
                eval_metrics = evaluate(model,val_loader,loss_fn,device)
                eval_time = time.time() - t0
                #SAVE EPOCH TIME
                epoch_time_list.append(train_time+eval_time)
                # INTER-EPOCH STATS
                print("TRAIN_LOSS: ",train_metrics["train_loss"]," TRAIN_ACC: ",train_metrics["train_acc"],
                "\nEVAL_LOSS: ",eval_metrics["eval_loss"]," EVAL_ACC: ",eval_metrics["eval_acc"],
                "\nLR: ",optimizer.param_groups[0]['lr'])
                # IS THE BEST?
                if (best_eval_loss > eval_metrics["eval_loss"]):
                    best_eval_loss = eval_metrics["eval_loss"]
                    best_train_loss = train_metrics["train_loss"]
                    best_loss_epoch = act_step
                if (best_eval_acc < eval_metrics["eval_acc"]):
                    best_eval_acc = eval_metrics["eval_acc"]
                    best_train_acc = train_metrics["train_acc"]
                    best_acc_epoch = act_step
                #SAVE DATA IN CSV
                with open(csv_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile,delimiter=",")
                    writer.writerow([act_step,"train",train_metrics["train_loss"],train_metrics["train_acc"],optimizer.param_groups[0]["lr"],train_time])
                    writer.writerow([act_step,"eval",eval_metrics["eval_loss"],eval_metrics["eval_acc"],optimizer.param_groups[0]["lr"],eval_time])
                train_time = 0
                train_metrics = {"train_loss": 0.0, "train_acc": 0.0}
                steps_pre_eval = 0
                #LROnPlateau logic, the only scheduler that works with loss
                if (act_step%N_STEPS == 0 and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and (warmuper is None or warmuper.is_finished())):
                    scheduler.step(eval_metrics["eval_loss"])
                #UPDATE LOOP
                if pre_eval_loss is not None and eval_metrics["eval_loss"] >= pre_eval_loss:
                    last_improve+=1
                else:
                    utils.save_checkpoint(model,optimizer,act_step,best_ckpt_path,steps_mode=STEP_MODE,extra=scheduler)
                    last_improve=0
                pre_eval_loss = eval_metrics["eval_loss"] 
                utils.save_checkpoint(model,optimizer,act_step,last_ckpt_path,steps_mode=STEP_MODE,extra=scheduler)
                if((max_steps and act_step >= max_steps - 1) or last_improve > vpatience):
                    act_step+=1
                    break
            act_step+=1
        act_epoch+=1           
    if last_improve >= vpatience:
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
     
    viz.plot_from_csv(csv_path,step_mode=STEP_MODE)
    
    gap_best_loss = best_eval_loss-best_train_loss
    gap_best_acc = best_train_acc-best_eval_acc
    avg_cycle_time = sum(epoch_time_list) / len(epoch_time_list)
    print("-----------------")
    print(f"Best eval loss: {best_eval_loss:.4f} | Best train loss: {best_train_loss:.4f} | GAP: {gap_best_loss:.4f} | Step: {best_loss_epoch}")
    print(f"Best eval acc : {best_eval_acc:.4f} | Best train acc : {best_train_acc:.4f} | GAP: {gap_best_acc:.4f} | Step: {best_acc_epoch}")
    print(f"Average epoch time: {avg_cycle_time:.4f}")



def train_one_step(model, xn, yn, optimizer,  loss_fn, device, scheduler = None, warmuper : warmup = None): 
    model.train()
    xn, yn = xn.to(device), yn.to(device)  
    optimizer.zero_grad()
    logits = model(xn)
    # Reshaping 
    logits = torch.reshape(logits,(-1,logits.shape[-1]))
    # Loss
    loss = loss_fn(logits,yn)
    loss.backward()
    #for name,param in model.named_parameters():
    #   print(name,param.grad.norm())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # Gradient Clipping      
    # OPTIMIZER, SCHEDULER AND WARMUP
    if warmuper is not None and not warmuper.is_finished():    
        warmuper.step()
        optimizer.step()
    elif scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        optimizer.step()
        scheduler.step()
    else:
        optimizer.step()
    #Metrics
    act_loss = loss.item()
    act_acc = utils.binary_accuracy_from_logits(logits, yn)
    return {"train_loss":act_loss,"train_acc":act_acc}


def evaluate(model,loader , loss_fn, device):
    model.eval()
    eval_loss,eval_acc,n_samples = 0.0,0.0,0
    with torch.no_grad():
        for xn,yn in loader:
            xn, yn = xn.to(device), yn.to(device)  
            logits = model(xn)
            # Reshaping 
            logits = torch.reshape(logits,(-1,logits.shape[-1]))
            #Loss
            loss = loss_fn(logits,yn)
            #Metrics
            samples = xn.size(0)
            eval_loss += loss.item()*samples
            eval_acc += utils.binary_accuracy_from_logits(logits, yn)*samples
            n_samples+=samples
    return {"eval_loss":eval_loss/n_samples,"eval_acc":eval_acc/n_samples}