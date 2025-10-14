import torch
import os,csv,datetime, time
from . import utils, viz
#CSV: epoch, split, loss, acc, lr, time.
def fit(model, device, train_loader, val_loader, optimizer: torch.optim.Optimizer, loss_fn, epochs, scheduler=None, early_stopping=None, run_dir=os.path.join(".","runs")):
    act_epoch,last_improve = 0,0
    pre_eval_loss = None
    vpatience = early_stopping if early_stopping is not None else float("inf") 
    run_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    thisrun_path = os.path.join(run_dir,run_date)
    os.makedirs(thisrun_path,exist_ok=True)
    csv_path = os.path.join(thisrun_path,"metrics.csv")
    last_ckpt_path = os.path.join(thisrun_path,"last.pt")
    best_ckpt_path = os.path.join(thisrun_path,"best.pt")
    print(thisrun_path, csv_path)
    # Cabecera CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["epoch","split","loss","acc","lr","duration_s"])
    while(act_epoch < epochs and last_improve < vpatience):
        #TRAIN
        t0 = time.time()
        train_metrics = train_one_epoch(model,train_loader,optimizer,loss_fn,device)
        train_time = time.time() - t0
        #EVALUATE
        t0 = time.time()
        eval_metrics = evaluate(model,val_loader,loss_fn,device)
        eval_time = time.time() - t0
        #SCHEDULER 
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(eval_metrics["eval_loss"])
            else:
                scheduler.step()
        #SAVE DATA IN CSV
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile,delimiter=",")
            writer.writerow([act_epoch,"train",train_metrics["train_loss"],train_metrics["train_acc"],optimizer.param_groups[0]["lr"],train_time])
            writer.writerow([act_epoch,"eval",eval_metrics["eval_loss"],eval_metrics["eval_acc"],optimizer.param_groups[0]["lr"],eval_time])
        #UPDATE LOOP
        if pre_eval_loss is not None and eval_metrics["eval_loss"] >= pre_eval_loss:
            last_improve+=1
        else:
            utils.save_checkpoint(model,optimizer,act_epoch,best_ckpt_path)
            last_improve=0
        pre_eval_loss = eval_metrics["eval_loss"]
        utils.save_checkpoint(model,optimizer,act_epoch,last_ckpt_path)
        act_epoch+=1
    if last_improve >= vpatience:
        utils.load_checkpoint(best_ckpt_path,model,optimizer)
        
    viz.plot_from_csv(csv_path)
    
        
        
          
        
        
        
        
def train_one_epoch(model, loader, optimizer, loss_fn, device): 
    model.train()
    train_loss,train_acc,n_samples = 0.0,0.0,0
    for xn,yn in loader:
        xn, yn = xn.to(device), yn.to(device)  
        optimizer.zero_grad()
        logits = model(xn)
        loss = loss_fn(logits,yn)
        loss.backward()
        optimizer.step()
        #Metrics
        samples = xn.size(0)
        train_loss += loss.item()*samples
        train_acc += utils.binary_accuracy_from_logits(logits, yn)*samples
        n_samples+=samples
    return {"train_loss":train_loss/n_samples,"train_acc":train_acc/n_samples}
        
        
def evaluate(model,loader, loss_fn, device):
    model.eval()
    eval_loss,eval_acc,n_samples = 0.0,0.0,0
    with torch.no_grad():
        for xn,yn in loader:
            xn, yn = xn.to(device), yn.to(device)  
            logits = model(xn)
            loss = loss_fn(logits,yn)
            #Metrics
            samples = xn.size(0)
            eval_loss += loss.item()*samples
            eval_acc += utils.binary_accuracy_from_logits(logits, yn)*samples
            n_samples+=samples
    return {"eval_loss":eval_loss/n_samples,"eval_acc":eval_acc/n_samples}