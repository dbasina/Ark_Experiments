from utils import MetricLogger, ProgressLogger, metric_AUROC
from sklearn.metrics import accuracy_score
import time
import torch
import numpy as np
import sys
import copy
from tqdm import tqdm
# import wandb

def train_one_epoch(model, use_head_n, dataset, data_loader_train, device, criterion, optimizer, epoch, ema_mode, teacher, momentum_schedule, coef_schedule, it):
    batch_time = MetricLogger('Time', ':6.3f')
    losses_cls = MetricLogger('Loss_'+dataset+' cls', ':.4e')
    losses_mse = MetricLogger('Loss_'+dataset+' mse', ':.4e')
    progress = ProgressLogger(
        len(data_loader_train),
        [batch_time, losses_cls, losses_mse],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    MSE = torch.nn.MSELoss()
    # coefficient scheduler from  0 to 0.5 
    coff = coef_schedule[it]
    print("Teacher_Loss_Coef: ",coff)
    end = time.time()
    for i, (samples1, samples2, targets) in enumerate(data_loader_train):
        samples1, samples2, targets = samples1.float().to(device), samples2.float().to(device), targets.float().to(device)
        
        # Debug: Print shapes on first iteration
        if i == 0:
            print(f"##DEBUG## Training {dataset} with head {use_head_n}")
            print(f"##DEBUG## Target shape: {targets.shape}")

        feat_t, pred_t = teacher(samples2, use_head_n)
        feat_s, pred_s = model(samples1, use_head_n)

        # Debug: Print prediction shape on first iteration  
        if i == 0:
            print(f"##DEBUG## Prediction shape: {pred_s.shape}")
        loss_cls = criterion(pred_s, targets)
        loss_const = MSE(feat_s, feat_t)
        
        loss = (1-coff) * loss_cls + coff * loss_const

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_cls.update(loss_cls.item(), samples1.size(0))
        losses_mse.update(loss_const.item(), samples1.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            progress.display(i)

        if ema_mode == "iteration":
            ema_update_teacher(model, teacher, momentum_schedule, it)
            it += 1

    if ema_mode == "epoch":
        ema_update_teacher(model, teacher, momentum_schedule, it)
        it += 1

    # wandb.log({"train_loss_cls_{}".format(dataset): losses_cls.avg})
    # wandb.log({"train_loss_mse_{}".format(dataset): losses_mse.avg})


def ema_update_teacher(model, teacher, momentum_schedule, it):
    with torch.no_grad():
        m = momentum_schedule[it]  # momentum parameter
        for param_q, param_k in zip(model.parameters(), teacher.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)


def evaluate(model, use_head_n, data_loader_val, device, criterion, dataset):
    model.eval()

    with torch.no_grad():
        batch_time = MetricLogger('Time', ':6.3f')
        losses = MetricLogger('Loss', ':.4e')
        progress = ProgressLogger(
        len(data_loader_val),
        [batch_time, losses], prefix='Val_'+dataset+': ')

        end = time.time()
        for i, (samples, _, targets) in enumerate(data_loader_val):
            samples, targets = samples.float().to(device), targets.float().to(device)

            _, outputs = model(samples, use_head_n)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), samples.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                progress.display(i)

    return losses.avg


def test_classification(model, use_head_n, data_loader_test, device, multiclass = False): 
       
    model.eval()

    y_test = torch.FloatTensor().to(device)
    p_test = torch.FloatTensor().to(device)

    with torch.no_grad():
        for i, (samples, _, targets) in enumerate(tqdm(data_loader_test, disable= not sys.stdout.isatty())):
            targets = targets.cuda()
            y_test = torch.cat((y_test, targets), 0)

            if len(samples.size()) == 4:
                bs, c, h, w = samples.size()
                n_crops = 1
            elif len(samples.size()) == 5:
                bs, n_crops, c, h, w = samples.size()

            varInput = torch.autograd.Variable(samples.view(-1, c, h, w).to(device))

            _, out = model(varInput, use_head_n)
            if multiclass:
                out = torch.softmax(out,dim = 1)
            else:
                out = torch.sigmoid(out)
            outMean = out.view(bs, n_crops, -1).mean(1)
            p_test = torch.cat((p_test, outMean.data), 0)

    return y_test, p_test
    
def mid_epoch_eval(model,teacher, most_recent_dataset, dataset_list, data_loaders_list, device, epoch, datasets_config, mid_epoch_eval_file): 
    model.eval()
    teacher.eval()
    print ("## STAGE ## Performing mid-epoch evaluation...")
    with torch.no_grad():
        for i,dataset in enumerate(dataset_list):
            data_loader = data_loaders_list[i]
            diseases = datasets_config[dataset]['diseases']
            task_type = datasets_config[dataset]['task_type']
            
            if task_type == "multi-class classification":
                multiclass = True
            else:
                multiclass = False

            y_test_student, p_test_student = test_classification(model, i, data_loader, device, multiclass)
            y_test_teacher, p_test_teacher = test_classification(teacher, i, data_loader, device, multiclass)

            if dataset == "CheXpert":
                test_diseases_name = datasets_config['CheXpert']['test_diseases_name']
                test_diseases = [diseases.index(c) for c in test_diseases_name]

                y_test_student = copy.deepcopy(y_test_student[:,test_diseases])
                p_test_student = copy.deepcopy(p_test_student[:, test_diseases])
                individual_results = metric_AUROC(y_test_student, p_test_student, len(test_diseases))

                y_test_teacher = copy.deepcopy(y_test_teacher[:,test_diseases])
                p_test_teacher = copy.deepcopy(p_test_teacher[:, test_diseases])
                individual_results_teacher = metric_AUROC(y_test_teacher, p_test_teacher, len(test_diseases)) 
            else: 
                individual_results_student = metric_AUROC(y_test_student, p_test_student, len(diseases))
                individual_results_teacher = metric_AUROC(y_test_teacher, p_test_teacher, len(diseases))

            # Compute AUC for all task types using one-vs-rest approach
            
            # individual_results_student = metric_AUROC(y_test_student, p_test_student, len(diseases))
            # individual_results_teacher = metric_AUROC(y_test_teacher, p_test_teacher, len(diseases))
            mean_auroc_student = np.nanmean(individual_results_student)
            mean_auroc_teacher = np.nanmean(individual_results_teacher)
            result_str = f"Epoch {epoch}, Recent Dataset {most_recent_dataset}, Test Dataset {dataset} ({task_type}): Student AUC: {mean_auroc_student:.4f}, Teacher AUC: {mean_auroc_teacher:.4f}"
            
            with open(mid_epoch_eval_file, "a") as f:
                f.write(result_str + "\n")
    print("## STAGE ## Mid-epoch evaluation complete. Results appended to", mid_epoch_eval_file)