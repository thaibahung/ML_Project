import models
import time
import torch
import math
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, average_precision_score, roc_auc_score

def build_model(model_name):
   
    if model_name == 'XCLIP_Dualist':
        model = models.XCLIP_Dualist()
    if model_name == 'CLIP_Dualist':
        model = models.CLIP_Dualist()
    if model_name == 'ViT_B_MINTIME':
        model = models.ViT_B_MINTIME()
    return model

def eval_model(cfg, model, val_loader, loss_ce, val_batch_size):
    model.eval()
    outpred_list = []
    gt_label_list = []
    video_list = []
    valLoss = 0
    lossTrainNorm = 0
    print("******** Start Testing. ********")

    with torch.no_grad():  # No need to track gradients during validation
        for i, (_, input, target, binary_label, video_id) in enumerate(tqdm(val_loader, desc="Validation", total=len(val_loader))):
            if i == 0:
                ss_time = time.time()

            input = input[:,0]
            varInput = torch.autograd.Variable(input.float().cuda())
            varTarget = torch.autograd.Variable(target.contiguous().cuda())
            var_Binary_Target = torch.autograd.Variable(binary_label.contiguous().cuda())

            logit = model(varInput)
            lossvalue = loss_ce(logit, var_Binary_Target)

            valLoss += lossvalue.item()
            lossTrainNorm += 1
            outpred_list.append(logit[:,0].sigmoid().cpu().detach().numpy())
            gt_label_list.append(varTarget.cpu().detach().numpy())
            video_list.append(video_id)

    valLoss = valLoss / lossTrainNorm

    outpred = np.concatenate(outpred_list, 0)
    gt_label = np.concatenate(gt_label_list, 0)
    video_list = np.concatenate(video_list, 0)
    predictLabels = [1 if item > 0.5 else 0 for item in outpred]
    givenLabels = np.argmax(gt_label, axis=1)

    pred_accuracy = accuracy_score(givenLabels, predictLabels)
    pred_f1_score = f1_score(givenLabels, predictLabels, average='macro')

    return pred_accuracy, pred_f1_score, video_list, predictLabels, givenLabels, outpred

def sample_frames(video_path, interval):
    cap = 0
    sampled_frames = []
    frame_count = 0
    video_path = cap 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            sampled_frames.append(frame)
        frame_count += 1

    cap.release()
    return sampled_frames

def interpolate_frame_indices(frame_indices, num_interpolations):
    interpolated = []
    for i in range(len(frame_indices) - 1):
        start, end = frame_indices[i], frame_indices[i + 1]
        interpolated.append(start)
        step = (end - start) / (num_interpolations + 1)
        for j in range(1, num_interpolations + 1):
            interpolated.append(int(start + j * step))
    interpolated.append(frame_indices[-1])
    return interpolated


def train_one_epoch(cfg, model, loss_ce, scheduler, optimizer, epochID, max_epoch, max_acc, train_loader, val_loader, snapshot_path):
    model.train()
    trainLoss = 0
    lossTrainNorm = 0

    scheduler.step()
    pbar = tqdm(total=cfg['bath_per_epoch'])
    for batchID, (index, input, target, binary_label) in enumerate(train_loader):
        if batchID > cfg['bath_per_epoch']:
            break
        if batchID == 0:
            ss_time = time.time()
        input = input[:,0].float()
        varInput = torch.autograd.Variable(input).cuda()
        varTarget = torch.autograd.Variable(target.contiguous().cuda())
        var_Binary_Target = torch.autograd.Variable(binary_label.contiguous().cuda())
        optimizer.zero_grad()

        logit = model(varInput)
        lossvalue = loss_ce(logit, var_Binary_Target)

        lossvalue.backward()
        optimizer.step()

        trainLoss += lossvalue.item()
        lossTrainNorm += 1
        pbar.set_postfix(loss=trainLoss / lossTrainNorm)
        pbar.update(1)
        del lossvalue

    trainLoss = trainLoss / lossTrainNorm

    if (epochID+1) % 1 == 0:
        pred_accuracy, pred_f1_score, video_id, predictLabels, givenLabels, outpred = eval_model(cfg, model, val_loader, loss_ce, cfg['val_batch_size'])    

        torch.save(
            {"epoch": epochID + 1, "model_state_dict": model.state_dict()},
            snapshot_path + "/last"+ ".pth",
            )

        if pred_accuracy > max_acc:
            max_epoch, max_acc = epochID, pred_accuracy
            torch.save(
            {"epoch": epochID + 1, "model_state_dict": model.state_dict()},
            snapshot_path + "/best_acc"+ ".pth",
            )

        df_result = pd.DataFrame({
            'data_path': video_id,
            'predicted_label': predictLabels,
            'actual_label': givenLabels,
            'predicted_prob':outpred
        })

        tempResult = snapshot_path+'/Epoch_'+str(epochID)+'_accuracy_f1.txt'
        with open(tempResult, 'w') as file:
            givenLabels = df_result['actual_label']
            predictProb = df_result['predicted_prob'] 
            auc = roc_auc_score(givenLabels, predictProb)
            ap = average_precision_score(givenLabels, predictProb)
            file.write(f"Accuracy: {pred_accuracy:.2%}\n")
            file.write(f"F1 Score: {pred_f1_score:.2%}\n")
            file.write(f"AUC: {auc:.2%}\n")
            file.write(f"AP: {ap:.2%}\n")

        print("*****Average Training loss",str(trainLoss),"*****\n")
        print("*****Epoch", str(epochID), "*****Acc", str(pred_accuracy), '*****', "F1", str(pred_f1_score), '\n', "*****Max acc epoch", str(max_epoch), "*****Acc", str(max_acc), '*****\n')
    end_time = time.time()

    return max_epoch, max_acc, end_time - ss_time
