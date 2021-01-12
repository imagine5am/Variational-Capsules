import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from CustomDataset import CustomDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

DEBUG = False

def evaluate(model, args, dataloader):
    model.eval()
    sample_count = 0
    running_loss = 0
    # running_acc = 0
    
    loss_fn = nn.BCELoss()

    with torch.no_grad():
        # num_train_samples = len(dataloader.dataset)
        
        for i, (inputs, labels) in enumerate(tqdm(dataloader)):
            if inputs.shape[0] == 1:
                continue
            
            if DEBUG:
                print(f'inputs.shape: {inputs.shape} | inputs.dtype: {inputs.dtype}')
                print(f'labels.shape: {labels.shape} | labels.dtype: {labels.dtype}')
            
            inputs = inputs.permute(0, 3, 1, 2)
            inputs = inputs.type(torch.FloatTensor).cuda()
            labels = labels.permute(0, 3, 1, 2)
            labels = labels.type(torch.FloatTensor)
            
            if DEBUG:
                print(f'inputs.shape: {inputs.shape} | inputs.dtype: {inputs.dtype}')
                print(f'labels.shape: {labels.shape} | labels.dtype: {labels.dtype}')

            yhat = model(inputs)

            loss = loss_fn(yhat, labels.cuda())

            sample_count += inputs.size(0)
            running_loss += loss.item() * inputs.size(0) # smaller batches count less
            # running_acc += (yhat.argmax(-1) == labels.data.cuda()).sum().item() # n_corrects

        loss = running_loss / sample_count
        # acc = running_acc / sample_count

    # return loss, acc
    return loss
