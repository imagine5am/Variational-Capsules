import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from DataLoader import DataLoader


def evaluate(model, args):
    model.eval()
    sample_count = 0
    running_loss = 0
    running_acc = 0

    with torch.no_grad():
        dataloader = DataLoader(split_type='test')
        
        for i, (frame, mask, dataset) in enumerate(dataloader.data):
            
            mask = np.transpose(mask, (2, 0, 1))
            mask = np.expand_dims(mask, axis=0)
            mask = mask.type(torch.LongTensor)
            # onehot_labels = torch.zeros(labels.size(0),
            #     args.n_classes).scatter_(1, labels.view(-1, 1), 1).cuda()
            frame = np.transpose(frame, (2, 0, 1)) / 255.
            frame = np.expand_dims(frame, axis=0)
            frame = frame.type(torch.FloatTensor).cuda()

            yhat = model(frame)

            loss = F.BCEWithLogitsLoss(yhat, mask.cuda())

            sample_count += frame.size(0)
            running_loss += loss.item() * frame.size(0) # smaller batches count less

        loss = running_loss / sample_count

    return loss
