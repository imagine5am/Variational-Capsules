import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from CustomDataset import CustomDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate(model, args):
    model.eval()
    sample_count = 0
    running_loss = 0
    running_acc = 0

    with torch.no_grad():
        dataset = CustomDataset(split_type='test')
        train_dataloader = DataLoader(dataset, pin_memory=True, num_workers=8,batch_size=args.    batch_size)
        num_train_samples = len(train_dataloader.dataset)
        
        for i, (inputs, labels) in enumerate(tqdm(train_dataloader)):
            labels = np.transpose(labels, (3, 1, 2))
            #mask = np.expand_dims(mask, axis=0)
            labels = labels.type(torch.LongTensor)
            # onehot_labels = torch.zeros(labels.size(0),
            #     args.n_classes).scatter_(1, labels.view(-1, 1), 1).cuda()
            inputs = np.transpose(inputs, (3, 1, 2)) / 255.
            # frame = np.expand_dims(frame, axis=0)
            inputs = inputs.type(torch.FloatTensor).cuda()

            yhat = model(inputs)

            loss = F.BCEWithLogitsLoss(yhat, labels.cuda())

            sample_count += inputs.size(0)
            running_loss += loss.item() * inputs.size(0) # smaller batches count less

        loss = running_loss / sample_count

    return loss
