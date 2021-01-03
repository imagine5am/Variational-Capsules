import gc, time, copy, torch, logging, torchvision
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from DataLoader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from evaluate import evaluate


def train(model, args):

    # logging.info('\ntrain: {} - valid: {} - test: {}'.format(
    #     len(dataloaders['train'].dataset), len(dataloaders['valid'].dataset),
    #     len(dataloaders['test'].dataset)))

    optimiser = optim.Adam(model.parameters(),
        lr=args.learning_rate, weight_decay=args.weight_decay)

    since = time.time()
    best_model = copy.deepcopy(model.state_dict())
    best_valid_loss = np.inf
    best_valid_acc = 0
    patience_counter = 0

    for epoch in range(args.n_epochs):
        dataloader = DataLoader()
        num_train_samples = len(dataloader.data)
        model.train()
        sample_count = 0
        running_loss = 0

        logging.info('\nEpoch {}/{}:\n'.format(epoch+1, args.n_epochs))

        for i, (frame, mask, dataset) in enumerate(tqdm(dataloader.data)):
            args.step = (epoch * num_train_samples) + i + 1

            mask = np.transpose(mask, (2, 0, 1))
            mask = np.expand_dims(mask, axis=0)
            # mask = mask.type(torch.LongTensor)
            mask = torch.LongTensor(mask)
            # onehot_labels = torch.zeros(labels.size(0),
            #     args.n_classes).scatter_(1, labels.view(-1, 1), 1).cuda()
            
            frame = np.transpose(frame, (2, 0, 1)) / 255.
            frame = np.expand_dims(frame, axis=0)
            # frame = frame.type(torch.FloatTensor).cuda()
            frame = torch.FloatTensor(frame).cuda()

            optimiser.zero_grad()
            print(f'frame.shape: {frame.shape}')
            yhat = model(frame)
            print(f'yhat.shape: {yhat.shape}')
            print(f'mask.shape: {mask.shape}')
            loss = F.BCEWithLogitsLoss(yhat, mask.cuda())

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimiser.step()

            sample_count += frame.size(0)
            running_loss += loss.item() * frame.size(0) # smaller batches count less
            # running_acc += (yhat.argmax(-1) == mask.cuda()).sum().item() # n_corrects

        epoch_train_loss = running_loss / sample_count
        # epoch_train_acc = running_acc / sample_count

        epoch_valid_loss, epoch_valid_acc = evaluate(model, args, dataloaders['valid'])

        logging.info('\nTrain loss: {:.4f} | Valid loss: {:.4f}'.format(
            epoch_train_loss, epoch_valid_loss))

        args.writer.add_scalars('epoch_loss', {'train': epoch_train_loss,
                                          'valid': epoch_valid_loss}, epoch+1)

        if epoch_valid_acc >= best_valid_acc:
            patience_counter = 0
            best_epoch = epoch + 1
            best_valid_acc = epoch_valid_acc
            best_valid_loss = epoch_valid_loss

            torch.save(model.state_dict(), args.checkpoint_dir)

            if args.test_affNIST:
                if epoch_valid_acc >= 0.992 and epoch_valid_acc < 0.993:
                    test_loss, test_acc = evaluate(model, args, dataloaders['test'])
                    logging.info('affNIST Test loss: {:.4f} - acc: {:.4f}'.format(
                        test_loss, test_acc))
                    torch.save(model.state_dict(), args.checkpoint_dir)

        else: # early stopping
            patience_counter += 1
            if patience_counter == (args.patience-10):
                logging.info('\nPatience counter {}/{}.'.format(
                    patience_counter, args.patience))
            if patience_counter == args.patience:
                logging.info('\nEarly stopping... no improvement after {} Epochs.'.format(
                    args.patience))
                break

        gc.collect() # release unreferenced memory

    time_elapsed = time.time() - since
    logging.info('\nTraining time: {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    model.load_state_dict(torch.load(args.checkpoint_dir)) # load best model

    test_loss, test_acc = evaluate(model, args)

    logging.info('\nBest Valid: Epoch {} - Loss {:.4f} - Acc. {:.4f}'.format(
        best_epoch, best_valid_loss, best_valid_acc))
    logging.info('Test: Loss {:.4f} - Acc. {:.4f}'.format(test_loss, test_acc))

    return test_loss
