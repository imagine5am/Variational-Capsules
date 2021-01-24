import gc, time, copy, torch, logging, torchvision
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from CustomDataset import CustomDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from evaluate import evaluate

DEBUG = False

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
    
    dataset = CustomDataset()
    dataloader = DataLoader(dataset, shuffle=True, pin_memory=True, num_workers=8,batch_size=args.batch_size)
    num_train_samples = len(dataloader.dataset)
    
    test_dataset = CustomDataset(split_type='test')
    test_dataloader = DataLoader(test_dataset, pin_memory=True, num_workers=8,batch_size=args.batch_size)
    
    # loss_fn = F.BCEWithLogitsLoss
    loss_fn = nn.BCELoss(reduce='sum')
    # loss_fn = F.cross_entropy    
    
    for epoch in range(args.n_epochs):
        model.train()
        sample_count = 0
        running_loss = 0

        logging.info('\nEpoch {}/{}:\n'.format(epoch+1, args.n_epochs))

        for i, (inputs, labels) in enumerate(tqdm(dataloader)):
            if inputs.shape[0] == 1:
                continue
            
            args.step = (epoch * num_train_samples) + i + 1
            
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

            optimiser.zero_grad()
            yhat = model(inputs)
            
            if DEBUG:
                print(f'yhat.shape: {yhat.shape} | yhat.dtype: {yhat.dtype}')
            
            loss = loss_fn(yhat, labels.cuda())
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimiser.step()

            sample_count += inputs.size(0)
            running_loss += loss.item() * inputs.size(0) # smaller batches count less
            # running_acc += (yhat.argmax(-1) == mask.cuda()).sum().item() # n_corrects

        train_loss = running_loss / sample_count
        # epoch_train_acc = running_acc / sample_count
        
        if (epoch+1) % 2 == 0:
            del dataset, dataloader
            
            dataset = CustomDataset()
            dataloader = DataLoader(dataset, shuffle=True, pin_memory=True, 
                                          num_workers=8, batch_size=args.batch_size)
            num_train_samples = len(dataloader.dataset)
        
        if (epoch+1) % 5 == 0:
            test_loss = evaluate(model, args, test_dataloader)
            logging.info('\nTrain loss: {:.4f} | Test Loss: {:.4f}'.format(train_loss, 
                                                                           test_loss))
            
            args.writer.add_scalars('epoch_loss', {'train': train_loss,
                                          'valid': test_loss}, epoch+1)
            
            
            
        else:
            logging.info('\nTrain loss: {:.4f}'.format(train_loss))

        
        """
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
    """
    return test_loss
