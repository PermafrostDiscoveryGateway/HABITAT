# Preparation and Preprocessing

# Import libraries

from final_model_config import *
from dataloader import *
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from utils import *

def model_train():


    # Paths to folders containing training/validation images and masks

    x_train_dir = Final_Config.INPUT_IMG_DIR + '/train'
    y_train_dir = Final_Config.INPUT_MASK_DIR + '/train'

    x_val_dir = Final_Config.INPUT_IMG_DIR + '/val'
    y_val_dir = Final_Config.INPUT_MASK_DIR + '/val'

    # Functions for transfer learning

    def freeze_encoder(model):
        for child in model.encoder.children():
            for param in child.parameters():
                param.requires_grad = False
        return

    def unfreeze(model):
        for child in model.children():
            for param in child.parameters():
                param.requires_grad = True
        return

    model = Final_Config.MODEL

    freeze_encoder(model)

    # Create training and validation datasets and dataloaders with augmentations and proper preprocessing.

    # If no augmentations are to be used, set augmentation to None
    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(Final_Config.PREPROCESS)
    )

    val_dataset = Dataset(
        x_val_dir,
        y_val_dir,  
        preprocessing=get_preprocessing(Final_Config.PREPROCESS)
    )

    train_loader = DataLoader(train_dataset, batch_size=Final_Config.TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)

    train_loader = DataLoader(train_dataset, batch_size=Final_Config.TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Final_Config.VAL_BATCH_SIZE, shuffle=False, num_workers=0)

    # Create epoch runners to iterating over dataloader`s samples.

    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=Final_Config.LOSS, 
        metrics=Final_Config.METRICS, 
        optimizer=Final_Config.OPTIMIZER,
        device=Final_Config.DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=Final_Config.LOSS, 
        metrics=Final_Config.METRICS, 
        device=Final_Config.DEVICE,
        verbose=True,
    )


    # Train model and save weights

    max_score = 0

    # Lists to keep track of losses and accuracies.
    train_acc = []
    train_loss = []

    val_acc = []
    val_loss = []

    for i in range(0, Final_Config.EPOCHS):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        val_logs = valid_epoch.run(val_loader)

        # Print and log F1-score
        print(train_logs['fscore'])
        print(val_logs['fscore'])
        train_acc.append(train_logs['fscore'])
        val_acc.append(val_logs['fscore'])

        # Print and log focal loss   
        print(train_logs['FocalLoss'])
        print(val_logs['FocalLoss'])
        train_loss.append(train_logs['FocalLoss'])
        val_loss.append(val_logs['FocalLoss'])
        
        # do something (save model, change lr, etc.)
        if max_score < val_logs['fscore']:
            max_score = val_logs['fscore']
            torch.save(model, Final_Config.WEIGHT_DIR + '.pth')
            print('Model saved!')
        
        # If desired, the below code adds in learning rate decay.

        if i == 35:
            Final_Config.OPTIMIZER.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

        # Save the loss and accuracy plots.
        save_plots(
            train_acc, val_acc, train_loss, val_loss,
            Final_Config.PLOT_DIR + '_accuracy.png',
            Final_Config.PLOT_DIR + '_loss.png',
        ) 
        







