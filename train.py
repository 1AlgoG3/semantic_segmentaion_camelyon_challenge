epochs=500
total_train_patches=10000
total_val_patches=5000

model_logs_dir='model_logs/loss_iou_from_checkpoint'

batch_size=8
patch_size=512
overlap=184

target_mpp=1
gradient_clipping=1.0
weight_decay=1e-8
learning_rate=1e-4
amp=True

import sys
sys.path.append('utility_box/')
from torch_gpu_utils import get_device
from image_utils import plot_image_series, plot_overlay, plot_image, plot_image_series, plot_overlay_series

import random
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
import numpy as np

import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader_with_augs import CamelyonDataset

from network_definition import UNet
from loss_functions import get_dice_loss, IoULoss

import segmentation_models_pytorch as smp

def create_model_logdir(model_logs_dir):
    """
    Creates a directory structure for model logging and checkpoints in the specified main folder.
    
    The structure created is as follows:
    ./<main_folder_name>/
        train_logs/
            train_images/
        val_logs/
            val_images/
        model_check_points/
    
    Args:
    main_folder_name (str): The name of the main folder where the directory structure will be created.
    """
    # Get the current working directory
    base_path = Path.cwd()

    # Define the directory structure using Path objects and the main folder name
    dir_structure = [
        base_path / model_logs_dir / 'train_logs' / 'train_images',
        base_path / model_logs_dir / 'val_logs' / 'val_images',
        base_path / model_logs_dir / 'model_check_points' / 'max_val',
        base_path / model_logs_dir / 'model_check_points' / 'recurrent'
    ]
    
    # Create the directories
    for dir_path in dir_structure:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

def crop_masks(masks):
    return masks

def crop_images(images):
    return images

def train_logic(model, epoch, train_loader, model_logs_dir):
    model.train()
    batch_train_logs=[]
    train_pbar=tqdm(total=len(train_loader), desc='training')
    train_epoch_loss=0
    save_image_counter=0
    for batch_idx, batch in enumerate(train_loader):
        images=batch[0].to(device) 
        true_masks=batch[1].to(device)
        true_masks=crop_masks(true_masks)
        
        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
            pred_masks=model(images)
            
        if n_classes == 1:
            loss=criterion(pred_masks.squeeze(1), true_masks.float())

            prob_masks = F.sigmoid(pred_masks)
            thresh_masks = (prob_masks > 0.5).float()
            
            dice_loss=get_dice_loss(thresh_masks.squeeze(1), true_masks.float(), multiclass=False)
            loss+=dice_loss
            
        else:
            loss = criterion(pred_masks, true_masks)
            dice_loss=get_dice_loss(
                F.softmax(pred_masks, dim=1).float(),
                F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                multiclass=True
            )
            loss+=dice_loss
        
        optimizer.zero_grad(set_to_none=True)
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        grad_scaler.step(optimizer)
        grad_scaler.update()

        train_epoch_loss += loss.item()

        train_temp_dict={}
        train_temp_dict['epoch']=epoch
        train_temp_dict['batch_idx']=batch_idx
        train_temp_dict['loss']=loss.item()
        train_temp_dict['dice_loss']=dice_loss.item()
        train_temp_dict['n_samples']=images.shape[0]
        batch_train_logs.append(train_temp_dict)
        
        if batch_idx%2==0:
            #randomly save a prediction and its true values for progress tracking
            random_idx=random.randint(0, len(images)-1)
            true_mask=crop_masks((true_masks[random_idx].to('cpu').detach().numpy()).astype(np.uint8))
            thresh_mask=thresh_masks[random_idx].squeeze(0).to('cpu').detach().numpy().astype(np.uint8)
            image=crop_images(images[random_idx].to('cpu').numpy().transpose(1,2,0))

            image_series_path=Path(f"{model_logs_dir}/train_logs/train_images/epoch{epoch}/epoch{epoch}_batch{batch_idx}_image{save_image_counter}.jpg")
            overlay_series_path=Path(f"{model_logs_dir}/train_logs/train_images/epoch{epoch}/overlay_epoch{epoch}_batch{batch_idx}_image{save_image_counter}.jpg")
            image_series_path.parent.mkdir(parents=True, exist_ok=True)
            overlay_series_path.parent.mkdir(parents=True, exist_ok=True)
            
            plot_image_series([true_mask,thresh_mask,image], 
                              ['true_mask', 'thresh_mask', 'image'],
                              save_path=image_series_path,
                              plot=False
                             )
            plot_overlay_series([image, image, image],
                                [true_mask, thresh_mask, torch.zeros(image.shape[:2])],
                                ['true_mask', 'thresh_mask', 'image'],
                                save_path=overlay_series_path,
                                plot=False)
            
            save_image_counter+=1

        pd.DataFrame(batch_train_logs).to_csv(f"{model_logs_dir}/train_logs/epoch{epoch}_batch_train_logs.csv", index=False)
        train_pbar.update()
    
    return model, train_epoch_loss, pd.DataFrame(batch_train_logs)

def val_logic(model, epoch, val_loader, model_logs_dir):
    model.eval()
    batch_val_logs=[]
    val_pbar=tqdm(total=len(val_loader), desc='validating')
    val_epoch_loss=0
    save_image_counter=0
    
    for batch_idx, batch in enumerate(val_loader):
        images=batch[0].to(device) 
        true_masks=batch[1].to(device)
        true_masks=crop_masks(true_masks)

        with torch.inference_mode():
            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                pred_masks=model(images)

            if n_classes == 1:
                val_loss=criterion(pred_masks.squeeze(1), true_masks.float())

                prob_masks = F.sigmoid(pred_masks)
                thresh_masks = (prob_masks > 0.5).float()
                
                val_dice_loss=get_dice_loss(thresh_masks.squeeze(1), true_masks.float(), multiclass=False)
                val_loss+=val_dice_loss
            else:
                val_loss = criterion(pred_masks, true_masks)
                val_dice_loss=get_dice_loss(
                    F.softmax(pred_masks, dim=1).float(),
                    F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                    multiclass=True
                )
                val_loss+=val_dice_loss

        val_epoch_loss += val_loss.item()

        val_temp_dict={}
        val_temp_dict['epoch']=epoch
        val_temp_dict['batch_idx']=batch_idx
        val_temp_dict['loss']=val_loss.item()
        val_temp_dict['dice_loss']=val_dice_loss.item()
        val_temp_dict['n_samples']=images.shape[0]
        batch_val_logs.append(val_temp_dict)
        
        #randomly save a prediction and its true values for progress tracking
        if batch_idx%2==0:
            random_idx=random.randint(0, len(images)-1)
            true_mask=crop_masks((true_masks[random_idx].to('cpu').detach().numpy()>0.5).astype(np.uint8))
            thresh_mask=thresh_masks[random_idx].squeeze(0).to('cpu').detach().numpy().astype(np.uint8)
            image=crop_images(images[random_idx].to('cpu').numpy().transpose(1,2,0))

            image_series_path=Path(f"{model_logs_dir}/val_logs/val_images/epoch{epoch}/epoch{epoch}_batch{batch_idx}_image{save_image_counter}.jpg")
            overlay_series_path=Path(f"{model_logs_dir}/val_logs/val_images/epoch{epoch}/overlay_epoch{epoch}_batch{batch_idx}_image{save_image_counter}.jpg")
            
            image_series_path.parent.mkdir(parents=True, exist_ok=True)
            overlay_series_path.parent.mkdir(parents=True, exist_ok=True)
            
            plot_image_series([true_mask,thresh_mask,image],
                              ['true_mask', 'thresh_mask', 'image'],
                              save_path=image_series_path,
                              plot=False
                             )
            plot_overlay_series([image,image,image],
                                [true_mask,thresh_mask,torch.zeros(image.shape[:2])],
                                ['true_mask', 'thresh_mask', 'image'],
                                save_path=overlay_series_path,
                                plot=False)
            save_image_counter+=1

        pd.DataFrame(batch_val_logs).to_csv(f"{model_logs_dir}/val_logs/epoch{epoch}_batch_val_logs.csv", index=False)
        val_pbar.update()
        
    return model, val_epoch_loss, pd.DataFrame(batch_val_logs)


def get_training_augmentation():
    scale = 0.25 #Maximum Should be 0.5, Downscale
    
    scale_setting = 0.25 #Color Jitter
    scale_color = 0.1
    
    augmentations = [
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Blur(p=0.1, blur_limit=9),
        A.GaussNoise(p=0.2, var_limit=10),
        A.ColorJitter(p=0.5,brightness=scale_setting,contrast=scale_setting,saturation=scale_color,hue=scale_color / 2,),
        A.RandomBrightnessContrast(p=0.2),
    ]
    
    augmentations = A.Compose(augmentations)
    return augmentations

train_augs=get_training_augmentation()

train_distribution=pd.read_csv('distributions/train_distribution.csv')
val_distribution=pd.read_csv('distributions/val_distribution.csv')
test_distribution=pd.read_csv('distributions/test_distribution.csv')

create_model_logdir(model_logs_dir)
device=get_device(0)

train_dataset_args_dict1 = {
    'data_distribution': train_distribution,
    'patch_size': patch_size,
    'overlap': overlap,
    'target_mpp': target_mpp,
    'training_mode': 'pure_positive',
    'slice_tumor': False,
    'total_patches': total_train_patches,
    'sample_without_replacement': True,
    'n_sample_rows': 40
}

train_dataset_args_dict2 = {
    'data_distribution': train_distribution,
    'patch_size': patch_size,
    'overlap': overlap,
    'target_mpp': target_mpp,
    'training_mode': 'pure_negative',
    'slice_tumor': False,
    'total_patches': total_train_patches,
    'sample_without_replacement': True,
    'n_sample_rows': 40
}

train_dataset_args_dict3 = {
    'data_distribution': train_distribution,
    'patch_size': patch_size,
    'overlap': overlap,
    'target_mpp': target_mpp,
    'training_mode': 'mixed',
    'slice_tumor': False,
    'total_patches': total_train_patches,
    'sample_without_replacement': True,
    'n_sample_rows': 40
}

train_dataset_args_dict4 = {
    'data_distribution': train_distribution,
    'patch_size': patch_size,
    'overlap': overlap,
    'target_mpp': target_mpp,
    'training_mode': 'pure_positive',
    'slice_tumor': True,
    'total_patches': total_train_patches,
    'sample_without_replacement': True,
    'n_sample_rows': 1
}

val_dataset_args_dict1 = {
    'data_distribution': val_distribution,
    'patch_size': patch_size,
    'overlap': overlap,
    'target_mpp': target_mpp,
    'training_mode': 'pure_positive',
    'slice_tumor': False,
    'total_patches': total_val_patches,
    'sample_without_replacement': True,
    'n_sample_rows': 10
}

val_dataset_args_dict2 = {
    'data_distribution': val_distribution,
    'patch_size': patch_size,
    'overlap': overlap,
    'target_mpp': target_mpp,
    'training_mode': 'pure_negative',
    'slice_tumor': False,
    'total_patches': total_val_patches,
    'sample_without_replacement': True,
    'n_sample_rows': 10
}

val_dataset_args_dict3 = {
    'data_distribution': val_distribution,
    'patch_size': patch_size,
    'overlap': overlap,
    'target_mpp': target_mpp,
    'training_mode': 'mixed',
    'slice_tumor': False,
    'total_patches': total_val_patches,
    'sample_without_replacement': True,
    'n_sample_rows': 10
}

val_dataset_args_dict4 = {
    'data_distribution': val_distribution,
    'patch_size': patch_size,
    'overlap': overlap,
    'target_mpp': target_mpp,
    'training_mode': 'pure_positive',
    'slice_tumor': True,
    'total_patches': total_val_patches,
    'sample_without_replacement': True,
    'n_sample_rows': 1
}

train_dataset_args_choices=[train_dataset_args_dict1, train_dataset_args_dict2, train_dataset_args_dict3, train_dataset_args_dict4]
val_dataset_args_choices=[val_dataset_args_dict1, val_dataset_args_dict2, val_dataset_args_dict3, val_dataset_args_dict4]

state_dict_path='/workspace/code/NodeSeg/model_logs/smp_unet_500epochs_run2/model_check_points/max_val/checkpoint_epoch305_0.9842888563871384.pth'
state_dict=torch.load(state_dict_path, weights_only=True)

n_classes=1
model = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=n_classes,                      # model output channels (number of classes in your dataset)
)
model.load_state_dict(state_dict)
model.to(device);
model=model.to(memory_format=torch.channels_last)

optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)  # goal: minimize validation loss
grad_scaler=torch.amp.GradScaler(enabled=amp)
criterion = IoULoss().to(device)

epoch_logs=[]
for epoch in tqdm(range(1,epochs+1)):
    try:
        if epoch%2==0:
            train_dataset_args_dict=train_dataset_args_choices[3]
            train_dataset=CamelyonDataset(**train_dataset_args_dict, augmentations=train_augs)
            train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
            model, train_epoch_loss, batch_train_logs_df=train_logic(model, epoch, train_loader, model_logs_dir)
    
            val_dataset_args_dict=val_dataset_args_choices[3]
            val_dataset=CamelyonDataset(**val_dataset_args_dict)
            val_loader=DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=8)
            model, val_epoch_loss, batch_val_logs_df = val_logic(model, epoch, val_loader, model_logs_dir)
        else:
            train_dataset_args_dict=random.choice(train_dataset_args_choices)
            train_dataset=CamelyonDataset(**train_dataset_args_dict, augmentations=train_augs)
            train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
            model, train_epoch_loss, batch_train_logs_df=train_logic(model, epoch, train_loader, model_logs_dir)
    
            val_dataset_args_dict=random.choice(val_dataset_args_choices)
            val_dataset=CamelyonDataset(**val_dataset_args_dict)
            val_loader=DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=8)
            model, val_epoch_loss, batch_val_logs_df = val_logic(model, epoch, val_loader, model_logs_dir)
    
        train_dice_score=1-batch_train_logs_df['dice_loss'].mean()
        val_dice_score=1-batch_val_logs_df['dice_loss'].mean()
        scheduler.step(val_dice_score)
    
        epoch_temp_dict={}    
        epoch_temp_dict['epoch']=epoch
        epoch_temp_dict['avg_train_loss']=train_epoch_loss/len(train_dataset)
        epoch_temp_dict['avg_val_loss']=val_epoch_loss/len(val_dataset)
        epoch_temp_dict['train_dice']=train_dice_score
        epoch_temp_dict['val_dice']=val_dice_score
        epoch_temp_dict['learning rate']=optimizer.param_groups[0]['lr']
        
        epoch_temp_dict['train_loss']=train_epoch_loss
        epoch_temp_dict['val_loss']=val_epoch_loss
        epoch_temp_dict['train_mode']=train_dataset_args_dict['training_mode']
        epoch_temp_dict['train_slice_tumor']=train_dataset_args_dict['slice_tumor']
        epoch_temp_dict['train_sample_without_replacement']=train_dataset_args_dict['sample_without_replacement']
        epoch_temp_dict['val_mode']=val_dataset_args_dict['training_mode']
        epoch_temp_dict['val_slice_tumor']=val_dataset_args_dict['slice_tumor']
        epoch_temp_dict['val_sample_without_replacement']=val_dataset_args_dict['sample_without_replacement']
    
        epoch_logs.append(epoch_temp_dict)
        pd.DataFrame(epoch_logs).to_csv(f"{model_logs_dir}/model_learning_logs.csv", index=False)
     
        if epoch==1:
            max_val_score=val_dice_score
            print(f"max val score initialised to {round(max_val_score, 2)} for epoch {epoch}.")
            state_dict = model.state_dict()
            checkpoint_path=f"{model_logs_dir}/model_check_points/recurrent/checkpoint_epoch{epoch}_{val_dice_score}.pth"
            torch.save(state_dict, checkpoint_path)
            
        elif val_dice_score>max_val_score:
            max_val_score=val_dice_score
            print(f"max val score changed to {round(max_val_score, 2)} for epoch {epoch}.")
            state_dict = model.state_dict()
            checkpoint_path=f"{model_logs_dir}/model_check_points/max_val/checkpoint_epoch{epoch}_{val_dice_score}.pth"
            torch.save(state_dict, checkpoint_path)
            
        elif epoch%10==0:
            print(f"max val {round(max_val_score, 2)} score unchanged for epoch {epoch}, val score {val_dice_score}")
            state_dict = model.state_dict()
            checkpoint_path=f"{model_logs_dir}/model_check_points/recurrent/checkpoint_epoch{epoch}_{val_dice_score}.pth"
            torch.save(state_dict, checkpoint_path)
        else:
            print(f"max val score unchanged for epoch {epoch}, val score : {round(max_val_score, 2)}")
    except Exception as e:
        print(f"epoch_error:{e}")


