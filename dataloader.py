import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import albumentations
import random
import os
import numpy as np
import cv2
import math
import warnings


def crop_center_by_percentage(image, percentage):
    height, width = image.shape[:2]

    if width > height:
        leftPix = int(width * percentage)
        rightPix = int(width * percentage)
        startPos = leftPix
        endPos = width - rightPix
        crop = image[:, startPos:endPos]
    else:
        up_pixels = int(height * percentage)
        down_pixels = int(height * percentage)
        start_y = up_pixels
        end_y = height - down_pixels
        crop = image[start_y:end_y, :]

    return crop

def is_valid_image_placement(pixlesGrid, row, col, img_id):
    for i in range(9):
        if pixlesGrid[row][i] == img_id or pixlesGrid[i][col] == img_id:
            return False
        if pixlesGrid[row - row % 3 + i // 3][col - col % 3 + i % 3] == img_id:
            return False
    return True

def solve_image_grid(pixlesGrid):
    for row in range(9):
        for col in range(9):
            if pixlesGrid[row][col] == 0:  
                for img_id in range(1, 10):
                    if is_valid_image_placement(pixlesGrid, row, col, img_id):
                        pixlesGrid[row][col] = img_id
                        if solve_image_grid(pixlesGrid):
                            return True
                        pixlesGrid[row][col] = 0
                return False
    return True

def compress_frame(frame):
    compressed = []
    for row in frame:
        count = 1
        compressed_row = []
        for i in range(1, len(row)):
            if row[i] == row[i - 1]:
                count += 1
            else:
                compressed_row.append((row[i - 1], count))
                count = 1
        compressed_row.append((row[-1], count))
        compressed.append(compressed_row)
    return compressed

class Ours_Dataset_train(Dataset):
    def __init__(self, index_list=None, df=None):
        self.index_list = index_list
        self.df = df
        self.positive_indices = df[df['label'] == 1].index.tolist()
        self.negative_indices = df[df['label'] == 0].index.tolist()
        self.balanced_indices = []
        self.resample()

    def resample(self):
        # Ensure each epoch uses a balanced dataset
        min_samples = min(len(self.positive_indices), len(self.negative_indices))
        self.balanced_indices.clear()
        self.balanced_indices.extend(random.sample(self.positive_indices, min_samples))
        self.balanced_indices.extend(random.sample(self.negative_indices, min_samples))
        random.shuffle(self.balanced_indices)  # Shuffle to mix positive and negative samples


    def __getitem__(self, idx):
        real_idx = self.balanced_indices[idx]
        row = self.df.iloc[real_idx]
        vidID = row['content_path']
        label = row['label']
        frameSet = eval(row['frame_seq'])
        label_onehot = [0]*2
        useFrameCount = 8

        augSet  = [
                    albumentations.Resize(224, 224)
                    ]

        if random.random() < 0.5:
            augSet.append(albumentations.HorizontalFlip(p=1.0))
        if random.random() < 0.5:
            qualityRating = random.randint(50, 100)
            augSet.append(albumentations.ImageCompression(quality_lower=qualityRating, quality_upper=qualityRating))
        if random.random() < 0.3:
            augSet.append(albumentations.GaussNoise(p=1.0))
        if random.random() < 0.3:
            augSet.append(albumentations.GaussianBlur(blur_limit=(3, 5), p=1.0))
        if random.random() < 0.001:
            augSet.append(albumentations.ToGray(p=1.0))
            
        augSet.append(albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0))
        trans = albumentations.Compose(augSet)

        if len(frameSet) >= useFrameCount:
            startFrame = random.randint(0, len(frameSet)-useFrameCount)
            selectedFrame = frameSet[startFrame:startFrame+useFrameCount]
            frames = []
            for x in frameSet[startFrame:startFrame+useFrameCount]:
                while True:
                    try:
                        temp_image_path = vidID+'/'+str(x)+'.jpg'
                        image = download_oss_file('GenVideo/'+ temp_image_path)  
                        if vidID.startswith("real/youku"):
                            image = crop_center_by_percentage(image, 0.15)
                        break
                    except Exception as e:
                        if x+1 < len(frameSet):
                            x = x + 1
                        elif x - 1 >=0 :
                            x = x - 1
                augmented = trans(image=image)
                image = augmented["image"]
                frames.append(image.transpose(2,0,1)[np.newaxis,:])
        else:
            pad_num = useFrameCount-len(frameSet)
            frames = []
            for x in frameSet:
                temp_image_path = vidID+'/'+str(x)+'.jpg'
                image = download_oss_file('GenVideo/'+temp_image_path)
                if vidID.startswith("real/youku"):
                    image = crop_center_by_percentage(image, 0.15)
                augmented = trans(image=image)
                image = augmented["image"]
                frames.append(image.transpose(2,0,1)[np.newaxis,:])    
            for i in range(pad_num):
                frames.append(np.zeros((224,224,3)).transpose(2,0,1)[np.newaxis,:])
        
        label_onehot[int(label)] = 1
        frames = np.concatenate(frames, 0)
        frames = torch.tensor(frames[np.newaxis,:])
        label_onehot = torch.FloatTensor(label_onehot)
        binary_label = torch.FloatTensor([int(label)])

        return self.index_list[idx], frames, label_onehot, binary_label

    def __len__(self):
        return len(self.balanced_indices)

def blocker(frame):
    def directFrameSelect(x, y):
        if x < 0 or y < 0 or x >= len(frame) or y >= len(frame[0]) or frame[x][y] == 0:
            return 0
        frame[x][y] = 0 
        return 1 + directFrameSelect(x+1, y) + directFrameSelect(x-1, y) + directFrameSelect(x, y+1) + directFrameSelect(x, y-1)

    max_block = 0
    for i in range(len(frame)):
        for j in range(len(frame[0])):
            if frame[i][j] == 1:
                max_block = max(max_block, directFrameSelect(i, j))
    return max_block

class Ours_Dataset_val(data.Dataset):
    def __init__(self, cfg, index_list=None, df=None):
        self.index_list = index_list
        self.cfg = cfg
        self.df = df
        self.frame_dir = df['image_path'].tolist()

    def __getitem__(self, idx):
        augSet  = [
                    albumentations.Resize(224, 224),
                    ]
        
        if self.cfg['task'] == 'JPEG_Compress_Attack':
            augSet.append(albumentations.JpegCompression(quality_lower=35, quality_upper=35,p=1.0))
        if self.cfg['task'] == 'FLIP_Attack':
            if random.random() < 0.5:
                augSet.append(albumentations.HorizontalFlip(p=1.0))
            else:
                augSet.append(albumentations.VerticalFlip(p=1.0))
        if self.cfg['task'] == 'CROP_Attack':
            random_crop_x = random.randint(0, 16)  
            random_crop_y = random.randint(0, 16)  
            crop_width = random.randint(160, 208) 
            crop_height = random.randint(160, 208)
            augSet.append(albumentations.Crop(x_min=random_crop_x, y_min=random_crop_y, x_max=random_crop_x+crop_width, y_max=random_crop_y+crop_height))
            augSet.append(albumentations.Resize(224, 224))

        if self.cfg['task'] == 'Color_Attack':
            index = random.choice([i for i in range(4)])
            dicts = {0:[0.5,0,0,0],1:[0,0.5,0,0],2:[0,0,0.5,0],3:[0,0,0,0.5]}
            brightness,contrast,saturation,hue = dicts[index]
            augSet.append(albumentations.ColorJitter(
                brightness=brightness, contrast=contrast, saturation=saturation, hue=hue))

        if self.cfg['task'] == 'Gaussian_Attack':     
            augSet.append(albumentations.GaussianBlur(blur_limit=(7, 7), p=1.0))

        augSet.append(albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0))
        trans = albumentations.Compose(augSet)


        df_v = self.df.loc[self.index_list[idx]]
        vidID = df_v['content_path']
        activity_id = df_v['activity_id']
        label = df_v['label']
        label_onehot = [0]*2
        frameSet = eval(df_v['frame_seq'])

        useFrameCount = 8

        if len(frameSet) >= useFrameCount:
            startFrame = random.randint(0, len(frameSet)-useFrameCount)
            selectedFrame = frameSet[startFrame:startFrame+useFrameCount]
            frames = []
            for x in frameSet[startFrame:startFrame+useFrameCount]:
                while True:
                    try:
                        temp_image_path = vidID+'/'+str(x)+'.jpg'
                        image = download_oss_file('GenVideo/'+ temp_image_path)
                        image = crop_center_by_percentage(image, 0.1)
                        break
                    except Exception as e:
                        if x+1 < len(frameSet):
                            x = x + 1
                        elif x - 1 >=0 :
                            x = x - 1
                augmented = trans(image=image)
                image = augmented["image"]
                frames.append(image.transpose(2,0,1)[np.newaxis,:])
        else:
            pad_num = useFrameCount-len(frameSet)
            frames = []
            for x in frameSet:
                temp_image_path = vidID+'/'+str(x)+'.jpg'
                image = download_oss_file('GenVideo/'+temp_image_path)
                image = crop_center_by_percentage(image, 0.1)
                augmented = trans(image=image)
                image = augmented["image"]
                frames.append(image.transpose(2,0,1)[np.newaxis,:])    
            for i in range(pad_num):
                frames.append(np.zeros((224,224,3)).transpose(2,0,1)[np.newaxis,:])

        label_onehot[int(label)] = 1
        frames = np.concatenate(frames, 0)
        frames = torch.tensor(frames[np.newaxis,:])
        label_onehot = torch.FloatTensor(label_onehot)
        binary_label = torch.FloatTensor([int(label)])
        return self.index_list[idx], frames, label_onehot, binary_label, vidID

    
    def __len__(self):
        return len(self.index_list)

def optimize_video_compression(frame_sizes, frame_qualities, storage_limit):
    num_frames = len(frame_sizes)
    dp = [[0] * (storage_limit + 1) for _ in range(num_frames + 1)]

    for i in range(1, num_frames + 1):
        for storage in range(1, storage_limit + 1):
            if frame_sizes[i - 1] <= storage:
                dp[i][storage] = max(dp[i - 1][storage], 
                                     dp[i - 1][storage - frame_sizes[i - 1]] + frame_qualities[i - 1])
            else:
                dp[i][storage] = dp[i - 1][storage]

    return dp[num_frames][storage_limit]

def calculate_pixel_gradient(frame):
    gradients = []
    for row in frame:
        gradient_row = []
        for i in range(1, len(row)):
            gradient_row.append(abs(row[i] - row[i - 1]))
        gradients.append(gradient_row)
    return gradients

def normalize_frame(frame):
    max_pixel = max(max(row) for row in frame)
    min_pixel = min(min(row) for row in frame)

    return [
        [(pixel - min_pixel) / (max_pixel - min_pixel) if max_pixel > min_pixel else 0 for pixel in row]
        for row in frame
    ]

def generate_dataset_loader(cfg):
    df_train = pd.read_csv('GenVideo/datasets/train.csv')

    if cfg['task'] == 'normal':
        df_val = pd.read_csv('GenVideo/datasets/val_id.csv')
    elif cfg['task'] == 'robust_compress':
        df_val = pd.read_csv('GenVideo/datasets/com_28.csv')
    elif cfg['task'] == 'Image_Water_Attack':
        df_val = pd.read_csv('GenVideo/datasets/imgwater.csv')
    elif cfg['task'] == 'Text_Water_Attack':
        df_val = pd.read_csv('GenVideo/datasets/textwater.csv')
    elif cfg['task'] == 'one2many':
        df_val = pd.read_csv('GenVideo/datasets/val_ood.csv')
        if cfg['train_sub_set'] == 'pika':
            prefixes = ["fake/pika", "real"]
            video_condition = df_train['content_path'].str.startswith(prefixes[0])
            for prefix in prefixes[1:]:
                video_condition |= df_train['content_path'].str.startswith(prefix)
            df_train = df_train[video_condition]
        elif cfg['train_sub_set'] == 'SEINE':
            prefixes = ["fake/SEINE", "real"]
            video_condition = df_train['content_path'].str.startswith(prefixes[0])
            for prefix in prefixes[1:]:
                video_condition |= df_train['content_path'].str.startswith(prefix)
            df_train = df_train[video_condition]
        elif cfg['train_sub_set'] == 'OpenSora':
            prefixes = ["fake/OpenSora", "real"]
            video_condition = df_train['content_path'].str.startswith(prefixes[0])
            for prefix in prefixes[1:]:
                video_condition |= df_train['content_path'].str.startswith(prefix)
            df_train = df_train[video_condition]
        elif cfg['train_sub_set'] == 'Latte':
            prefixes = ["fake/Latte", "real"]
            video_condition = df_train['content_path'].str.startswith(prefixes[0])
            for prefix in prefixes[1:]:
                video_condition |= df_train['content_path'].str.startswith(prefix)
            df_train = df_train[video_condition]
    else:
        df_val = pd.read_csv('GenVideo/datasets/val_ood.csv')

    df_train.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)
    
    index_val = df_val.index.tolist()
    index_val = index_val[:]

    val_dataset = Ours_Dataset_val(cfg, index_val, df_val)
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg['val_batch_size'], shuffle=False, num_workers=cfg['num_workers'], pin_memory=True, drop_last=False
        )

    index_train = df_train.index.tolist()
    index_train = index_train[:]
    train_dataset = Ours_Dataset_train(index_train, df_train)
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg['train_batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=True, drop_last=True
        )

    print("******* Training Video IDs", str(len(index_train))," Training Batch size ", str(cfg['train_batch_size'])," *******")
    print("******* Testing Video IDs", str(len(index_val)), " Testing Batch size ", str(cfg['val_batch_size'])," *******")

    return train_loader, val_loader


