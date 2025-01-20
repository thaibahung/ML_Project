from transformers import XCLIPVisionModel
import os
import sys
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from dualist_base import DualistConfig, ResidualBlock
import torch.nn.init as init
from clip import clip
import math

def blend_frames(frame1, frame2):
    blended = []
    for row1, row2 in zip(frame1, frame2):
        blended_row = [(p1 + p2) // 2 for p1, p2 in zip(row1, row2)]
        blended.append(blended_row)
    return blended

def detect_edges(frame, threshold):
    edges = []
    for i in range(len(frame)):
        edge_row = []
        for j in range(len(frame[i])):
            is_edge = False
            if i > 0 and abs(frame[i][j] - frame[i-1][j]) > threshold:
                is_edge = True
            if j > 0 and abs(frame[i][j] - frame[i][j-1]) > threshold:
                is_edge = True
            edge_row.append(255 if is_edge else 0)
        edges.append(edge_row)
    return edges

def box_blur(frame, blur_size):
    blurred_frame = []
    for i in range(len(frame)):
        blurred_row = []
        for j in range(len(frame[i])):
            total, count = 0, 0
            for di in range(-blur_size, blur_size + 1):
                for dj in range(-blur_size, blur_size + 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < len(frame) and 0 <= nj < len(frame[i]):
                        total += frame[ni][nj]
                        count += 1
            blurred_row.append(total // count)
        blurred_frame.append(blurred_row)
    return blurred_frame

def convert_to_grayscale(frame):
    grayscale_frame = []
    for row in frame:
        grayscale_row = []
        for pixel in row:
            grayscale_value = sum(pixel) // len(pixel)
            grayscale_row.append(grayscale_value)
        grayscale_frame.append(grayscale_row)
    return grayscale_frame

def create_reorder_index(N, device):
    new_order = []
    for col in range(N):
        if col % 2 == 0:
            new_order.extend(range(col, N*N, N))
        else:
            new_order.extend(range(col + N*(N-1), col-1, -N))
    return torch.tensor(new_order, device=device)

def reorder_data(data, N):
    assert isinstance(data, torch.Tensor), "require torch.Tensor data"
    device = data.device
    new_order = create_reorder_index(N, device)
    B, t, _, _ = data.shape
    index = new_order.repeat(B, t, 1).unsqueeze(-1)
    reordered_data = torch.gather(data, 2, index.expand_as(data))
    return reordered_data

class XCLIP_Dualist(nn.Module):
    def __init__(
        self, channel_size=768, class_num=1
    ):
        super(XCLIP_Dualist, self).__init__()
        self.encoder = XCLIPVisionModel.from_pretrained("GenVideo/pretrained_weights/xclip")
        blocks = []
        channel = 768
        self.fusing_ratios = 1
        self.patch_nums = (14//self.fusing_ratios)**2
        self.Dualist_configs = DualistConfig(d_model=channel)
        self.Dualist = ResidualBlock(config = self.Dualist_configs)
        self.fc1 = nn.Linear((self.patch_nums+1)*channel, class_num)
        self.fc_norm = nn.LayerNorm(self.patch_nums*channel)
        self.fc_norm2 = nn.LayerNorm(768)
        self.initialize_weights(self.fc1)
        self.dropout = nn.Dropout(p=0.0)

    def initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        b, t, _, h, w = x.shape
        images = x.view(b * t, 3, h, w)
        outputs = self.encoder(images, output_hidden_states=True)
        sequence_output = outputs['last_hidden_state'][:,1:,:]
        _, _, c = sequence_output.shape

        global_feat = outputs['pooler_output'].reshape(b, t, -1)
        global_feat = global_feat.mean(1)
        global_feat = self.fc_norm2(global_feat)

        sequence_output = sequence_output.view(b, t, -1, c)
        _, _, f_w, _ = sequence_output.shape
        f_h, f_w = int(math.sqrt(f_w)), int(math.sqrt(f_w))

        s = f_h//self.fusing_ratios
        sequence_output = sequence_output.view(b, t, self.fusing_ratios, s, self.fusing_ratios, s, c)
        x = sequence_output.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(b*s*s, t, -1, c)
        b_l = b*s*s
        
        x = reorder_data(x, self.fusing_ratios)
        x = x.permute(0, 2, 1, 3).contiguous().view(b_l, -1, c)
        res = self.Dualist(x)

        video_level_features = res.mean(1)
        video_level_features = video_level_features.view(b, -1)
        video_level_features = self.fc_norm(video_level_features)
        video_level_features = torch.cat((global_feat, video_level_features), dim=1)

        pred = self.fc1(video_level_features)
        pred = self.dropout(pred)

        return pred



class CLIP_Dualist(nn.Module):
    def __init__(
        self, channel_size=512, class_num=1
    ):
        super(CLIP_Dualist, self).__init__()
        self.clip_model, preprocess = clip.load('ViT-B-14')
        self.clip_model = self.clip_model.float()
        blocks = []
        channel = 512
        self.fusing_ratios = 2
        self.patch_nums = (14//self.fusing_ratios)**2
        self.Dualist_configs = DualistConfig(d_model=channel)
        self.Dualist = ResidualBlock(config = self.Dualist_configs)
        self.fc1 = nn.Linear(channel*(self.patch_nums+1), class_num)
        self.bn1 = nn.BatchNorm1d(channel)
        self.initialize_weights(self.fc1)

    def initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        b, t, _, h, w = x.shape
        images = x.view(b * t, 3, h, w)
        sequence_output = self.clip_model.encode_image(images)
        _, _, c = sequence_output.shape
        sequence_output = sequence_output.view(b, t, -1, c)

        global_feat = sequence_output.reshape(b, -1, c)
        global_feat = global_feat.mean(1)

        _, _, f_w, _ = sequence_output.shape
        f_h, f_w = int(math.sqrt(f_w)), int(math.sqrt(f_w))

        s = f_h//self.fusing_ratios
        sequence_output = sequence_output.view(b, t, self.fusing_ratios, s, self.fusing_ratios, s, c)
        x = sequence_output.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(b*s*s, t, -1, c)
        b_l = b*s*s
        
        x = reorder_data(x, self.fusing_ratios)
        x = x.permute(0, 2, 1, 3).contiguous().view(b_l, -1, c)
        res = self.Dualist(x)
        video_level_features = res.mean(1)
        video_level_features = video_level_features.view(b, -1)

        video_level_features = torch.cat((global_feat, video_level_features), dim=1)
        x = self.fc1(video_level_features)

        return x

if __name__ == '__main__':
    model = CLIP_Dualist()
    print(model)
