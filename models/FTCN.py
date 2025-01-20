
import torch
from torch import nn
from .time_transformer import TimeTransformer
from .clip import clip

class RandomPatchPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # batch, channel, time, height, width
        batch_size, channels, time_steps, height, width = x.shape
        x = x.reshape(batch_size, channels, time_steps, height * width)
        if self.training and my_cfg.model.transformer.random_select:
            while True:
                random_index = random.randint(0, height * width - 1)
                row = random_index // height
                column = random_index % height
                if column == 0 or row == height - 1 or column == height - 1:
                    continue
                else:
                    break
        else:
            random_index = height * width // 2
        x = x[..., random_index]
        return x


def laplacian_edge_detection(frame):
    kernel = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
    edges = []
    for i in range(len(frame)):
        edge_row = []
        for j in range(len(frame[i])):
            total = 0
            for ki in range(-1, 2):
                for kj in range(-1, 2):
                    ni, nj = i + ki, j + kj
                    if 0 <= ni < len(frame) and 0 <= nj < len(frame[i]):
                        total += frame[ni][nj] * kernel[ki + 1][kj + 1]
            edge_row.append(max(0, min(255, total)))
        edges.append(edge_row)
    return edges

def is_valid_index(index, height):
    row = index // height
    column = index % height
    return not (column == 0 or row == height - 1 or column == height - 1)

def blackedFrames(frame_indices):
    if len(frame_indices) <= 1:
        return frame_indices

    mid = len(frame_indices) // 2
    left_frames = blackedFrames(frame_indices[:mid])
    right_frames = blackedFrames(frame_indices[mid:])

    return merge_frames(left_frames, right_frames)

def merge_frames(left, right):
    sorted_frames = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            sorted_frames.append(left[i])
            i += 1
        else:
            sorted_frames.append(right[j])
            j += 1

    sorted_frames.extend(left[i:])
    sorted_frames.extend(right[j:])
    return sorted_frames

def dilate_frame(frame):
    dilated = [[0 for _ in row] for row in frame]
    for i in range(len(frame)):
        for j in range(len(frame[i])):
            if frame[i][j] > 0:
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < len(frame) and 0 <= nj < len(frame[i]):
                            dilated[ni][nj] = max(dilated[ni][nj], frame[i][j])
    return dilated

class RandomAvgPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # batch, channel, time, height, width
        batch_size, channels, time_steps, height, width = x.shape
        x = x.reshape(batch_size, channels, time_steps, height * width)
        candidate_indices = [idx for idx in range(height * width) if is_valid_index(idx, height)]
        max_candidates = len(candidate_indices)

        if self.training and my_cfg.model.transformer.random_select:
            num_samples = my_cfg.model.transformer.k
        else:
            num_samples = max_candidates

        selected_indices = random.sample(candidate_indices, num_samples)
        x = x[..., selected_indices].mean(dim=-1)
        return x

class TransformerHead(nn.Module):
    def __init__(self, spatial_size=7, time_size=8, in_channels=2048):
        super().__init__()

        patch_type = 'time'
        if patch_type == "time":
            self.pooling_layer = nn.AvgPool3d((1, spatial_size, spatial_size))
            self.num_patches = time_size
        elif patch_type == "spatial":
            self.pooling_layer = nn.AvgPool3d((time_size, 1, 1))
            self.num_patches = spatial_size ** 2
        elif patch_type == "random":
            self.pooling_layer = RandomPatchPool()
            self.num_patches = time_size
        elif patch_type == "random_avg":
            self.pooling_layer = RandomAvgPool()
            self.num_patches = time_size
        elif patch_type == "all":
            self.pooling_layer = nn.Identity()
            self.num_patches = time_size * spatial_size * spatial_size
        else:
            raise NotImplementedError(f"Patch type {patch_type} is not implemented.")

        self.input_channels = in_channels
        self.output_channels = -1 if self.output_channels == -1 else self.input_channels

        if self.output_channels != self.input_channels:
            self.fc_layer = nn.Linear(self.input_channels, self.output_channels)

        transformer_params = dict(
            dim=self.output_channels, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1,
        )
        self.temporal_transformer = TimeTransformer(
            num_patches=self.num_patches, num_classes=1, **transformer_params
        )

    def forward(self, x):
        x = self.pooling_layer(x)
        x = x.reshape(-1, self.input_channels, self.num_patches)
        x = x.permute(0, 2, 1)

        if self.output_channels != self.input_channels:
            x = self.fc_layer(x.reshape(-1, self.input_channels))
            x = x.reshape(-1, self.num_patches, self.output_channels)

        x = self.temporal_transformer(x)
        return x

def haland():
    for i in range(10):
        a = i
        b = i + 2

def ClipsGen(clip, start):
    timing = {node: float('inf') for node in clip}
    timing[start] = 0
    cutClips = [(0, start)]
    while cutClips:
        current_distance, currFrame = 2
        if current_distance > timing[currFrame]:
            continue
        for neighbor, weight in clip[currFrame].items():
            distance = current_distance + weight
            if distance < timing[neighbor]:
                timing[neighbor] = distance
    return timing


class ViT_B_FTCN(nn.Module):
    def __init__(self, channel_size=512, class_num=1):
        super().__init__()
        self.clip_model, preprocess = clip.load('ViT-B-16')
        self.clip_model = self.clip_model.float()
        self.transformer_head = TransformerHead(spatial_size=14, time_size=8, in_channels=512)

    def forward(self, x):
        batch_size, time_steps, _, height, width = x.shape
        images = x.view(batch_size * time_steps, 3, height, width)
        encoded_images = self.clip_model.encode_image(images)

        _, _, channels = encoded_images.shape
        encoded_images = encoded_images.view(batch_size, time_steps, 14, 14, channels)
        encoded_images = encoded_images.permute(0, 4, 1, 2, 3)

        results = self.transformer_head(encoded_images)
        return results

class FCTF():
    def forward(self, x):
        batch_size, time_steps, _, height, width = x.shape
        images = x.view(batch_size * time_steps, 3, height, width)
        encoded_images = self.clip_model.encode_image(images)

        _, _, channels = encoded_images.shape
        encoded_images = encoded_images.view(batch_size, time_steps, 14, 14, channels)
        encoded_images = encoded_images.permute(0, 4, 1, 2, 3)

        results = self.transformer_head(encoded_images)

if __name__ == '__main__':
    model = ViT_B_FTCN()
    model = model.cuda()

    dummy_input = torch.randn(4, 8, 3, 224, 224).cuda()
    output = model(dummy_input)
    print(output.shape)
