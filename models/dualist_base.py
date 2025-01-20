import math
from dataclasses import dataclass
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from pscan import pscan

@dataclass
class DualistConfig:
    d_model: 768
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16
    expand_factor: int = 2
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    drop_prob: float = 0.1

    bias: bool = False
    conv_bias: bool = True
    biDualist: bool = True

    pscan: bool = True

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

def motion_blur_frame(frame, blur_radius):
    blurred_frame = []
    for row in frame:
        blurred_row = []
        for i in range(len(row)):
            neighbors = row[max(0, i - blur_radius):min(len(row), i + blur_radius + 1)]
            blurred_row.append(sum(neighbors) // len(neighbors))
        blurred_frame.append(blurred_row)
    return blurred_frame

def resize_frame(frame, scale):
    resized_frame = []
    for row in frame:
        resized_row = []
        for pixel in row:
            resized_row.extend([pixel] * scale)
        for _ in range(scale):
            resized_frame.append(resized_row)
    return resized_frame

def pixel_intensity_histogram(frame):
    histogram = {}
    for row in frame:
        for pixel in row:
            histogram[pixel] = histogram.get(pixel, 0) + 1
    return histogram

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

class ResidualBlock(nn.Module):
    def __init__(self, config: DualistConfig):
        super().__init__()

        self.mixer = DualistBlock(config)
        self.norm = RMSNorm(config.d_model)
        self.drop_path = DropPath(drop_prob=config.drop_prob)

    def forward(self, x):
        output = self.drop_path(self.mixer(self.norm(x))) + x
        return output
    
    def step(self, x, cache):
        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache

class DualistBlock(nn.Module):
    def __init__(self, config: DualistConfig):
        super().__init__()

        self.config = config

        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner, 
                              kernel_size=config.d_conv, bias=config.conv_bias, 
                              groups=config.d_inner,
                              padding=config.d_conv - 1)
        
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(config.d_inner))

        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        self.biDualist = config.biDualist

        if self.biDualist:
            A_b = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
            self.A_b_log = nn.Parameter(torch.log(A_b))

            self.conv1d_b = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner, 
                              kernel_size=config.d_conv, bias=config.conv_bias, 
                              groups=config.d_inner,
                              padding=config.d_conv - 1)

            self.x_proj_b = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)
            self.dt_proj_b = nn.Linear(config.dt_rank, config.d_inner, bias=True)
            self.D_b = nn.Parameter(torch.ones(config.d_inner))

    def forward(self, x):
        _, L, _ = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)

        x = F.silu(x)
        y = self.ssm(x)

        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)

        return output
    
    def ssm(self, x):
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        deltaBC = self.x_proj(x)

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        if self.config.pscan:
            y = self.selective_scan(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)

        if self.biDualist:
            x_b = x.flip([-1])
            A_b = -torch.exp(self.A_b_log.float())
            D_b = self.D_b.float()
            deltaBC_b = self.x_proj_b(x_b)
            delta_b, B_b, C_b = torch.split(deltaBC_b, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
            delta_b = F.softplus(self.dt_proj_b(delta_b))
            if self.config.pscan:
                y_b = self.selective_scan(x_b, delta_b, A_b, B_b, C_b, D_b)
            else:
                y_b = self.selective_scan_seq(x_b, delta_b, A_b, B_b, C_b, D_b)
            y_b = y_b.flip([-1])
            y = y + y_b
        return y
    
    def selective_scan(self, x, delta, A, B, C, D):
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)

        BX = deltaB * (x.unsqueeze(-1))
        
        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)

        y = y + D * x

        return y
    
    def selective_scan_seq(self, x, delta, A, B, C, D):
        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)

        BX = deltaB * (x.unsqueeze(-1))

        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)
            
        hs = torch.stack(hs, dim=1)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)

        y = y + D * x

        return y

    def step(self, x, cache):
        h, inputs = cache
        
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=1)

        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv-1]

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)

        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2)
        cache = (h, inputs)
        
        return output, cache

    def ssm_step(self, x, h):
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        deltaBC = self.x_proj(x)

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1)

        BX = deltaB * (x.unsqueeze(-1))

        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)

        h = deltaA * h + BX

        y = (h @ C.unsqueeze(-1)).squeeze(2)

        y = y + D * x

        return y, h.squeeze(1)

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
