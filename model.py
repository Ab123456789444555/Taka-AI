# -*- coding: utf-8 -*-
"""
model.py —— CNN 神经网络定义（纯 CPU 优化轻量残差网络）
======================================================
输入:  (B, 1, 8, 8)  带符号整数矩阵（encode.py 编码，已 /6 归一化）
输出:  policy  (B, 4096)  每个走法（起点64 x 终点64）的打分 logits
       value   (B,)      局面评估 [-1, 1]，+1 当前行棋方大优 / -1 大劣

网络结构（约 140 万参数，2 核 CPU 单次前向仅几毫秒）：
    输入卷积 1->64 (3x3)
    4 x 残差块 [Conv64-BN-ReLU-Conv64-BN + 跳连]
    策略头: Conv64->16(3x3) -> Conv16->4(1x1) -> Flatten256 -> FC256->4096
    价值头: Conv64->8(1x1)  -> Flatten128 -> FC128->64 -> ReLU -> FC64->1 -> tanh
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from encode import POLICY_SIZE


class ResBlock(nn.Module):
    """标准残差块：Conv-BN-ReLU-Conv-BN + 恒等跳连"""

    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(x + out)


class ChessNet(nn.Module):
    """策略 + 价值 双头 CNN"""

    def __init__(self, filters: int = 64, blocks: int = 4):
        super().__init__()
        self.arch = {"filters": filters, "blocks": blocks}

        self.input_conv = nn.Conv2d(1, filters, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(filters)
        self.res_blocks = nn.Sequential(*[ResBlock(filters) for _ in range(blocks)])

        # ---- 策略头 ----
        self.policy_conv = nn.Conv2d(filters, 16, 3, padding=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(16)
        self.policy_sq = nn.Conv2d(16, 4, 1, bias=False)          # 4 x 8 x 8 = 256
        self.policy_fc = nn.Linear(4 * 64, POLICY_SIZE)           # 256 -> 4096

        # ---- 价值头 ----
        self.value_sq = nn.Conv2d(filters, 8, 1, bias=False)      # 8 x 8 x 8 = 128
        self.value_fc1 = nn.Linear(8 * 64, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: (B,1,8,8) 已归一化
        h = F.relu(self.input_bn(self.input_conv(x)))
        h = self.res_blocks(h)

        p = F.relu(self.policy_bn(self.policy_conv(h)))
        p = F.relu(self.policy_sq(p))
        logits = self.policy_fc(p.flatten(1))                     # (B,4096)

        v = F.relu(self.value_sq(h).flatten(1))
        v = torch.tanh(self.value_fc2(F.relu(self.value_fc1(v))))
        return logits, v.squeeze(1)

    @torch.no_grad()
    def predict(self, x_np):
        """单批前向。输入 (B,1,8,8) numpy，返回 (logits np, value np)。"""
        self.eval()
        x = torch.from_numpy(x_np)
        logits, value = self(x)
        return logits.numpy(), value.numpy()


def build_model(arch: dict | None = None) -> ChessNet:
    """按结构参数构建网络（客户端加载 checkpoint 时用同一套参数）。"""
    arch = arch or {}
    return ChessNet(arch.get("filters", 64), arch.get("blocks", 4))


def normalize_input(mat: np.ndarray) -> np.ndarray:
    """矩阵数值 /6 归一化到 [-1, 1] 区间，网络训练更稳定。"""
    return (mat / 6.0).astype(np.float32)
