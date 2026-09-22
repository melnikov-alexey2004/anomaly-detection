import numpy as np

from dataset import *
from torch.utils.data import DataLoader
import tqdm.auto as tqdm
import torch

d=BGL(window_size=200, step_size=200, use_in_colab=False)
ds=SuperComputerDataset(d, 10, train_ratio=0.3)
s=JitterBalancedSampler(ds, target_ratio=0.3)


"""

доля: число нормальных окон n=20964, аномальных (возможных стартовых позиций) a=2775, общее n+a=23739
их доли: доля нормальных=88.310, аномальных=11.690
a показывает колво возможных  стартовых позиций для аномального окна: число нормальных окон n=20964, аномальных (возможных стартовых позиций) a=348698, общее n+a=369662
их доли: доля нормальных=5.671, аномальных=94.329
n = число норм окон. a = число анорм. окон после оверсемплинга: число нормальных окон n=20964, аномальных (возможных стартовых позиций) a=348698, общее n+a=369662
их доли: доля нормальных=5.671, аномальных=94.329
после применения ограничителей на размер датасета: число нормальных окон n=20964, аномальных (возможных стартовых позиций) a=348698, общее n+a=369662
их доли: доля нормальных=5.671, аномальных=94.329

"""
# def cf(list_of_item):
#     dts, cnts, labels = zip(*list_of_item)
#     labels = torch.tensor(labels, dtype=torch.int32)
#     time_ia = [0]
#     for i, dt in enumerate(dts[1:]):
#         delta = 0
#         if dt and dts:
#             delta = dt - dts[i]
#         time_ia.append(delta)
#     return torch.tensor(time_ia)

"""
# TypeError: default_collate: batch must contain tensors, numpy arrays,
# numbers, dicts or lists; found <class 'datetime.datetime'
"""

dl=DataLoader(ds, sampler=s, batch_size=32)
for batch in tqdm.tqdm(dl, total=len(dl)):
    pass