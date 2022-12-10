#%%
import torch as tc
import torch.nn as nn
import matplotlib.pyplot as plt

# pool of square window of size=3, stride=2
m = nn.AvgPool2d(kernel_size=1, stride=1)
input = tc.randn(1, 3, 4, 4)
output = m(input)
print(output.shape)
# https://matplotlib.org/3.1.1/gallery/subplots_axes_and_figures/figure_title.html