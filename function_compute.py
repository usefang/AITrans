import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.integrate import dblquad
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
# lambda = 1;
# g = @(u,v) exp(-u.^2/2).*exp(-v.^2/2)/3.5970;
# G = @(u) exp(-u.^2/2);
# gf = @(u,v) exp(lambda-G(u)).*g(u,v); % 被积函数
# % 求F并画图
# x = linspace(-10,10,41);
# y = linspace(-10,10,41);
# [X,Y] = meshgrid(x,y);
# [i,j] = meshgrid(1:length(x), 1:length(y));
# F = @(x,y) exp(-lambda)*(1+lambda*integral2(gf, -inf, x, -inf, y));
# F0 = arrayfun(@(i,j) F(X(i,j),Y(i,j)), i, j);
# mesh(X,Y,F0)

def g(u, v):
    return np.e ** (-u ** 2 / 2) * np.e ** (-v ** 2 / 2) / 3.5970

def G(u):
    return np.e ** (-u ** 2 / 2)

def gf(u, v):
    return np.e ** (1 - G(u)) * g(u, v)

def Function(x, y):
    return np.e ** (-1) * (1 + dblquad(gf, -np.inf, x, -np.inf, y)[0])



figure = plt.figure()
ax = Axes3D(figure)#设置图像为三维格式
X = np.arange(0,5,0.1)
Y = np.arange(0,5,0.1)#X,Y的范围
# Z = []
# count = 0
# time_begin = time.time()
# for x in X:
#     for y in Y:
#         Z.append(Function(x, y))
#         count += 1
#         if count % 1000 == 0:
#             time_end = time.time()
#             print(f'count is {count}, and time is {time_end - time_begin}')
#             time_begin = time.time()
# np.savetxt('x_1.txt', X)
# np.savetxt('y_1.txt', Y)
# np.savetxt('z_1.txt', Z)
#
# print('Z is done')
# Z = np.array(Z)
# X = np.loadtxt('x.txt')
# Y = np.loadtxt('y.txt')
Z = np.loadtxt('z_noisy_1.txt')

Z = Z.reshape((50,50))
X,Y = np.meshgrid(X,Y)#绘制网格

ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='rainbow')
#绘制3D图，后面的参数为调节图像的格式
plt.show()#展示图片`
