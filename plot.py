import matplotlib.pyplot as plt
import numpy as np

# 设置字体的属性
# plt.rcParams["font.sans-serif"] = "Arial Unicode MS"
plt.rcParams["font.sans-serif"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots() # 创建图实例
x = np.linspace(0,1000,1000) # 创建x的取值范围
y1 = np.loadtxt("D:\Desktop\智能交通\home_work\output\work5\hiddensize256_layersize150_activationSigmoid_optimizerSGD_lossSmoothL1Loss_RegularizationNone_dropoutFalse/loss.txt")
ax.plot(x, y1, label='SGD') # 作y1 = x 图，并标记此线名为linear
y2 = np.loadtxt("D:\Desktop\智能交通\home_work\output\work5\hiddensize256_layersize150_activationSigmoid_optimizerMomentum_lossSmoothL1Loss_RegularizationNone_dropoutFalse/loss.txt")
ax.plot(x, y2, label='Momentum') #作y2 = x^2 图，并标记此线名为quadratic
y3 = np.loadtxt("D:\Desktop\智能交通\home_work\output\work1\hiddensize256_layersize150_activationRelu_optimizerAdam_lossSmoothL1Loss_RegularizationNone_dropoutFalse/loss.txt")
ax.plot(x, y3, label='Adam') # 作y3 = x^3 图，并标记此线名为cubic
y4 = np.loadtxt("D:\Desktop\智能交通\home_work\output\work5\hiddensize256_layersize150_activationSigmoid_optimizerNAG_lossSmoothL1Loss_RegularizationNone_dropoutFalse/loss.txt")
ax.plot(x, y4, label='NAG') #作y2 = x^2 图，并标记此线名为quadratic
y5 = np.loadtxt("D:\Desktop\智能交通\home_work\output\work5\hiddensize256_layersize150_activationSigmoid_optimizerAdaGrad_lossSmoothL1Loss_RegularizationNone_dropoutFalse/loss.txt")
ax.plot(x, y5, label='AdaGrad') #作y2 = x^2 图，并标记此线名为quadratic
y6 = np.loadtxt("D:\Desktop\智能交通\home_work\output\work5\hiddensize256_layersize150_activationSigmoid_optimizerRMSProp_lossSmoothL1Loss_RegularizationNone_dropoutFalse/loss.txt")
ax.plot(x, y6, label='AdaGrad') #作y2 = x^2 图，并标记此线名为quadratic
y7 = np.loadtxt("D:\Desktop\智能交通\home_work\output\work5\hiddensize256_layersize150_activationSigmoid_optimizerAdaDelta_lossSmoothL1Loss_RegularizationNone_dropoutFalse/loss.txt")
ax.plot(x, y7, label='AdaDelta') #作y2 = x^2 图，并标记此线名为quadratic
ax.set_xlabel('Epoch') #设置x轴名称 x label
ax.set_ylabel('Loss') #设置y轴名称 y label
ax.set_title('不同激活函数对比图') #设置图名为Simple Plot
ax.legend() #自动检测要在图例中显示的元素，并且显示
plt.savefig('D:\Desktop\智能交通\home_work\output\work2/result.png')
plt.show() #图形可视化
