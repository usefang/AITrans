# --coding:utf-8--
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.utils.data as Data
import torch.optim as optim
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import time

def data_process():
    Z = np.loadtxt('z_1.txt')

    noisy_index = np.random.choice(2500, 100, replace=False)
    print(noisy_index[0])
    for i in noisy_index:
        Z[i] = Z[i] + 0.1*np.random.random()
    np.savetxt('z_noisy_1.txt', Z)

class my_network(nn.Module):
    def __init__(self, config):
        super(my_network, self).__init__()

        # 根据config选择不同的激活函数
        if config.activation == 'Relu':
            self.activation = nn.ReLU()
        elif config.activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif config.activation == 'Tanh':
            self.activation = nn.Tanh()
        elif config.activation == 'Swish':
            self.activation = F.hardswish
        elif config.activation == 'Mish':
            self.activation = nn.Mish()
        elif config.activation == 'LogSigmoid':
            self.activation = nn.LogSigmoid()
        elif config.activation == 'PReLU':
            self.activation = nn.PReLU()
        elif config.activation== 'ELU':
            self.activation = nn.ELU()

        if config.loss == 'MSE':
            self.loss_fun = F.mse_loss
        elif config.loss == 'L1Loss':
            self.loss_fun = F.l1_loss
        elif config.loss == 'SmoothL1Loss':
            self.loss_fun = F.smooth_l1_loss

        self.layers = nn.Sequential()
        self.layers.add_module('input', nn.Linear(2, config.hidden_size) )
        self.layers.add_module('activation', self.activation)
        for i in range(config.layer_size):
            self.layers.add_module('hidden', nn.Linear( config.hidden_size, config.hidden_size ) )
            if config.drop_out:
                self.layers.add_module('dropout', nn.Dropout(0.5) )
            self.layers.add_module('activation', self.activation)
        self.layers.add_module('out', nn.Linear( config.hidden_size, 1 ) )

    def forward(self, x, train = True):
        out = self.layers(x[:,:2])
        if train:
            return self.loss_fun(out, x[:, 2])
        else:
            return out

def train(model, data_loader, optimizer):
    # train
    model.train()
    train_loss = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i, x in enumerate(data_loader):
        x = x.to(device)
        loss = model(x, train=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t = torch.Tensor.cpu(loss).detach().numpy()
        train_loss.append(t)

    return train_loss


def evaluate(model, data_loader):
    # train
    model.eval()
    print_freq = 50
    eval_loss = []
    for i, x in enumerate(data_loader):
        loss = model(x, train=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t = np.array(loss)
        eval_loss.append(t)

    return eval

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--layer_size', default=150, type=int)
    parser.add_argument('--activation', default='Relu', type=str)
    parser.add_argument('--optimizer', default='Adam', type=str)
    parser.add_argument('--loss', default='SmoothL1Loss', type=str)
    parser.add_argument('--Regularization', default='None', type=str)
    parser.add_argument('--drop_out', default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = my_network(args).to(device)

    x = np.loadtxt('x_1.txt')
    y = np.loadtxt('y_1.txt')
    z = np.loadtxt('z_noisy_1.txt')
    new_x = []
    for x_t in x:
        for y_t in y:
            new_x.append( [x_t, y_t ])
    new_x = np.array(new_x)

    x = torch.tensor(new_x, dtype=torch.float)
    z = torch.tensor(z, dtype=torch.float).unsqueeze(1)
    X = torch.cat( [ x, z ],dim=1)
    train_size = int(0.99 * X.shape[0])
    test_size = X.shape[0] - train_size
    train_dataset, test_dataset = Data.random_split(X, [train_size, test_size])

    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )

    test_loader = Data.DataLoader(
        dataset = test_dataset,
        batch_size = 32,
        shuffle = True,
        num_workers = 0,
    )

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr = 1e-5, momentum=0, dampening=0.5, weight_decay=0.001 , nesterov=False)
    elif args.optimizer == 'Momentum':
        optimizer = optim.SGD(model.parameters(), lr = 1e-5, momentum=0.9, dampening=0.5, weight_decay=0.001, nesterov=False)
    elif args.optimizer == 'NAG':
        optimizer = optim.SGD(model.parameters(), lr = 1e-5, momentum=0.9, dampening=0, weight_decay=0.001, nesterov=True)
    elif args.optimizer == 'AdaGrad':
        optimizer = optim.Adagrad(model.parameters(), lr=1e-5, lr_decay=0, weight_decay=0.001, initial_accumulator_value=0)
    elif args.optimizer == 'RMSProp':
        optimizer = optim.RMSprop(model.parameters(), lr=1e-5, alpha=0.99, eps=1e-08, momentum=0, weight_decay=0.001, centered=False)
    elif args.optimizer == 'AdaDelta':
        optimizer = optim.Adadelta(model.parameters(), lr=1e-5, rho=0.9, eps=1e-6, weight_decay=0.001)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
    elif args.optimizer == 'Nadam':
        optimizer = optim.NAdam(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, momentum_decay=0.004)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.001)
    print("Start training")
    Total_loss = []
    path = f"output/hiddensize{args.hidden_size}_layersize{args.layer_size}_activation{args.activation}_optimizer{args.optimizer}_loss{args.loss}_Regularization{args.Regularization}_dropout{args.drop_out}"
    if not os.path.exists(path):
        os.makedirs(path)
    for epoch in range(1000):
        loss = train(model, train_loader, optimizer)
        avg_train_loss = np.mean(loss)
        print(f'Epoch : {epoch} | train_loss : {format(avg_train_loss, ".4f")}')
        Total_loss.append(avg_train_loss)
        # evaluate(model, test_loader)
        if epoch % 50 == 0:
            model.eval()
            figure = plt.figure()
            ax = Axes3D(figure)  # 设置图像为三维格式
            X = np.arange(0,5,0.1)
            Y = np.arange(0,5,0.1)#X,Y的范围
            x = x.to(device)
            Z =  torch.Tensor.cpu(model(x, train=False)).detach().numpy()
            Z = Z.reshape((50, 50))
            X, Y = np.meshgrid(X, Y)  # 绘制网格

            ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
            # 绘制3D图，后面的参数为调节图像的格式
            plt.savefig(f'{path}/Epoch{epoch}.jpg')
            # plt.show()  # 展示图片`

    total_loss = np.array(Total_loss)
    np.savetxt(f"{path}/loss.txt", total_loss)
    end_time = time.time()
    print(f"total_time is {end_time - start_time} s | { ( end_time - start_time ) / 60} min | { ( end_time - start_time) / 3600} h")