# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 17:20:05 2022
@author: Jonas Peter
"""
import sys
import os
import time
#insert path of parent folder to import helpers
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import torch
import torch.nn as nn
from torch.autograd import Variable
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
from helpers import *
import warnings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train = True

class Net(nn.Module):
    def __init__(self, num_layers, layers_size):
        super(Net, self).__init__()
        assert num_layers == len(layers_size)
        self.linears = nn.ModuleList([nn.Linear(1, layers_size[0])])
        self.linears.extend([nn.Linear(layers_size[i-1], layers_size[i])
                            for i in range(1, num_layers)])
        self.linears.append(nn.Linear(layers_size[-1], 2))

    def forward(self, x):  # ,p,px):
        # torch.cat([x,p,px],axis=1) # combined two arrays of 1 columns each to one array of 2 columns
        x = torch.unsqueeze(x, 1)
        for i in range(len(self.linears)-1):
            x = torch.tanh(self.linears[i](x))
        output = self.linears[-1](x)
        return output.reshape(-1, 1)
##
# Hyperparameter
learning_rate = 0.01


# Definition der Parameter des statischen Ersatzsystems
Lb = float(input('Länge des Kragarms [m]: '))
EI = 21
K = 5/6
G = 80
A = 100

#Normierungsfaktor (siehe Kapitel 10.3)
normfactor = 10/((11*Lb**5)/(120*EI))

# ODE als Loss-Funktion, Streckenlast
Ln = 0 #float(input('Länge Einspannung bis Anfang der ' + str(i + 1) + '. Streckenlast [m]: '))
Lq = Lb # float(input('Länge der ' + str(i + 1) + '. Streckenlast [m]: '))
s = str(normfactor)+"*x"#input(str(i + 1) + '. Streckenlast eingeben: ')

def h(x):
    return eval(s)

#Netzwerk für Biegung
def f(x, net):
    vb = net(x)[0::2]
    _, _, _, vb_xxxx = deriv(vb, x, 4)
    ode = vb_xxxx + (h(x - Ln))/EI
    return ode

#Netzwerk für Schub
def g(x, net):
    vs = net(x)[1::2]
    _, vs_xx = deriv(vs, x, 2)
    #0 = vs'' - q(x)/KAG
    ode = vs_xx - (h(x - Ln)) / (K * A * G)
    return ode

x = np.linspace(0, Lb, 1000)
qx = h(x) * (x <= (Ln + Lq)) * (x >= Ln)

Q0 = integrate.cumtrapz(qx, x, initial=0)

qxx = qx * x

M0 = integrate.cumtrapz(qxx, x, initial=0)

#Die nächsten Zeilen bis Iterationen geben nur die Biegelinie aus welche alle 10 Iterationen refreshed wird während des Lernens, man kann also den Lernprozess beobachten

def gridSearch(num_layers, layers_size):
    start = time.time()
    net = Net(num_layers, layers_size)
    net = net.to(device)
    mse_cost_function = torch.nn.MSELoss()  # Mean squared error
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # Der Scheduler sorgt dafür, dass die Learning Rate auf einem Plateau mit dem factor multipliziert wird
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, verbose=True, factor=0.75)
    if train:
        y1 = net(myconverter(x, False))
        fig = plt.figure()
        plt.grid()
        ax1 = fig.add_subplot()
        ax1.set_xlim([0, Lb])
        ax1.set_ylim([-10, 0])
        net_out_plot = myconverter(y1)
        line1, = ax1.plot(x, net_out_plot[0::2] + net_out_plot[1::2])
        plt.title(f'{num_layers =}, {layers_size =}')
        plt.show(block=False)
        pt_x = myconverter(x)
        f_anal_b = (-1 / 120 * normfactor * x ** 5 + 1 / 6 * Q0[-1] * x ** 3 - M0[-1] / 2 * x ** 2) / EI
        f_anal_s = (1 / 6 * normfactor * x ** 3 - Q0[-1] * x) / (K * A * G)
        lossfactor = ((-1 / 120 * normfactor * Lb ** 5 + 1 / 6 * Q0[-1] * Lb ** 3 - M0[-1] / 2 * Lb ** 2) / EI) / (
                    (1 / 6 * normfactor * Lb ** 3 - Q0[-1] * Lb) / (K * A * G))

    iterations = 100000
    for epoch in range(iterations):
        if not train: break
        optimizer.zero_grad()  # to make the gradients zero
        x_bc = np.linspace(0, Lb, 500)
        pt_x_bc = torch.unsqueeze(myconverter(x_bc), 1)

        x_collocation = np.random.uniform(low=0.0, high=Lb, size=(250 * int(Lb), 1))
        all_zeros = np.zeros((250 * int(Lb), 1))

        pt_x_collocation = torch.unsqueeze(myconverter(x_collocation), 1)
        f_out_B = f(pt_x_collocation, net)
        f_out_S = g(pt_x_collocation, net)

        # Randbedingungen
        net_bc_out_B = net(pt_x_bc)[0::2]
        net_bc_out_S = net(pt_x_bc)[1::2]
        # ei --> Werte, die minimiert werden müssen
        vb_x, vb_xx, vb_xxx = deriv(net_bc_out_B, pt_x_bc, 3)
        vs_x = deriv(net_bc_out_S, pt_x_bc, 1)

        # RB für Biegung
        BC3 = net_bc_out_B[0]
        BC6 = vb_xxx[0] - Q0[-1] / EI
        BC7 = vb_xxx[-1]
        BC8 = vb_xx[0] + M0[-1] / EI
        BC9 = vb_xx[-1]
        BC10 = vb_x[0]

        # RB für Schub
        BC2 = net_bc_out_S[0]
        BC4 = vs_x[0] + Q0[-1] / (K * A * G)
        BC5 = vs_x[-1]

        mse_Gamma_B = errsum(mse_cost_function, BC3, 1 / normfactor * BC6, BC7, 1 / normfactor * BC8, BC9, BC10)
        mse_Gamma_S = errsum(mse_cost_function, BC2, 1 / normfactor * BC4, BC5)
        mse_Omega_B = errsum(mse_cost_function, f_out_B)
        mse_Omega_S = errsum(mse_cost_function, f_out_S)

        loss_B = mse_Gamma_B + mse_Omega_B
        loss_S = mse_Gamma_S + mse_Omega_S

        loss = loss_B + lossfactor * loss_S
        loss.backward()

        optimizer.step()
        scheduler.step(loss)
        with torch.autograd.no_grad():
            if epoch % 10 == 9:
                print(epoch, "Traning Loss:", loss.data)
                plt.grid()
                net_out = myconverter(net(pt_x))
                net_out_B = net_out[0::2]
                net_out_S = net_out[1::2]
                err_b = np.linalg.norm(net_out_B - f_anal_b, 2)
                err_s = np.linalg.norm(net_out_S - f_anal_s, 2)
                print(f'Error = {err_s =},{err_b =}')
                if err_b < 0.1 * Lb and err_s < Lb / lossfactor:
                    print(f"Die L^2 Norm der Fehler ist {err_b = },{ err_s = }.\nStoppe Lernprozess")
                    break
                line1.set_ydata(net_out_B + net_out_S)
                fig.canvas.draw()
                fig.canvas.flush_events()
##
# GridSearch
time_elapsed = []
num_layers= 1
for j in range(650, 1000, 10):  # Wieviele zufällige layers_size pro num_layers
    layers_size = [j]
    time_elapsed.append(
    (num_layers, layers_size, gridSearch(num_layers, layers_size)))
    plt.close()

with open(r'geordnet5m_650-1000.txt', 'w') as fp:
    for item in time_elapsed:
        # write each item on a new line
        fp.write(f'{item} \n')

# os.system('shutdown /s /t 1')