# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 17:20:05 2022
@author: Jonas Peter
"""
import scipy.integrate
import torch
import torch.nn as nn
from torch.autograd import Variable
import scipy as sp
import scipy.integrate as integrate
from scipy.integrate import quad
import scipy.special as special
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splrep, splev
import math
import time
##
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
        inputs = x
        for i in range(len(self.linears)-1):
            x = torch.tanh(self.linears[i](x))
        output = self.linears[-1](x)
        return output.reshape(-1, 1)
##
choice_load = input("Möchtest du ein State_Dict laden? y/n")
if choice_load == 'y':
    train=False
    filename = input("Welches State_Dict möchtest du laden?")
    net = Net()
    net = net.to(device)
    net.load_state_dict(torch.load('C:\\Users\\Administrator\\Desktop\\Uni\\Master\\Masterarbeit\\Timoshenko NN Kragarm 5.3\\saved_data\\'+filename))
    net.eval()
##
# Hyperparameter
learning_rate = 0.01

# Definition der Parameter des statischen Ersatzsystems

Lb = float(input('Länge des Kragarms [m]: '))
E = 21#float(input('E-Modul des Balkens [10^6 kNcm²]: '))
h = 10#float(input('Querschnittshöhe des Balkens [cm]: '))
b = 10#float(input('Querschnittsbreite des Balkens [cm]: '))
A = h*b
I = (b*h**3)/12
EI = E*I*10**-3
G = 80#float(input('Schubmodul des Balkens [GPa]: '))
LFS = 1#int(input('Anzahl Streckenlasten: '))
K = 5 / 6  # float(input(' Schubkoeffizient '))
Ln = np.zeros(LFS)
Lq = np.zeros(LFS)
s = [None] * LFS
x = np.linspace(0, Lb, 1000)
normfactor = 10/(Lb**3/(K*A*G)+(11*Lb**5)/(120*EI))

for i in range(LFS):
    # ODE als Loss-Funktion, Streckenlast
    Ln[i] = 0#float(input('Länge Einspannung bis Anfang der ' + str(i + 1) + '. Streckenlast [m]: '))
    Lq[i] = Lb#float(input('Länge der ' + str(i + 1) + '. Streckenlast [m]: '))
    s[i] = str(normfactor)+"*x"#input(str(i + 1) + '. Streckenlast eingeben: ')
def h(x, j):
    return eval(s[j])

#Netzwerk für Biegung
def f(x, net):
    v_b = net(x)[0::2]
    v_b_x = torch.autograd.grad(v_b, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(v_b))[0]
    v_b_xx = torch.autograd.grad(v_b_x, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(v_b))[0]
    v_b_xxx = torch.autograd.grad(v_b_xx, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(v_b))[0]
    v_b_xxxx = torch.autograd.grad(v_b_xxx, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(v_b))[0]
    ode = 0
    for i in range(LFS):
        #0 = vb'''' + q(x)/EI
        ode += v_b_xxxx + (h(x - Ln[i], i)) / EI * (x <= (Ln[i] + Lq[i])) * (x >= Ln[i])
    return ode

#Netzwerk für Schub
def g(x, net):
    v_s = net(x)[1::2]
    v_s_x = torch.autograd.grad(v_s, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(v_s))[0]
    v_s_xx = torch.autograd.grad(v_s_x, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(v_s))[0]
    #0 = vs'' - q(x)/KAG
    ode = v_s_xx - (h(x - Ln[i], i)) / (K * A * G) * (x <= (Ln[i] + Lq[i])) * (x >= Ln[i])
    return ode

x = np.linspace(0, Lb, 1000)
pt_x = torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=True).to(device), 1)
qx = np.zeros(1000)
for i in range(LFS):
    qx = qx + (h(torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=False).to(device), 1) - Ln[i], i).cpu().detach().numpy()).squeeze() * (x <= (Ln[i] + Lq[i])) * (x >= Ln[i])

Q0 = integrate.cumtrapz(qx, x, initial=0)
#Q0 = Q(0) = int(q(x)), über den ganzen Balken
qxx = qx * x
#M0 = M(0) = int(q(x)*x), über den ganzen Balken
M0 = integrate.cumtrapz(qxx, x, initial=0)

#Die nächsten Zeilen bis Iterationen geben nur die Biegelinie aus welche alle 10 Iterationen refreshed wird während des Lernens, man kann also den Lernprozess beobachten

f_anal_b = (-1/120 * normfactor * pt_x**5 + 1/6 * Q0[-1] * pt_x**3 - M0[-1]/2 * pt_x**2)/EI
f_anal_s = (1/6 * normfactor * pt_x**3 - Q0[-1] * pt_x)/(K*A*G)
lossfactor = ((-1/120 * normfactor * Lb**5 + 1/6 * Q0[-1] * Lb**3 - M0[-1]/2 * Lb**2)/EI)/((1/6 * normfactor * Lb**3 - Q0[-1] * Lb)/(K*A*G))
def gridSearch(num_layers, layers_size):
    start = time.time()
    net = Net(num_layers, layers_size)
    net = net.to(device)
    mse_cost_function = torch.nn.MSELoss()  # Mean squared error
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # Der Scheduler sorgt dafür, dass die Learning Rate auf einem Plateau mit dem factor multipliziert wird
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, verbose=True, factor=0.75)
    if train:
        # + net_S(torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=False).to(device), 1))
        y1 = net(torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=False).to(device), 1))[0::2] + net(
            torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=False).to(device), 1))[1::2]
        fig = plt.figure()
        plt.title(f'{num_layers =}, {layers_size =}')
        plt.grid()
        ax1 = fig.add_subplot()
        ax1.set_xlim([0, Lb])
        ax1.set_ylim([-20, 0])
        # ax2.set_
        line1, = ax1.plot(x, y1.cpu().detach().numpy())
        plt.show(block=False)

    iterations = 1000000
    for epoch in range(iterations):
        optimizer.zero_grad()  # to make the gradients zero
        x_bc = np.linspace(0, Lb, 500)
        pt_x_bc = torch.unsqueeze(Variable(torch.from_numpy(
            x_bc).float(), requires_grad=True).to(device), 1)
        # unsqueeze wegen Kompatibilität
        pt_zero = Variable(torch.from_numpy(np.zeros(1)).float(),
                           requires_grad=False).to(device)

        x_collocation = np.random.uniform(
            low=0.0, high=Lb, size=(250 * int(Lb), 1))
        #x_collocation = np.linspace(0, Lb, 1000*int(Lb))
        all_zeros = np.zeros((250 * int(Lb), 1))

        pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
        pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)
        ode_B = f(pt_x_collocation, net)
        ode_S = g(pt_x_collocation, net)

        # Randbedingungen
        net_bc_out_B = net(pt_x_bc)[0::2]
        net_bc_out_S = net(pt_x_bc)[1::2]
        # ei --> Werte, die minimiert werden müssen
        u_x_B = torch.autograd.grad(net_bc_out_B, pt_x_bc, create_graph=True, retain_graph=True,
                                    grad_outputs=torch.ones_like(net_bc_out_B))[0]
        u_xx_B = torch.autograd.grad(u_x_B, pt_x_bc, create_graph=True, retain_graph=True,
                                     grad_outputs=torch.ones_like(net_bc_out_B))[0]
        u_xxx_B = torch.autograd.grad(u_xx_B, pt_x_bc, create_graph=True, retain_graph=True,
                                      grad_outputs=torch.ones_like(net_bc_out_B))[0]
        u_x_S = torch.autograd.grad(net_bc_out_S, pt_x_bc, create_graph=True, retain_graph=True,
                                    grad_outputs=torch.ones_like(net_bc_out_S))[0]

        # Die Randbedingungen können in der Powerpoint-Präsentation angesehen werden [0] heißt erster Eintrag des Vektors, [-1] heißt letzter Eintrag des Vektors
        # Also z.B. u_x_B[0] ~ vb'(0) und vb'[-1] ~ vb'(L)
        # net_bc_out ist der Output des Netzwerks, also net_bc_out_b[0] ~ vb(0)
        # RB für Biegung
        e1_B = net_bc_out_B[0]
        e2_B = u_x_B[0]
        e3_B = u_xxx_B[0] - Q0[-1] / EI
        e4_B = u_xx_B[0] + M0[-1] / EI
        e5_B = u_xxx_B[-1]
        e6_B = u_xx_B[-1]

        # RB für Schub
        e1_S = net_bc_out_S[0]
        e2_S = u_x_S[0] + Q0[-1] / (K * A * G)
        e3_S = u_x_S[-1]

        # Alle e's werden gegen 0-Vektor (pt_zero) optimiert.
        mse_bc_B = mse_cost_function(e1_B, pt_zero) + mse_cost_function(e2_B,
                                                                        pt_zero) + 1 / normfactor * mse_cost_function(
            e3_B, pt_zero) + 1 / normfactor * mse_cost_function(e4_B, pt_zero) + mse_cost_function(e5_B,
                                                                                                   pt_zero) + mse_cost_function(
            e6_B, pt_zero)
        mse_ode_B = 1 / normfactor * mse_cost_function(ode_B, pt_all_zeros)
        mse_bc_S = mse_cost_function(e1_S, pt_zero) + 1 / normfactor * mse_cost_function(e2_S,
                                                                                         pt_zero) + mse_cost_function(
            e3_S, pt_zero)
        mse_ode_S = 1 / normfactor * mse_cost_function(ode_S, pt_all_zeros)

        loss_B = 1 / normfactor * (mse_ode_B + mse_bc_B)
        loss_S = 1 / normfactor * (mse_ode_S + mse_bc_S)

        loss = loss_B + lossfactor / (1.3 * Lb) * loss_S
        loss = loss.to(torch.float32)

        loss.backward()
        scheduler.step(loss)
        optimizer.step()
        with torch.autograd.no_grad():
            if epoch % 10 == 9:
                print(epoch, "Traning Loss:", loss.data)
                plt.grid()
                net_out_B = net(pt_x)[0::2]
                net_out_S = net(pt_x)[1::2]
                net_out_v = net_out_B + net_out_S
                err_b = torch.norm(net_out_B - f_anal_b, 2)
                err_s = torch.norm(net_out_S - f_anal_s, 2)
                print(f'Error = {err_s =},{err_b =}')
                if time.time()-start > 500:
                    return 500
                if err_b < 0.1 * Lb and err_s < Lb / lossfactor:
                    print(f"Die L^2 Norm der Fehler ist {err_b = },{ err_s = }.\nStoppe Lernprozess")
                    end = time.time()
                    return end - start
                line1.set_ydata(net_out_v.cpu().detach().numpy())
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