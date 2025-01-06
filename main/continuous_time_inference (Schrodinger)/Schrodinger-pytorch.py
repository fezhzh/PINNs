import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
import time

class PhysicsInformedNN:
    def __init__(self, x0, u0, v0, tb, X_f, layers, lb, ub):
        self.lb = lb
        self.ub = ub

        self.x0 = x0
        self.t0 = torch.zeros_like(x0)
        self.u0 = u0
        self.v0 = v0
        
        self.tb = tb
        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]

        # 初始化神经网络
        self.layers = layers
        self.model = self.build_network(layers)
        
        # 定义优化器
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_function = nn.MSELoss()

    def build_network(self, layers):
        layers_list = []
        for i in range(len(layers) - 1):
            layers_list.append(nn.Linear(layers[i], layers[i + 1]))
            if i != len(layers) - 2:
                layers_list.append(nn.Tanh())
        return nn.Sequential(*layers_list)

    def forward(self, x, t):
        X = torch.cat([x, t], dim=1)
        output = self.model(X)
        u = output[:, 0:1]
        v = output[:, 1:2]
        return u, v

    def net_f_uv(self, x, t):
        u, v = self.forward(x, t)
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]

        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u), create_graph=True)[0]
        
        v_t = torch.autograd.grad(v, t, torch.ones_like(v), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v), create_graph=True)[0]
        
        f_u = u_t + 0.5 * v_xx + (u**2 + v**2) * v
        f_v = v_t - 0.5 * u_xx - (u**2 + v**2) * u
        return f_u, f_v

    def loss(self):
        u_pred, v_pred = self.forward(self.x0, self.t0)
        f_u_pred, f_v_pred = self.net_f_uv(self.x_f, self.t_f)

        loss_u = self.loss_function(self.u0, u_pred)
        loss_v = self.loss_function(self.v0, v_pred)
        loss_f_u = self.loss_function(torch.zeros_like(f_u_pred), f_u_pred)
        loss_f_v = self.loss_function(torch.zeros_like(f_v_pred), f_v_pred)

        return loss_u + loss_v + loss_f_u + loss_f_v

    def train(self, nIter):
        for it in range(nIter):
            self.optimizer.zero_grad()
            loss_value = self.loss()
            loss_value.backward()
            self.optimizer.step()
            
            if it % 10 == 0:
                print(f'Iteration {it}, Loss: {loss_value.item()}')

    def predict(self, X_star):
        x_star = X_star[:, 0:1]
        t_star = X_star[:, 1:2]
        u_pred, v_pred = self.forward(x_star, t_star)
        f_u_pred, f_v_pred = self.net_f_uv(x_star, t_star)
        return u_pred.detach().numpy(), v_pred.detach().numpy(), f_u_pred.detach().numpy(), f_v_pred.detach().numpy()


if __name__ == "__main__":
    noise = 0.0
    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi/2])

    N0 = 50
    N_b = 50
    N_f = 20000
    layers = [2, 100, 100, 100, 100, 2]

    data = scipy.io.loadmat('../Data/NLS.mat')

    t = data['tt'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = data['uu']
    Exact_u = np.real(Exact)
    Exact_v = np.imag(Exact)

    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact_u.T.flatten()[:, None]
    v_star = Exact_v.T.flatten()[:, None]

    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    x0 = torch.tensor(x[idx_x, :], dtype=torch.float32)
    u0 = torch.tensor(Exact_u[idx_x, 0:1], dtype=torch.float32)
    v0 = torch.tensor(Exact_v[idx_x, 0:1], dtype=torch.float32)

    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    tb = torch.tensor(t[idx_t, :], dtype=torch.float32)

    X_f = lb + (ub - lb) * lhs(2, N_f)
    X_f = torch.tensor(X_f, dtype=torch.float32)

    model = PhysicsInformedNN(x0, u0, v0, tb, X_f, layers, lb, ub)

    start_time = time.time()
    model.train(50000)
    elapsed = time.time() - start_time
    print(f'Training time: {elapsed:.4f}s')

    u_pred, v_pred, f_u_pred, f_v_pred = model.predict(torch.tensor(X_star, dtype=torch.float32))

    # Plotting or further analysis goes here
   


    h_pred = np.sqrt(u_pred**2 + v_pred**2)
            
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    error_h = np.linalg.norm(h_star-h_pred,2)/np.linalg.norm(h_star,2)
    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
    print('Error h: %e' % (error_h))

    
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    V_pred = griddata(X_star, v_pred.flatten(), (X, T), method='cubic')
    H_pred = griddata(X_star, h_pred.flatten(), (X, T), method='cubic')

    FU_pred = griddata(X_star, f_u_pred.flatten(), (X, T), method='cubic')
    FV_pred = griddata(X_star, f_v_pred.flatten(), (X, T), method='cubic')     
    

    
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    
    
    X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0)
    X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
    X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)
    X_u_train = np.vstack([X0, X_lb, X_ub])

    fig, ax = newfig(1.0, 0.9)
    ax.axis('off')
    
    ####### Row 0: h(t,x) ##################    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    h = ax.imshow(H_pred.T, interpolation='nearest', cmap='YlGnBu', 
                  extent=[lb[1], ub[1], lb[0], ub[0]], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (X_u_train.shape[0]), markersize = 4, clip_on = False)
    
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[75]*np.ones((2,1)), line, 'k--', linewidth = 1)
    ax.plot(t[100]*np.ones((2,1)), line, 'k--', linewidth = 1)
    ax.plot(t[125]*np.ones((2,1)), line, 'k--', linewidth = 1)    
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    leg = ax.legend(frameon=False, loc = 'best')
#    plt.setp(leg.get_texts(), color='w')
    ax.set_title('$|h(t,x)|$', fontsize = 10)
    
    ####### Row 1: h(t,x) slices ##################    
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,Exact_h[:,75], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,H_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')    
    ax.set_title('$t = %.2f$' % (t[75]), fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-5.1,5.1])
    ax.set_ylim([-0.1,5.1])
    
    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,Exact_h[:,100], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,H_pred[100,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')
    ax.axis('square')
    ax.set_xlim([-5.1,5.1])
    ax.set_ylim([-0.1,5.1])
    ax.set_title('$t = %.2f$' % (t[100]), fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), ncol=5, frameon=False)
    
    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,Exact_h[:,125], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,H_pred[125,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')
    ax.axis('square')
    ax.set_xlim([-5.1,5.1])
    ax.set_ylim([-0.1,5.1])    
    ax.set_title('$t = %.2f$' % (t[125]), fontsize = 10)
    
    # savefig('./figures/NLS')  
    
