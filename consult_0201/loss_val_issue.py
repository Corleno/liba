# imports for model
import  torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.io
import random

import os
import argparse
import sys

########################################################################
# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 60

# Optimal training parameters
MLP_L1 = 1024
MLP_L2 = 1024
MLP_L3 = 1024
MLP_L4 = 1024
MLP_L5 = 1024
MLP_L6 = 1024
MLP_L7 = 512
MLP_L8 = 256

LR = 3e-5
L2_REGU = 1e-5

SIREN_W0 = 1.0
L1_WEIGHT = 1/40.0
L2_WEIGHT = 1/20.0

# early stopping
MIN_DELTA=1e-4
TOLERANCE=1000

RHO = 1
ALPHA = 0.999
TEMPATURE = 0.1

INPUT_DIM = 4
OUTPUT_DIM = 3
k_m = 1.5

########################################################################
# dynamic model parameters
current_dir = os.getcwd()
STORAGE_PATH = os.path.abspath(os.path.join(current_dir,'training_out_2DOF_MLP2Layer_relobralo_Model1_normalizedwt_L1L2'))

PRINT_PATH = os.path.abspath(os.path.join(STORAGE_PATH,"out.txt"))
with open(PRINT_PATH,"w") as f:
    print("",file=f)

data_path = os.path.abspath(os.path.join(current_dir,"x.mat"))
onestep_data_path = os.path.abspath(os.path.join(current_dir,"x_1step.mat"))
MODEL_NAME = "2DOF_MLP2Layer_relobralo_Model1_normalizedwt_L1L2"
NUM_EPOCHS = 1000
m = 5
m_H = 10
a = 0.5
b = 0.5
g = 9.8
B_left_annihilator = torch.tensor([[0, 0],[0, 1.0]]).to(device)

########################################################################
# load data
# input: path to input data
def load_data(data_path):
    mat = scipy.io.loadmat(data_path)
    X = torch.tensor(mat['xs'], requires_grad=True).float()
    return X

# TODO: test random sampling for training data
########################################################################
# fix random seed for reproductivity
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
########################################################################
"""
sinusoidal activation function
"""
class Sinusoidal(nn.Module):
    """
    Custom Sinusoidal activation function
    """
    def forward(self, x):
        return torch.sin(x)

# model
# 1 hiddenl layer MLP
# tunable params: l1, l2, activation function(TODO) 
class MLP(nn.Module):
    def __init__(self, l1=MLP_L1, l2=MLP_L2, l3=MLP_L3, l4 = MLP_L4, l5 = MLP_L5, l6 = MLP_L6, l7 = MLP_L7, l8 = MLP_L8):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(INPUT_DIM, l1),
            Sinusoidal(),
            nn.Linear(l1, l2),
            nn.Tanh(),
            nn.Linear(l2,l3),
            nn.Tanh(),
            nn.Linear(l3,l4),
            nn.Tanh(),
            nn.Linear(l4,l5),
            nn.Tanh(),
            nn.Linear(l5, l6),
            nn.Tanh(),
            nn.Linear(l6, l7),
            nn.Tanh(),
            nn.Linear(l7, l8),
            nn.Tanh(),
            nn.Linear(l8, OUTPUT_DIM)
            )
        self.model.apply(self.sine_init)

    def cholesky_decomposition(self, x):
        L = torch.stack([torch.stack([x[0], x[1]]),
                     torch.stack([x[1], x[2]])])
        M = torch.matmul(L,L.transpose(-1,-2))
        return M

    def forward(self, x):
        out = self.model(x)
        out = k_m * 0.5 * (out+1.0)
        return self.cholesky_decomposition(out)

    def sine_init(self, layer):
        """
        SIREN initialization for sine activations.
        Args:
            layer: Linear layer to initialize.
            w0: Frequency scaling factor (default: 1.0).
        """
        w0 = SIREN_W0
        if isinstance(layer, torch.nn.Linear):
            num_input = layer.weight.size(-1)
            std = np.sqrt(6 / num_input) / w0
            with torch.no_grad():
                layer.weight.uniform_(-std, std)
                if layer.bias is not None:
                    layer.bias.uniform_(-std, std)

########################################################################
# loss
class customLoss:
    def __init__(self, m, m_H, a, b, g, B_left_annihilator):
        self.m = m
        self.m_H = m_H
        self.a = a
        self.b = b 
        self.g = g 
        self.l = a+b
        self.B_left_annihilator = B_left_annihilator
        
        # precalculation for matrices
        self.C_ma = (m_H+m)*self.l*self.l+m*a*a
        self.C_mb = -m*self.l*b
        self.C_mc = m*b*b

        # record the losses for previous time step for relobralo
        self.prev_losses = None
        self.lam = torch.tensor([1.0, 1.0, 1.0, 1.0]).to(device)
        self.L0 = None
        self.rho = RHO
        self.alpha = ALPHA
        self.T = TEMPATURE

    ##################################
    
    def get_PDE_Loss_trajectory(self, model, X):
        # helpers
        def calculate_M(x):
            q1 = x[0]
            q2 = x[1]
            C_ma_tensor = torch.tensor(self.C_ma).to(device)
            C_mb_tensor = torch.tensor(self.C_mb).to(device)
            C_mc_tensor = torch.tensor(self.C_mc).to(device)
            cos_term = torch.cos(q1 - q2).to(device)
            M = torch.stack([torch.stack([C_ma_tensor, C_mb_tensor * cos_term]),
                torch.stack([C_mb_tensor * cos_term, C_mc_tensor])])
            return M

        def calculate_M_hat(x):
            y = model(x)
            return y

        def func_M1(x):
            M = calculate_M(x)
            q_dot = torch.stack([x[2], x[3]]).view(-1,1)
            return M @ q_dot

        def func_M2(x):
            M = calculate_M(x)
            q_dot = torch.stack([x[2], x[3]]).view(-1,1)
            return torch.transpose(q_dot,0,1) @ M @ q_dot

        def func_Mhat1(x):
            M_hat = calculate_M_hat(x)
            q_dot = torch.stack([x[2], x[3]]).view(-1,1)
            return M_hat @ q_dot

        def func_Mhat2(x):
            M_hat = calculate_M_hat(x)
            q_dot = torch.stack([x[2], x[3]]).view(-1,1)
            return torch.transpose(q_dot,0,1) @ M_hat @ q_dot

        # x:(4,1) y(3,1)
        def matching_condition_vmap(x):
            q1 = x[0]
            q2 = x[1]
            q = torch.stack([q1,q2]).view(-1,1)
            q_dot = torch.stack([x[2], x[3]]).view(-1,1)
            M = calculate_M(x)
            M_hat = calculate_M_hat(x)
            M_hat_inv = torch.linalg.pinv(M_hat)
            jacobianM1 = torch.func.jacrev(lambda x: func_M1(x))(x).reshape(2,4)[:,:2]
            jacobianM2 = torch.func.jacrev(lambda x: func_M2(x))(x).reshape(1,4)[:,:2]
            jacobianMhat1 = torch.func.jacrev(lambda x: func_Mhat1(x))(x).reshape(2,4)[:,:2]
            jacobianMhat2 = torch.func.jacrev(lambda x: func_Mhat2(x))(x).reshape(1,4)[:,:2]
            matching_tensor = (self.B_left_annihilator @ M) @ (torch.inverse(M) @ (jacobianM1 @ q_dot - 0.5 * torch.transpose(jacobianM2,0,1)) - M_hat_inv @ (jacobianMhat1 @ q_dot - 0.5 * torch.transpose(jacobianMhat2,0,1)))
            return 0.5 * torch.linalg.vector_norm(matching_tensor, ord=2)

        # use torch.vmap to vectorize the transform function
        vmap_pde_loss_func = torch.vmap(matching_condition_vmap)

        # apply the vectorized transform function on data
        L1s = vmap_pde_loss_func(X)
        return L1s

    #################################################################
    # PDE loss (mean across all data points: scalar)
    def get_PDE_Loss(self, model, X):
        L1s = self.get_PDE_Loss_trajectory(model, X)
        L1_mean = L1s.mean()
        # return avg loss, normalized avg loss
        return L1_mean, L1_mean/(torch.max(L1s) + 1e-8)

    ##################################
    def get_Boundary_Loss_trajectory(self, model):
        x0 = torch.tensor([0., 0., 0., 0.]).to(device)
        y0_pred = model(x0)
        y0_true = torch.tensor([[self.C_ma, self.C_mb],[self.C_mb, self.C_mc]]).to(device)
        out = torch.nn.functional.mse_loss(y0_pred, y0_true)
        return out

    ##################################
    # boundary loss (mean across all data points: scalar)
    def get_Boundary_Loss(self, model):
        out = self.get_Boundary_Loss_trajectory(model)
        L2_mean = out.mean()
        # return avg loss, normalized avg loss
        return L2_mean, L2_mean/(torch.max(out) + 1e-8)

    ##################################
    # condition loss (condition 1 mc_constant is trivial(does not need extra regulation) only adding condition2)
    # (mean across all data points: scalar)
    def get_condition_loss(self, model, X):
        def condition2_vmap(x):
            theta1 = x[0]
            theta2 = x[1]

            def get_ma_hat(x):
                y = model(x)
                return y[0][0]

            def get_mb_hat(x):
                y = model(x)
                return y[0][1]

            dma_hat_dtheata2 = torch.func.grad(lambda x: get_ma_hat(x))(x).reshape(1,4)[:,1]
            dmb_hat_dtheta1 = torch.func.grad(lambda x: get_mb_hat(x))(x).reshape(1,4)[:,0]
            part3 = 2*self.m*self.l*self.b*torch.sin(theta1-theta2)

            return dma_hat_dtheata2 - 2*dmb_hat_dtheta1 + part3

        # use torch.vmap to vectorize the transform function
        vmap_conditon2 = torch.vmap(condition2_vmap)

        # apply the vectorized transform function on data
        condtion2_error = vmap_conditon2(X)

        out = torch.pow(condtion2_error, 2)
        L3_mean = out.mean()
        # return avg loss, normalized avg loss
        return L3_mean, L3_mean/(torch.max(out) + 1e-8)

    ##################################
    # Mdot-2C constraint loss (mean across all data points: scalar)
    def get_dynamics_constraint_loss(self, model, X):
        def func_Mhat(x):
            return model(x)

        def get_C(dMdq, qdot):
            C = torch.stack((qdot, qdot), dim=1).reshape(2,2)

            for k in range(2):
                for j in range(2):
                    C[k,j] = 0
                    for i in range(2):
                        C[k,j] += 0.5 * (dMdq[k,j,i] + dMdq[k,i,j] - dMdq[i,j,k])*qdot[i]
            return C

        # transpose(Mhatdot-2C)+(Mhatdot-2C)
        def skew_symmetric_vmap(x):
            dMdq = torch.func.jacrev(lambda x: func_Mhat(x))(x).reshape(2,2,4)[:,:,:2]
            qdot = torch.stack([x[2], x[3]]).view(-1,1)
            Mhatdot = dMdq @ qdot
            Mhatdot = Mhatdot.reshape(2,2)
            C = get_C(dMdq, qdot)
            mat = Mhatdot - 2*C
            return torch.linalg.matrix_norm(torch.transpose(mat, 0, 1) + mat)


        # use torch.vmap to vectorize the transform function
        vmap_css = torch.vmap(skew_symmetric_vmap)

        # apply the vectorized transform function on data
        out = vmap_css(X)
        L4_mean = out.mean()
        # return avg loss, normalized avg loss
        return L4_mean, L4_mean/(torch.max(out) + 1e-8)

    ##################################
    # add loss weights in the param
    # TODO: use ReLoBRaLo to adjust weight for each loss terms
    # right now just use constant weights and tune those for comparison

    # https://github.com/rbischof/relative_balancing/blob/main/src/update_rules.py
    
    def total_loss(self, model, X):
        L1, normalized_L1 = self.get_PDE_Loss(model, X)
        L2, normalized_L2 = self.get_Boundary_Loss(model)
        L3, normalized_L3 = self.get_condition_loss(model, X)
        L4, normalized_L4 = self.get_dynamics_constraint_loss(model, X)
        
        losses = torch.stack([normalized_L1, normalized_L2])

        # if self.prev_losses is None:
        #     self.prev_losses = losses.clone().detach()
        #     self.L0 = losses.clone().detach()

        # lambs_hat = torch.nn.functional.softmax(torch.stack([losses[i]/(self.prev_losses[i]*self.T+1e-12) for i in range(len(losses))]), dim=0)*len(losses)
        # lambs0_hat = torch.nn.functional.softmax(torch.stack([losses[i]/(self.L0[i]*self.T+1e-12) for i in range(len(losses))]), dim=0)*len(losses)

        # lambs =  torch.stack([self.rho*self.alpha*self.lam[i]+(1-self.rho)*self.alpha*lambs0_hat[i] + (1-self.alpha) * lambs_hat[i] for i in range(len(losses))])

        # train_loss = torch.dot(lambs, losses)

        # self.prev_losses = losses.clone().detach()
        # self.lam = lambs.clone().detach()

        total_loss = normalized_L1 + normalized_L2

        train_loss = L1_WEIGHT*L1 + L2

        return train_loss, total_loss, L1, L2, L3, L4, normalized_L1, normalized_L2, normalized_L3, normalized_L4

    ##################################
    # calculated normalized sum of all losses for evaluation
    # def normalized_loss(self, model, X):
    #     L1, L1s = self.get_PDE_Loss(model, X)
    #     L2, L2s = self.get_Boundary_Loss(model)
    #     L3, L3s = self.get_condition_loss(model, X)
    #     L4, L4s = self.get_dynamics_constraint_loss(model, X)
    #     return float(L1/torch.max(L1s) + L2/torch.max(L2s) + L3/torch.max(L3s) + L4/torch.max(L4s))


########################################################################
# train function with early stopping 
def train(model, loss_funcs, optimizer, X, device, num_epochs, min_delta=1e-4, tolerance=5):
  # add adaptive learning weights later in for loop

  if not device:
      device = torch.device("cpu") 

  l1_epoch = []
  l2_epoch = []
  l3_epoch = []
  l4_epoch = []
  normalized_L1_epoch = []
  normalized_L2_epoch = []
  normalized_L3_epoch = []
  normalized_L4_epoch = []
  total_loss_epoch = []
  train_loss_epoch = []

  # early stopping
  #best_loss = float('inf')
  #patience = 0

  for ep in range(num_epochs):
      try:
          X = X.to(device)
          train_loss, total_loss, L1, L2, L3, L4, normalized_L1, normalized_L2, normalized_L3, normalized_L4 = loss_funcs.total_loss(model, X)
          # log losses
          l1_epoch.append(L1.detach().cpu())
          l2_epoch.append(L2.detach().cpu())
          l3_epoch.append(L3.detach().cpu())
          l4_epoch.append(L4.detach().cpu())
          normalized_L1_epoch.append(normalized_L1.detach().cpu())
          normalized_L2_epoch.append(normalized_L2.detach().cpu())
          normalized_L3_epoch.append(normalized_L3.detach().cpu())
          normalized_L4_epoch.append(normalized_L4.detach().cpu())
          total_loss_epoch.append(total_loss.detach().cpu())
          train_loss_epoch.append(train_loss.detach().cpu())
          # backward prop
          # TODO: schedule gradient descent alteratively for different loss if needed or adjust learning rate
          optimizer.zero_grad()
          train_loss.backward()
          optimizer.step()
          if ep%10==0 or ep==num_epochs-1:
              with open(PRINT_PATH, "a") as f:
                  print(f"epoch: {ep}, train loss: {train_loss.item():>7f}, total_loss:{total_loss}, pde_loss :{L1}, boundary_loss:{L2}", file=f)
          #curr_loss = L1.detach()
          #if (best_loss - curr_loss) > min_delta:
          #    best_loss = curr_loss
          #    patience = 0
          #else:
          #    patience += 1
          #if patience >= tolerance:
          #    print(f"Early stopping at epoch {ep+1} ")
          #    break
      except Exception as e:
          # error handling
          print(f"Error at epoch {ep+1}: {e}")
          sys.exit(1)


  return l1_epoch, l2_epoch, l3_epoch, l4_epoch, normalized_L1_epoch, normalized_L2_epoch, normalized_L3_epoch, normalized_L4_epoch, total_loss_epoch, train_loss_epoch

########################################################################
# test function
# def test(model, loss_funcs, X, device):
#     if not device:
#         device = torch.device("cpu") 

#     X = X.to(device)
#     normalized_total_loss = loss_funcs.normalized_loss(model, X)
#     return normalized_total_loss


########################################################################  
## main function ##
# plot and data for each training worker
# TODO: let each work save loss plots
# select a metric to evaluate training performance (consider normalizing loss)

# evaluate and select best model
########################################################################
if __name__ == "__main__":

    set_seed(SEED)

    # parse command line argument for gpu and cpu resourse
    parser = argparse.ArgumentParser(description="Training for 2DOF dynamic model with 1 layer MLP")
    parser.add_argument(
        "--num_epoch", type=int, default=NUM_EPOCHS, help="Number of epochs for training"
        )
    args, _ = parser.parse_known_args()

    num_epochs = args.num_epoch

    # load params for the dynamic model to create loss functions
    loss_funcs = customLoss(m, m_H, a, b, g, B_left_annihilator)

    # load training data
    X = load_data(data_path)

    model = MLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=L2_REGU) # add L2 regularization
 
    L1s, L2s, L3s, L4s, normalized_L1s, normalized_L2s, normalized_L3s, normalized_L4s, TotalLs, TrainLs = train(model, loss_funcs, opt, X, device, num_epochs, MIN_DELTA, TOLERANCE)
     
    if not os.path.exists(STORAGE_PATH):
        os.mkdir(STORAGE_PATH)

    plt.legend(loc='upper right')

    # original weights
    filename = MODEL_NAME+"_total.png"
    plt_title = os.path.abspath(os.path.join(STORAGE_PATH,filename))
    plt.plot(L1s, label="PDE Loss")
    plt.plot(L2s, label="Boundary Loss")
    plt.plot(L3s, label="Condition Loss")
    plt.plot(L4s, label="Constraint Loss")
    plt.plot(TotalLs, label="normalized PDE loss + Boundary Loss")
    plt.yscale("log") 
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('Loss(Log)')
    plt.title('Original Losses v Epochs')
    plt.savefig(plt_title)
    plt.close()

    # normalized weights
    filename = MODEL_NAME+"_normtotal.png"
    plt_title = os.path.abspath(os.path.join(STORAGE_PATH,filename))
    plt.plot(normalized_L1s, label="normalized PDE Loss")
    plt.plot(normalized_L2s, label="normalized Boundary Loss")
    plt.plot(normalized_L3s, label="normalized Condition Loss")
    plt.plot(normalized_L4s, label="normalized Constraint Loss")
    plt.plot(TotalLs, label="normalized PDE loss + Boundary Loss")
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.title('Normalized Losses v Epochs')
    plt.savefig(plt_title)
    plt.close()

    # normalized total (PDE+Boundary)
    filename = MODEL_NAME+"_normL1L2.png"
    plt_title = os.path.abspath(os.path.join(STORAGE_PATH, filename))
    plt.plot(L1s)
    plt.xlabel('epochs')
    plt.ylabel('Total Loss (L1+L2)')
    plt.title('Normalized Total Loss v Epochs')
    plt.savefig(plt_title)
    plt.close()

    # train loss (ReLoBRaLo)
    filename = MODEL_NAME+"_ReLoBRaLoLoss.png"
    plt_title = os.path.abspath(os.path.join(STORAGE_PATH,filename))
    plt.plot(TrainLs)
    plt.xlabel('epochs')
    plt.ylabel('Train Loss (L1+L2 ReLoBRaLo)')
    plt.title('Train Loss (PDE+Boundary using ReLoBRaLo) v Epochs')
    plt.savefig(plt_title)
    plt.close()

    # PDE Loss
    filename = MODEL_NAME+"_PDELoss.png"
    plt_title = os.path.abspath(os.path.join(STORAGE_PATH,filename))
    plt.plot(L1s)
    plt.xlabel('epochs')
    plt.ylabel('PDE Loss')
    plt.title('PDE Loss v Epochs')
    plt.savefig(plt_title)
    plt.close()

    # Boundary Loss
    filename = MODEL_NAME+"_BoundaryLoss.png"
    plt_title = os.path.abspath(os.path.join(STORAGE_PATH,filename))
    plt.plot(L2s)
    plt.xlabel('epochs')
    plt.ylabel('Boundary Loss')
    plt.title('Boundary Loss v Epochs')
    plt.savefig(plt_title)
    plt.close()

    # Condition Loss
    filename = MODEL_NAME+"_ConditionLoss.png"
    plt_title = os.path.abspath(os.path.join(STORAGE_PATH,filename))    
    plt.plot(L3s)
    plt.xlabel('epochs')
    plt.ylabel('Condition Loss')
    plt.title('Condition Loss v Epochs')
    plt.savefig(plt_title)
    plt.close()

    # Constraint Loss
    filename = MODEL_NAME+"_ConstraintLoss.png"
    plt_title = os.path.abspath(os.path.join(STORAGE_PATH,filename))
    plt.plot(L4s)
    plt.xlabel('epochs')
    plt.ylabel('Constraint Loss')
    plt.title('Constraint Loss v Epochs')
    plt.savefig(plt_title)
    plt.close()

    # PDE Loss and Boundary Loss for one trajectory
    X_1step = load_data(onestep_data_path).to(device)
    L1s_traj = loss_funcs.get_PDE_Loss_trajectory(model,X_1step)
    L1s_traj = L1s_traj.detach().cpu()
    filename = MODEL_NAME+"_PDELossTraj.png"
    plt_title = os.path.abspath(os.path.join(STORAGE_PATH,filename))
    plt.plot(L1s_traj)
    plt.xlabel('t')
    plt.ylabel('PDE Loss')
    plt.title('PDE Loss v time step')
    plt.savefig(plt_title)
    plt.close()

    # save model
    filename = MODEL_NAME+"_model.pth"
    filepath = os.path.abspath(os.path.join(STORAGE_PATH,filename))
    torch.save(model.state_dict(), filepath)
    print("Model parameters saved.")



