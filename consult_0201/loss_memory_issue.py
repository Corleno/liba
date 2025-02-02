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

from eightDOF_loss import customLoss
########################################################################
# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 60

# Optimal training parameters
COMMON_L1 = 2048
COMMON_L2 = 2048
COMMON_L3 = 2048
COMMON_L4 = 2048
COMMON_L5 = 2048
COMMON_L6 = 2048
COMMON_L7 = 1024
COMMON_L8 = 1024
BRANCH_L1 = 1024
BRANCH_L2 = 1024
BRANCH_L3 = 1024
BRANCH_L4 = 1024
BRANCH_L5 = 512

LR = 1e-4
L2_REGU = 1e-3

SIREN_W0 = 1.0
L1_WEIGHT = 1/40.0
L2_WEIGHT = 1/20.0

# early stopping
MIN_DELTA=1e-5
TOLERANCE=20

RHO = 1
ALPHA = 0.999
TEMPATURE = 0.1

# model
INPUT_DIM = 16
OUTPUT_DIM = 12
k_m = 1.5

# epochs
NUM_EPOCHS = 1000

# approximation for checking if condition met
EPSILON = 1e-3

########################################################################
# dynamic model parameters
current_dir = os.getcwd()
STORAGE_PATH = os.path.abspath(os.path.join(current_dir,'training_out_8DOF_branchedMLP_relobralo_Model1_L1L2'))
data_path = os.path.abspath(os.path.join(current_dir,"x_8dof.mat"))
MODEL_NAME = "8DOF_branchedMLP_relobralo_Model1_L1L2"

Mp = 31.73    # same as Mh in paper
Mf = 1.0
Ms = 4.053
Mt = 9.457
Mfs = Mf
Mss = Ms
Mts = Mt
g = 9.81

l1 = 0.428  # same as ls
l2 = 0.428  # same as lt
la = 0.07
lf = 0.2

I1x = 0.0369      # same as Is
I2x = 0.1995      # same as It
I1y, I1z, I2y, I2z, Ipx, Ipy, Ipz = 0, 0, 0, 0, 0, 0, 0



# B matrix
Bst = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]).to(device)

# Matlab (delete when not use)
dim = 16
massRatio=3.5/5
M = 13.51
M1 = M*(1-massRatio)
M2 = M*(massRatio)
Kp1 = 9*(Mp+2*M+2*Mf)
Kp2 = 9*(Mp+2*M+2*Mf)
Kd1 = 1.3*np.sqrt(Kp1)*.7
Kd2 = 1.3*np.sqrt(Kp2)*.7
temp_Kp = 3*(Mp+2*M+2*Mf)
slope = 0.095

########################################################################
# load data
# input size for 8DOF: 16
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
    def forward(self, x):
        return torch.sin(x)

# model
# 1 hiddenl layer MLP
# tunable params: l1, l2, activation function(TODO) 
class MLP(nn.Module):
    def __init__(self, common_l1=COMMON_L1, common_l2 = COMMON_L2, common_l3 = COMMON_L3, common_l4 = COMMON_L4, common_l5 = COMMON_L5, common_l6 = COMMON_L6, common_l7 = COMMON_L7, common_l8 = COMMON_L8, branch_l1 = BRANCH_L1, branch_l2 = BRANCH_L2, branch_l3 = BRANCH_L3, branch_l4 = BRANCH_L4, branch_l5 = BRANCH_L5):
        super().__init__()
        self.common_layers = nn.Sequential(
            nn.Linear(INPUT_DIM, common_l1),
            Sinusoidal(),
            nn.Linear(common_l1, common_l2),
            nn.Tanh(),
            nn.Linear(common_l2,common_l3),
            nn.Tanh(),
            nn.Linear(common_l3,common_l4),
            nn.Tanh(),
            nn.Linear(common_l4,common_l5),
            nn.Tanh(),
            nn.Linear(common_l5,common_l6),
            nn.Tanh(),
            nn.Linear(common_l6,common_l7),
            nn.Tanh(),
            nn.Linear(common_l7,common_l8),
            nn.Tanh()
            )

        self.branch1 = nn.Sequential(
            nn.Linear(common_l8,branch_l1),
            nn.Tanh(),
            nn.Linear(branch_l1,branch_l2),
            nn.Tanh(),
            nn.Linear(branch_l2,branch_l3),
            nn.Tanh(),
            nn.Linear(branch_l3,branch_l4),
            nn.Tanh(),
            nn.Linear(branch_l4,branch_l5),
            nn.Tanh(),
            nn.Linear(branch_l5,OUTPUT_DIM)
            )

        self.branch2 = nn.Sequential(
            nn.Linear(common_l8,branch_l1),
            nn.Tanh(),
            nn.Linear(branch_l1,branch_l2),
            nn.Tanh(),
            nn.Linear(branch_l2,branch_l3),
            nn.Tanh(),
            nn.Linear(branch_l3,branch_l4),
            nn.Tanh(),
            nn.Linear(branch_l4,branch_l5),
            nn.Tanh(),
            nn.Linear(branch_l5,OUTPUT_DIM)
            )

        self.branch3 = nn.Sequential(
            nn.Linear(common_l8,branch_l1),
            nn.Tanh(),
            nn.Linear(branch_l1,branch_l2),
            nn.Tanh(),
            nn.Linear(branch_l2,branch_l3),
            nn.Tanh(),
            nn.Linear(branch_l3,branch_l4),
            nn.Tanh(),
            nn.Linear(branch_l4,branch_l5),
            nn.Tanh(),
            nn.Linear(branch_l5,OUTPUT_DIM)
            )

        self.common_layers.apply(self.sine_init)
        self.branch1.apply(self.sine_init)
        self.branch2.apply(self.sine_init)
        self.branch3.apply(self.sine_init)



    def cholesky_decomposition(self, x):
        L = torch.stack([torch.stack([x[0], x[1]]),
                     torch.stack([x[1], x[2]])])
        M = torch.matmul(L,L.transpose(-1,-2))
        return M

    def forward(self, x):
        out = self.common_layers(x)
        out = k_m * 0.5 * (out + 1.0)

        # based on initial x: determine which phase
        # flat foot
        phase2_cond = torch.all(torch.stack([
        torch.abs(x[0]) < EPSILON,
        torch.abs(x[1]) < EPSILON,
        torch.abs(x[2] - slope) < EPSILON]), dim=0) 
        # heel contact
        phase1_cond = torch.all(torch.stack([torch.abs(x[0]) < EPSILON,
            torch.abs(x[1]) < EPSILON
            ]), dim=0) 
        # toe contact
        phase3_cond = torch.all(torch.stack([torch.abs(x[0] - lf*torch.cos(x[2])) < EPSILON,
            torch.abs(x[1] - lf*torch.sin(x[2])) < EPSILON
            ]), dim=0)


        phase2_mask = phase2_cond & ~phase3_cond
        phase1_mask = phase1_cond & (~phase2_cond)
        phase3_mask = phase3_cond & ~phase1_cond

        out = phase1_mask.to(torch.float32)*self.branch1(out) + phase2_mask.to(torch.float32)*self.branch2(out) + phase3_mask.to(torch.float32)*self.branch3(out) 

        K_m = torch.stack([
            torch.ones(8, device=device),  # First row
            torch.ones(8, device=device),  # Second row
            torch.cat([torch.ones(2, device=device), out[0].repeat(6)]),  # Third row
            torch.cat([torch.ones(2, device=device), out[0:2], out[1].repeat(4)]),  # Fourth row
            torch.cat([torch.ones(2, device=device), out[0:3], out[2].repeat(3)]),  # Fifth row
            torch.cat([torch.ones(2, device=device), out[0:4], out[3].repeat(2)]),  # Sixth row
            torch.cat([torch.ones(2, device=device), out[0:4], out[4].repeat(2)]),  # Seventh row
            torch.cat([torch.ones(2, device=device), out[0:6]])   # Eighth row
        ])

        K_n = torch.stack([
            torch.ones(1, device=device),
            torch.ones(1, device=device),
            torch.stack([out[6]]),
            torch.stack([out[7]]),
            torch.stack([out[8]]),
            torch.stack([out[9]]),
            torch.stack([out[10]]),
            torch.stack([out[11]])])

        return K_m, K_n

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


########################################################################
# train function with early stopping 
def train(model, loss_funcs, optimizer, X, device, num_epochs, min_delta=1e-4, tolerance=5):
    if not device:
        device = torch.device("cpu") 

    l1_epoch = []
    l2_epoch = []
    l4_epoch = []
    normalized_L1_epoch = []
    normalized_L2_epoch = []
    normalized_L4_epoch = []
    total_loss_epoch = []
    train_loss_epoch = []

    # early stopping
    #best_loss = float('inf')
    #patience = 0

    for ep in range(num_epochs):
        try:
            X = X.to(device)
            train_loss, total_loss, L1, L2, L4, normalized_L1, normalized_L2, normalized_L4 = loss_funcs.total_loss(model, X)
            # log losses
            l1_epoch.append(L1.detach().cpu())
            l2_epoch.append(L2.detach().cpu())
            l4_epoch.append(L4.detach().cpu())
            normalized_L1_epoch.append(normalized_L1.detach().cpu())
            normalized_L2_epoch.append(normalized_L2.detach().cpu())
            normalized_L4_epoch.append(normalized_L4.detach().cpu())
            total_loss_epoch.append(total_loss.detach().cpu())
            train_loss_epoch.append(train_loss.detach().cpu())

            # backward prop
            # TODO: schedule gradient descent alteratively for different loss if needed or adjust learning rate
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # print progress
            if ep%10==0:
                print(f"epoch: {ep}, train loss: {train_loss.item():>7f}, total_loss:{total_loss}, pde_loss :{L1}, boundary_loss:{L2}")
            
            # early stopping
            # curr_loss = L1.detach()+L2.detach()
            # if (best_loss - curr_loss) > min_delta:
            #   best_loss = curr_loss
            #   patience = 0
            # else:
            #   patience += 1

            # if patience >= tolerance:
            #   print(f"Early stopping at epoch {ep+1} ")
            #   break
        except Exception as e:
            # error handling
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f"Error at epoch {ep+1}: {e}")
            sys.exit(1)


    return l1_epoch, l2_epoch, l4_epoch, normalized_L1_epoch, normalized_L2_epoch, normalized_L4_epoch, total_loss_epoch, train_loss_epoch

########################################################################
# test function
# def test(model, loss_funcs, X, device):
#   if not device:
#       device = torch.device("cpu") 

#   X = X.to(device)
#   normalized_total_loss = loss_funcs.normalized_loss(model, X)
#   return normalized_total_loss


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
    loss_funcs = customLoss()

    # load training data
    X = load_data(data_path)

    model = MLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=L2_REGU) # add L2 regularization
 
    L1s, L2s, L4s, normalized_L1s, normalized_L2s, normalized_L4s, TotalLs, TrainLs = train(model, loss_funcs, opt, X, device, num_epochs, MIN_DELTA, TOLERANCE)

    # create folder if not exist
    if not os.path.exists(STORAGE_PATH):
        os.mkdir(STORAGE_PATH)

    # plots
    # original weights
    filename = MODEL_NAME+"_total.png"
    plt_title = os.path.abspath(os.path.join(STORAGE_PATH,filename))
    plt.plot(L1s, label="PDE Loss")
    plt.plot(L2s, label="Boundary Loss")
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

    # Constraint Loss
    filename = MODEL_NAME+"_ConstraintLoss.png"
    plt_title = os.path.abspath(os.path.join(STORAGE_PATH,filename))
    plt.plot(L4s)
    plt.xlabel('epochs')
    plt.ylabel('Constraint Loss')
    plt.title('Constraint Loss v Epochs')
    plt.savefig(plt_title)
    plt.close()

    # save model
    filename = MODEL_NAME+"_model.pth"
    filepath = os.path.abspath(os.path.join(STORAGE_PATH,filename))
    torch.save(model.state_dict(), filepath)
    print("Model parameters saved.")



