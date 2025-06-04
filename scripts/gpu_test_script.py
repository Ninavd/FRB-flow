import sys
from corner import corner
from numpy.lib.stride_tricks import sliding_window_view

sys.path.append('..')
print(sys.path)

from src.simulator import Model, BurstSimulator
from src.flow_matching.distributions import Prior, Posterior
from src.flow_matching.probability_path import GuidedLinearProbabilityPath
from src.flow_matching.training import GuidedConditionalFlowMatchingTrainer
from src.flow_matching.integration import EulerODESolver
from src.flow_matching.models import MLPGuidedVectorField, FRBLightCurveCNN, LightCurveThinner
from src.helpers import record_every, hist2d_samples

import numpy as np
from matplotlib import pyplot as plt
import torch

# choose device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
print("CURRENT DEVICE: ", device)

# __________________________________________
#
#             TRAINING
# __________________________________________

path = GuidedLinearProbabilityPath(
    p_simple=Prior(),
    p_data=Posterior()
)

latent_dim = 128
time_seq_encoder = FRBLightCurveCNN(latent_dim=latent_dim)
vector_field = MLPGuidedVectorField(dim=2, hiddens=[64, 64, 32, 16], y_dim=latent_dim, time_seq_encoder=time_seq_encoder)

trainer = GuidedConditionalFlowMatchingTrainer(path, vector_field)
losses = trainer.train(101, device, 5e-4, save_checkpoint=True, batch_size=256)

print('\n')
# ___________________________________________
#
#               EVALUATION
# ___________________________________________

# Plot the loss evolution
window_size = 5
windows = sliding_window_view(losses, window_shape=window_size)
running_avg = windows.mean(axis=1)

# log linear
plt.figure()
plt.plot(losses, alpha=0.3)
plt.plot(running_avg, label=f"running average N={window_size}")
plt.yscale('log')
plt.xlabel('step')
plt.ylabel("average loss")
plt.title("Evolution of loss")
plt.legend()
plt.savefig('loss_log-linear.png', bbox_inches="tight")

# loglog
plt.figure()
plt.plot(losses, alpha=0.3)
plt.plot(running_avg, label=f"running average N={window_size}")
plt.yscale('log')
plt.xscale('log')
plt.xlabel('step')
plt.ylabel("average loss")
plt.title("Evolution of loss")
plt.legend()
plt.savefig('loss_log-log.png', bbox_inches="tight")

# generate one instance of simulated data to guide prior samples with

# fixed model parameters (NOTE: Should be same as used during training in Posterior)
N = 2
time = np.linspace(0, 1.0, 1000)
amp  = 25.0
rise = 0.03
skew = 5
ybkg = 5.0
true_t0 = [0.2, 0.8] # NOTE: this has to obey the prior! (i.e. t0_1 < t0_2 < ... < t0_n)

# generate simulated data for one burst
burstparams = {
    't0'   : true_t0, 
    'amp'  : [amp, amp],
    'rise' : [rise, rise],
    'skew' : [skew, skew]
}

model = Model(time=time,ncomp=N, burstparams=burstparams, ybkg=ybkg)
simulator = BurstSimulator(model)
x_counts = simulator.simulate_burst() 

num_samples = 5000  # number of prior samples to transform 
num_marginals = 5   # number of snapshots

# TODO: maybe do this in batches
# use same data point for conditioning all prior samples
simulation_time_resolution = 1000
simulations = torch.zeros((num_samples, simulation_time_resolution)).to(device)

for row_nr in range(len(simulations)):
    simulations[row_nr, :] = torch.Tensor(x_counts)

# Initialize ODE solver
solver = EulerODESolver(vector_field)
ts = torch.linspace(0, 1, 100).to(device)

# simulate ODE starting from x0
x0 = path.p_simple.sample(num_samples).to(device)
xts = solver.solve_with_trajectory(x0, ts.view(1, -1, 1).expand(num_samples,-1,1), y=simulations)

# only save num_marginals snapshots 
record_every_idxs = record_every(len(ts), len(ts) // (num_marginals - 1))
xts = xts[:,record_every_idxs,:]

# plot snapshots of marginal path
plt.figure(figsize=(20, 5))

for idx in range(xts.shape[1]):
    xx = xts[:,idx,:]
    t = ts.cpu()[record_every_idxs[idx]]

    plt.subplot(1, num_marginals, idx + 1)
    
    hist2d_samples(xx.cpu(), range=[[0,1], [0,1]], percentile=100, cmap='Blues')
    
    plt.title(f"Learned, t={t:.2f}")
    plt.xlabel(f"t0_1")
    plt.ylabel(f"t0_2")

    plt.xlim(0, 1)
    plt.ylim(0, 1)

plt.savefig('marginal_path_snapshots.png', bbox_inches="tight")

fig = corner(xx.cpu().numpy(), labels=["t0_1", "t0_2"], truths=np.array(true_t0))
fig.savefig('corner_plot.png', bbox_inches="tight")