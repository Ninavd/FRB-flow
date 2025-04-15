import numpy as np
import matplotlib.pyplot as plt

def plot_trace(sampler, steps, true_value, ylabel="$t_0$"):
    nwalkers = sampler.nwalkers

    plt.figure(figsize=(12,3))

    # each walker a different color
    plt.subplot(121)
    for i, walker in enumerate(sampler.get_chain().reshape((steps, nwalkers)).T):
        start = int(i * steps)
        plt.plot(range(start, start + steps), walker)

    plt.title("MCMC traceplot")
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.hlines([true_value], xmin=0, xmax=nwalkers*steps, linestyle='dashed', color='black')

    # walkers back to back
    plt.subplot(122)
    plt.plot(sampler.get_chain(flat=True), 'k-')
    plt.title("MCMC traceplot")
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.hlines([true_value], xmin=0, xmax=nwalkers*steps, linestyle='dashed', color='red')


if __name__=="__main__":
    pass
