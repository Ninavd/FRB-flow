import matplotlib.pyplot as plt

def plot_1d_hist(samples, true_value):
    """
    Plots histogram of (1D) sample distribution.
    """
    frequencies, _, _ = plt.hist(samples[:, 0], 50, color="k", histtype="step")
    plt.vlines(true_value,  ymin=0, ymax=max(frequencies), linestyles="dashed", color ='red', label='true $t_0$')

    plt.title("Sample distribution")
    plt.xlabel("$t_0$")
    plt.ylabel("$p(t_0 | x_0)$")
    plt.gca().set_yticks([])

    plt.legend()

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