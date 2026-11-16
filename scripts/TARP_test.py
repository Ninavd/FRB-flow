import argparse
import sys

sys.path.append('..')

from sbi.analysis.plot import plot_tarp
from sbi.diagnostics import check_tarp, run_tarp

from src.flow_matching.distributions import DiscreteUniform
from src.flow_matching.helpers import choose_device
from src.flow_matching.loader import (
    load_model_info,
    load_trained_model,
    prob_path_from_config,
)
from src.flow_matching.neural_posterior import NeuralPosterior
        

def main(N_inf, job_id="15055686"):
    import matplotlib.pyplot as plt

    device=choose_device()
    vector_field, classifier, config = load_trained_model(job_id, device)

    # load probability path used
    prob_path = prob_path_from_config(config)
    model_info = load_model_info(config, prob_path, device)

    simulator = prob_path.p_data

    posterior = NeuralPosterior(vector_field, classifier, model_info, prob_path)

    # -------------- TESTING FMPE ----------------------------------

    posterior = NeuralPosterior(vector_field, classifier, model_info, prob_path, device, fixed_N=N_inf)

    # generate ground truth parameters and corresponding simulated observations for SBC.
    num_tarp_samples = 2000  # choose a number of sbc runs, should be ~100s
    simulator.N_prior = DiscreteUniform(low=N_inf, high=N_inf)
    true_params, simulated_bursts, _ = simulator.sample(num_tarp_samples) 
    
    # remove irrelevant part of param vector
    bs, _ = true_params.shape
    new_dim = N_inf * 4
    true_params = true_params.view(-1, 4, model_info["N"])[:, :, :N_inf].reshape(bs, new_dim)  
    
    # the tarp method returns the ECP values for a given set of alpha coverage levels.
    ecp, alpha = run_tarp(
        true_params, # true params of simulations
        simulated_bursts,
        posterior,
        references=None,  # will be calculated automatically.
        num_posterior_samples=1000,
        use_batched_sampling=False,  # `True` can give speed-ups, but can cause memory issues.
    )

    # Similar to SBC, we can check then check whether the distribution of ecp is close to
    # that of alpha.
    atc, ks_pval = check_tarp(ecp, alpha)
    print(atc, "Should be close to 0")
    print(ks_pval, "Should be larger than 0.05")

    save = True
    if save:
        import pandas as pd
        ecp = ecp.cpu().numpy()
        alpha = alpha.cpu().numpy()
        df = pd.DataFrame({
            "ecp":ecp,
            "alpha": alpha
        })
        df.to_csv(f"TARP_result_fm_N_{N_inf}_{job_id}.csv", index=False)


    plot_tarp(ecp, alpha)
    plt.savefig(f'tarp_test_fm_N_{N_inf}_{job_id}.png')

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Run TARP coverage test")

    parser.add_argument("-N","--ncomp", type=int, help="Number of burst components in sampled data", required=True)
    parser.add_argument("-j","--job_id", type=str, help="job id of trained model", required=True)

    args = parser.parse_args()

    main(args.ncomp, args.job_id)