from typing import Iterable

def update_modelparams(sample, inf_params, modelparams):
    """
    Extract parameters from flat sample to update parameters dictionary.
    """
    N = modelparams['ncomp']
    for i, key in enumerate(inf_params):
        start = i * N
        stop  = start + N
        modelparams['burstparams'][key] = sample[start:stop]
    return modelparams

def gen_parameter_labels(inf_params, N) -> Iterable[str]:
    """
    Generates parameter labels of shape t0_1, t0_2, amp_1, amp_2, etc.
    """
    labels = []
    for key in inf_params:
        for i in range(N):
            label = f"{key}_{i+1}"
            labels.append(label)

    return labels