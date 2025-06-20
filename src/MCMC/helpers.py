from typing import Iterable

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