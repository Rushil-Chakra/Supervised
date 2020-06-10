import torch
from typing import Optional, Tuple, Dict


def vfa(
    spikes: torch.Tensor,
    labels: torch.Tensor,
    n_labels: int,
    rates: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # lnaguage=rst
    """
    Classify data with the highest average spiking activity over all neurons,
    wieghted by calss-wise proportion. Based on VFA from <https://arxiv.org/abs/2001.01680>.

    :param spikes: Binary tensor of shape ``(n_samples, time, n_neurons)`` of a single
        layer's spiking activity or a binary tensor of shape ``(n_samples, time, n_total)``
        of the concatenation of several layers' spiking activity.
    :param labels: Vector of shape ``(n_samples,)`` with data labels corresponding to
        spiking activity.
    :param n_labels: The number of target labels in the data.
    :param rates: If passed, these represent spike rates from a previous
        ``vfa()`` call.
    :param alpha: Rate of decay of label assignments.
    :return: Tuple of predictions tensor of shape ``(n_samples,)`` resulting from the "Vote for All" 
        classification scheme and per-class firing rates.
    """
    n_neurons = spikes.size(2)

    if rates is None:
        rates = torch.zeros(n_neurons, n_labels)

    # Sum over time dimension (spike ordering doesn't matter).
    spikes = spikes.sum(1)

    for i in range(n_labels):
        # Count the number of samples with this label.
        n_labeled = torch.sum(labels == i).float()

        if n_labeled > 0:
            # Get indices of samples with this label.
            indices = torch.nonzero(labels == i).view(-1)
            # Compute average firing rates for this label.
            rates[:, i] = alpha * rates[:, i] + (
                torch.sum(spikes[indices], 0) / n_labeled
            )
    # Compute proportions of spike activity per class.
    proportions = rates / rates.sum(1, keepdim=True)
    proportions[proportions != proportions] = 0  # Set NaNs to 0

    votes = torch.matmul(spikes, proportions)
    predictions = torch.sort(votes, dim=1, descending=True)[1][:, 0]
    return predictions, rates
