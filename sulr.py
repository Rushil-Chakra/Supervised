from abc import ABC
from typing import Union, Optional, Sequence

import torch
import numpy as np

from bindsnet.network.nodes import SRM0Nodes
from bindsnet.network.topology import (
    AbstractConnection,
    Connection,
    Conv2dConnection,
    LocalConnection,
)
from bindsnet.utils import im2col_indices

from bindsnet.learning import LearningRule

class SupervisedPostPre(LearningRule):
    # language=rst
    """
    STDP rule involving both pre- and post-synaptic spiking activity. The post-synaptic
    update is positive and the pre- synaptic update is negative, and both are dependent
    on the magnitude of the synaptic weights.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        # language=rst
        """
        Constructor for ``WeightDependentPostPre`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the
            ``WeightDependentPostPre`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the batch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs
        )

        assert (
            self.source.traces
        ), "Pre-synaptic nodes must record spike traces."
        assert (
            connection.wmin != -np.inf and connection.wmax != np.inf
        ), "Connection must define finite wmin and wmax."

        self.wmin = connection.wmin
        self.wmax = connection.wmax

        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        elif isinstance(connection, Conv2dConnection):
            self.update = self._conv2d_connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Connection`` subclass of ``AbstractConnection``
        class.
        """
        batch_size = self.source.batch_size
        label = kwargs.get('label', None)
        source_s = self.source.s.view(batch_size, -1).unsqueeze(2).float()
        source_x = self.source.x.view(batch_size, -1).unsqueeze(2)
        target_s = self.target.s.view(batch_size, -1).unsqueeze(1).float()
        target_x = self.target.x.view(batch_size, -1).unsqueeze(1)

        update = 0
        ran = np.random.randint(10) *  10 
        # Pre-synaptic update.
        if self.nu[0]:
            outer_product = self.reduction(
                torch.bmm(source_s, target_x), dim=0
            )
            update -= (
                self.nu[0] * outer_product[:, label+ran] * (self.connection.w[:, label+ran] - self.wmin)
            )

        # Post-synaptic update.
        if self.nu[1]:
            outer_product = self.reduction(
                torch.bmm(source_x, target_s), dim=0
            )
            update += (
                self.nu[1] * outer_product[:, label+ran] * (self.wmax - self.connection.w[:, label+ran])
            )

        self.connection.w[:, label+ran] += update

        super().update()