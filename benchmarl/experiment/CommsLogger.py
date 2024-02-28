from tensordict import TensorDictBase
from benchmarl.experiment import Callback
import torch

class CommsLogger(Callback):
    """
    Callback to log some training metrics
    """

    def on_batch_collected(self, batch: TensorDictBase):
        key = 'bits'
        to_log = {}
        # print(batch.get("bits"))

        value = batch.get('bits', None)#[...,0,:2]
        k = value[...,0,0]
        if value.shape[-1] == 1:
            wv = value[...,0,1,0]
        else:
            wv = value[...,0,1]
        if value is not None:
            to_log.update(
                {
                    # "/".join(("collection",) + (key)): batch.get(key)
                    'collection/k_bits': k,
                    'collection/wv_bits': wv
                }
            )
        self.experiment.logger.log(
            to_log,
            step=self.experiment.n_iters_performed,
        )
