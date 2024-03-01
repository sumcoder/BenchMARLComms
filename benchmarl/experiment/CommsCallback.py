from tensordict import TensorDictBase, TensorDict
from benchmarl.experiment import Callback


class CommsCallback(Callback):
    def on_train_step(self, batch: TensorDictBase, group: str) -> TensorDictBase:
        # print("we have comms loss!")
        model = self.experiment.group_policies[group][0][0]
        # print(model)
        # opt = self.experiment.op[group]
        value = model._forward(batch.select(model.in_key)).get(
           "bits"
        )[0]
        # print("value", value)

        k = value[..., 0, 0]
        # print('key=',k)
        if value.shape[-1] == 1:
            wv = value[..., 0, 1, 0]
        else:
            wv = value[..., 0, 1]

        # loss_td = TensorDict({"k_loss": loss}, [])
        loss_td = TensorDict({"k_loss": k, 'wv_loss': wv}, [])
        # print(k, 'is comms loss and wv =',wv)

        b = k + wv
        if b:
            b.backward()
        # k.backward()
        # wv.backward()

        # grad_norm = self.experiment._grad_clip(opt)
        # loss_td.set(
        #     f"grad_norm_action_space",
        #     torch.tensor(grad_norm, device=self.experiment.config.train_device),
        # )

        # opt.step()
        # opt.zero_grad()

        return loss_td