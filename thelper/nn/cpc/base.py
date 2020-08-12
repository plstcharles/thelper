import typing

import numpy as np
import torch

import thelper.nn.resnet
import thelper.nn.cpc.pixelcnn


class CPC(thelper.nn.Module):

    def __init__(
            self,
            task: thelper.tasks.Task,
            resnet_block: typing.AnyStr = "thelper.nn.resnet.BasicBlock",
            resnet_layers: typing.Sequence = [3, 4, 6, 3],
            resnet_strides: typing.Sequence = [1, 2, 2, 2],
            resnet_conv1_config: typing.Sequence = [7, 2, 3],
            resnet_input_channels: int = 3,
            context_n_iters: int = 5,
            latents_ch: int = 1024,
            target_ch: int = 64,
            context_bottleneck_ch: int = 256,
            emb_scale: float = 0.1,
            patch_coords_key: typing.AnyStr = "patch_coords",
            patch_dim_count: typing.Sequence = [6, 6],
    ):
        super().__init__(task, **{k: v for k, v in vars().items() if k not in ["self", "task", "__class__"]})
        self.backbone = thelper.nn.resnet.ResNet(
            task=None,  # just get a clean backbone
            block=resnet_block,
            layers=resnet_layers,
            strides=resnet_strides,
            conv1_config=resnet_conv1_config,
            input_channels=resnet_input_channels,
            flexible_input_res=True,
            inplanes=latents_ch // 8,
        )
        self.pixelcnn = thelper.nn.cpc.pixelcnn.PixelCNN(
            n_iters=context_n_iters,
            input_ch=latents_ch,
            bottleneck_ch=context_bottleneck_ch,
        )
        self.target_proj_layer = torch.nn.Conv2d(
            in_channels=latents_ch,
            out_channels=target_ch,
            kernel_size=(1, 1),
        )
        self.context_proj_layer = torch.nn.Conv2d(
            in_channels=latents_ch,
            out_channels=target_ch,
            kernel_size=(1, 1),
        )
        self.target_ch = target_ch
        self.emb_scale = emb_scale
        self.patch_coords_key = patch_coords_key
        self.patch_dim_count = patch_dim_count

    def forward(self, sample):
        assert isinstance(sample, dict), "cpc impl requires full sample with patch idxs"
        in_patches = sample[self.task.input_key]
        in_shape = in_patches.shape
        assert len(in_shape) == 5  # [B x P x C x H x W]
        assert self.patch_coords_key in sample, "missing patch coords key in sample"
        assert in_shape[1] == sample[self.patch_coords_key].shape[1]
        assert np.multiply.reduce(self.patch_dim_count) == in_shape[1], \
            "unexpected patch dim count / latents patch shape mismatch"

        in_patches = in_patches.reshape((-1, *in_shape[2:]))  # combine patches with batch dim
        latents = self.backbone.get_embedding(in_patches, pool=True)
        latents = latents.reshape((in_shape[0], -1, *self.patch_dim_count))  # shape back into grid

        context = self.pixelcnn(latents)  # note: this pixelcnn supports top-to-bottom predictions only
        targets = self.target_proj_layer(latents)
        preds = self.context_proj_layer(context)
        batch_dim, ch_, rows_, cols_ = targets.shape
        assert ch_ == self.target_ch
        targets = targets.permute([1, 0, 2, 3]).contiguous().reshape((self.target_ch, -1))

        losses = []
        for step_start_idx in range(rows_ - 1):
            for step_idx in range(step_start_idx + 1, rows_):
                step_dim = rows_ - step_idx - 1
                total_samples = batch_dim * step_dim * cols_
                preds_i2 = preds[:, :, :-(step_idx + 1), :] * self.emb_scale
                preds_i3 = preds_i2.permute([0, 2, 3, 1]).contiguous().reshape((-1, self.target_ch))
                logits = torch.matmul(preds_i3, targets)
                batch_idxs = torch.arange(total_samples) // (step_dim * cols_)
                sample_idxs = torch.arange(total_samples) % (step_dim * cols_)
                labels = batch_idxs * rows_ * cols_ + (step_idx + 1) * cols_ + sample_idxs
                loss = torch.nn.functional.cross_entropy(logits, labels.long().to(logits.device))
                if not torch.isnan(loss):
                    losses.append(loss)
                else:
                    assert step_idx == rows_ - 1
        return torch.stack(losses).sum()

    def set_task(self, task):
        raise AssertionError("cannot set task on cpc model, should extract backbone instead")


if __name__ == "__main__":

    thelper.utils.init_logger()
    config = thelper.utils.load_config("/home/perf6/dev/thelper/configs/example-bigearthnet-cpc.yml")
    task, train_loader, valid_loader, test_loader = thelper.data.utils.create_loaders(config)
    model = thelper.nn.create_model(config, task=task)

    sample = next(iter(train_loader))
    loss_ = model(sample)

    print("okie")
