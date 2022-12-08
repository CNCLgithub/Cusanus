import os
import torch
import torchvision
import pytorch_lightning as pl
from cusanus.pytypes import *
from cusanus.utils import grids_along_depth


def aggregrate_depth_scans(qs : Tensor, ps : Tensor,
                           ny : int, nxz : int):
    # a (ny x m x 2) queries
    # where m is the number of pixels in the image
    # and the last dimension denotes the x-y image coords
    img_coords = qs.reshape(ny, -1, 3)[:, :, [0, 2]]
    img_coords = torch.floor((img_coords + 0.99) * 0.5 * nxz)
    img_coords = img_coords.long()
    img_vals = ps.reshape(ny, -1)
    # # number of channels (usually 1)
    # (_, nc) = ps.shape
    batched_imgs = ps.new_empty((ny, 1, nxz, nxz))
    for b in range(ny):
        for (k, (x,y)) in enumerate(img_coords[b]):
            batched_imgs[b, :, x, y] = img_vals[b, k]

        # coords = img_coords[b]
        # xcs = coords[:, 0]
        # ycs = coords[:, 1]
        # batched_imgs[b, :, xcs, ycs] = img_vals[b]

    return torchvision.utils.make_grid(batched_imgs,
                                       nrow = int(ny**(0.5)))

class RenderGeometry(pl.Callback):
    def __init__(self,
                 batch_size:int=9,
                 img_dim:int=128,
                 epoch_step:int=5):
        super().__init__()
        self.batch_size = batch_size
        self.img_dim = img_dim

        self.epoch_step = epoch_step
    def on_train_batch_end(self, trainer, exp, outputs, batch, batch_idx):
        # Skip for all other epochs
        if (batch_idx == 0) and trainer.current_epoch % self.epoch_step == 0:
            (qs, ys) = batch
            qs = qs[0]
            ys = ys[0]
            pred_qs = grids_along_depth(self.batch_size,
                                        self.img_dim)
            pred_qs = pred_qs.to(exp.device)
            m = exp.fit_modulation(qs, ys)
            pred_ys = exp.eval_modulation(m, pred_qs).detach().cpu()
            print(pred_ys.min(), pred_ys.max())
            self.log('min_y', pred_ys.min())
            self.log('max_y', pred_ys.max())
            pred_qs = pred_qs.cpu()
            grid = aggregrate_depth_scans(pred_qs, pred_ys,
                                          self.batch_size,
                                          self.img_dim)

            img_path = os.path.join(exp.logger.log_dir, "reconstructions",
                                    f"recons_{exp.logger.name}_Epoch_{exp.current_epoch}.png")
            torchvision.utils.save_image(grid, img_path, normalize=False)
