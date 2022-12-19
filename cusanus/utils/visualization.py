import os
import torch
import torchvision
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
import plotly.graph_objects as go
from cusanus.pytypes import *
from cusanus.utils import grids_along_depth, grids_along_axis


def aggregrate_depth_scans(qs : Tensor, ps : Tensor,
                           ny : int, nxz : int):
    # a (ny x m x 2) queries
    # where m is the number of pixels in the image
    # and the last dimension denotes the x-y image coords
    img_coords = qs.reshape(ny, -1, 3)[:, :, [0, 2]]
    max_c = img_coords.abs().max()
    img_coords /= max_c
    img_coords = torch.floor((img_coords + 0.99) * 0.5 * nxz)
    img_coords = img_coords.long()
    img_vals = ps.reshape(ny, -1)
    # # number of channels (usually 1)
    # (_, nc) = ps.shape
    batched_imgs = ps.new_empty((ny, 1, nxz, nxz))
    for b in range(ny):
        for (k, (x,y)) in enumerate(img_coords[b]):
            batched_imgs[b, :, x, y] = img_vals[b, k]

    return torchvision.utils.make_grid(batched_imgs,
                                       nrow = int(ny**(0.5)))

def plot_volume(qs, ys, **plot_args):
    fig = go.Figure(data=go.Volume(
        x=qs[:, 0],
        y=qs[:, 1],
        z=qs[:, 2],
        value=ys,
        # isomin=-5.0,
        # isomax=3.0,
        opacity=0.2, # needs to be small to see through all surfaces
        surface_count=20, # needs to be a large number for good volume rendering
        autocolorscale = True,
        **plot_args,
        ))
    return fig


class RenderGeometry(pl.Callback):
    def __init__(self,
                 samples:int=30,
                 batch_step:int=5,
                 delta:float=3.0):
        super().__init__()
        self.samples = samples
        self.batch_step = batch_step
        self.delta = delta

    def on_train_batch_end(self, trainer, exp, outputs, batch, batch_idx):
        # Skip for all other epochs
        if (batch_idx % self.batch_step) == 0:
            (qs, ys) = batch
            qs = qs[0]
            ys = ys[0]
            pred_qs = grids_along_axis(self.samples,
                                       self.samples,
                                       delta = self.delta)
            pred_qs = pred_qs.to(exp.device)
            m = exp.fit_modulation(qs, ys)
            pred_ys = exp.eval_modulation(m, pred_qs).detach().cpu()
            print(pred_ys.min(), pred_ys.max(), pred_ys.mean())
            pred_qs = pred_qs.detach().cpu()
            fig = plot_volume(pred_qs, pred_ys)

            path = os.path.join(exp.logger.log_dir, "volumes",
                                f"epoch_{exp.current_epoch}" + \
                                f"_batch_{batch_idx}.html")
            fig.write_html(path)

# adapted from https://plotly.com/python/visualizing-mri-volume-slices/
def plot_volume_slice(qs:Tensor, ys:Tensor, n : int):

    volume = ys.reshape((n,n,n))
    coords = qs.reshape((n,n,n,3))
    # (n n 2)
    axis_coords = coords[:, 0, 0, 0]
    plane_coords = coords[0, :n, :n, 1:]
    d1 = plane_coords[:, :, 0]
    d2 = plane_coords[:, :, 1]

    fig = go.Figure(frames=[
        go.Frame(
            data=go.Surface(
                z= torch.tile(axis_coords[k], (n,n)),
                surfacecolor=volume[k],
                x = d1,
                y = d2,
                colorscale='Gray',
            ),
        name=str(k))
    for k in range(n)])

    # Add data to be displayed before animation starts
    fig.add_trace(go.Surface(
        z= torch.tile(axis_coords[0], (n,n)),
        surfacecolor=volume[0],
        x = d1,
        y = d2,
        colorscale='Gray',
        cmin=0, cmax=1,
        colorbar=dict(thickness=20, ticklen=4)
        ))

    def frame_args(duration):
        return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
            }

    sliders = [
                {
                    "pad": {"b": 10, "t": 60},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[f.name], frame_args(0)],
                            "label": str(k),
                            "method": "animate",
                        }
                        for k, f in enumerate(fig.frames)
                    ],
                }
            ]

    min_axis = axis_coords.min()
    max_axis = axis_coords.max()
    # Layout
    fig.update_layout(
            title='Slices in volumetric data',
            width=600,
            height=600,
            scene=dict(
                        zaxis=dict(range=[min_axis, max_axis],
                                   autorange=False),
                        aspectratio=dict(x=1, y=1, z=1),
                        ),
            updatemenus = [
                {
                    "buttons": [
                        {
                            "args": [None, frame_args(n)],
                            "label": "&#9654;", # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "&#9724;", # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
            ],
            sliders=sliders
    )
    return fig
