import os
import torch
import torchvision
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from cusanus.pytypes import *
from cusanus.utils.coordinates import (grids_along_depth,
                                       grids_along_axis,
                                       motion_grids)


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


class RenderGField(pl.Callback):
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
            path = os.path.join(exp.logger.log_dir, "volumes",
                                "latest_volume.html")
            fig.write_html(path)
            fig = plot_volume_slice(pred_qs, pred_ys, self.samples)
            path = os.path.join(exp.logger.log_dir, "volumes",
                                f"epoch_{exp.current_epoch}" + \
                                f"_batch_{batch_idx}_sliced.html")
            fig.write_html(path)
            path = os.path.join(exp.logger.log_dir, "volumes",
                                "latest_slice.html")
            fig.write_html(path)

class RenderKField(pl.Callback):
    def __init__(self,
                 nt:int=10,
                 nxyz:int=30,
                 batch_step:int=5,
                 delta:float=1.0):
        super().__init__()
        self.nt = nt
        self.nxyz = nxyz
        self.batch_step = batch_step
        self.delta = delta

    def on_validation_batch_end(self, trainer, exp, outputs, batch, batch_idx,
                                data_loader_idx):
        (qs, ys) = batch
        qs = qs[0].detach().cpu()
        ys = ys[0].detach().cpu()
        pred_ys = outputs['pred'].detach().cpu()
        path = os.path.join(exp.logger.log_dir, "volumes",
                            "latest" + \
                            f"_batch_{batch_idx}.html")
        fig.write_html(path)

    def on_test_batch_end(self, trainer, exp, outputs, batch, batch_idx,
                                data_loader_idx):
        (qs, ys) = batch
        qs = qs[0].detach().cpu()
        ys = ys[0].detach().cpu()
        pred_ys = outputs['pred'].detach().cpu()
        path = os.path.join(exp.logger.log_dir, "test_volumes",
                            f"_batch_{batch_idx}.html")
        fig.write_html(path)


class RenderKFieldVolumes(pl.Callback):
    def __init__(self,
                 nt:int=8,
                 nxyz:int=20,
                 delta:float=2.0):
        super().__init__()
        self.nt = nt
        self.nxyz = nxyz
        self.delta = delta

    def on_validation_batch_end(self, trainer, exp, outputs, batch, batch_idx,
                                data_loader_idx):
        (qs, ys) = batch
        fit_qs = qs[0].detach().cpu()
        fit_ys = outputs['pred']
        fig = plot_motion_trace(fit_qs, fit_ys)
        path = os.path.join(exp.logger.log_dir, "volumes",
                            f"batch_{batch_idx}.html")
        fig.write_html(path)

    def on_test_batch_end(self, trainer, exp, outputs, batch, batch_idx,
                                data_loader_idx):
        (qs, ys) = batch
        fit_qs = qs[0].detach().cpu()
        fit_ys = outputs['pred']
        m = outputs['mod']
        pred_qs = motion_grids(self.nt,
                               self.nxyz,
                               delta = self.delta)
        pred_qs = pred_qs.to(exp.device)
        pred_ys = exp.eval_modulation(m, pred_qs).detach().cpu()
        pred_qs = pred_qs.detach().cpu()
        fig = plot_motion_trace(fit_qs, fit_ys)
        path = os.path.join(exp.logger.log_dir, "test_volumes",
                            f"batch_{batch_idx}_fit.html")
        fig.write_html(path)
        fig = plot_volume(pred_qs, pred_ys)
        path = os.path.join(exp.logger.log_dir, "test_volumes",
                            f"batch_{batch_idx}_pred.html")
        fig.write_html(path)

def plot_volume(qs, ys):
    slices = qs[:, 0].unique()
    fig = go.Figure(data=go.Volume(
        x=qs[:, 0],
        y=qs[:, 1],
        z=qs[:, 3],
        value=torch.log(ys).squeeze(),
        opacity=0.3, # needs to be small to see through all surfaces
        surface_count=20, # needs to be a large number for good volume rendering
        # autocolorscale = True,
        opacityscale = 'min',
        colorscale='Sunset',
        slices_x=dict(show=True,
                      locations=slices),
        caps= dict(x_show=False, y_show=False, z_show=False), # no caps
        # surface=dict(fill=0.50, pattern='A+B'),
        ))
    fig.update_layout(showlegend=True,
                      scene = dict(
                        xaxis_title='TIME',
                        yaxis_title='X DIM',
                        zaxis_title='Z DIM'))
    return fig

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


def plot_motion_trace(fit_qs, fit_ys):
    fig = go.Figure( data =
        go.Scatter3d(
            x = fit_qs[:, 0],
            y = fit_qs[:, 1],
            z = fit_qs[:, 3],
            mode = 'markers',
            marker=dict(
                # size=12,
                color=fit_ys.squeeze(),
                colorscale='Sunset',
                colorbar_title = 'L2 distance',),
            name = 'Fit'))
    fig.update_layout(showlegend=True,
                      scene = dict(
                        xaxis_title='TIME',
                        yaxis_title='X DIM',
                        zaxis_title='Z DIM'))

    return fig

def plot_3D_heatmap(qs:Tensor, ys:Tensor, t:int,
                    **plot_args):

    volume = ys.reshape((t, -1))
    coords = qs.reshape((t, -1, 4))
    ts = coords[:, 0, 0]

    fig = go.Figure(frames=[
        go.Frame( data=go.Heatmap(
            x=coords[k, :, 1],
            y=coords[k, :, 3],
            z=volume[k],
            type = 'heatmap',
            colorscale = 'Sunset'
        ),
        name=str(k))
    for k in range(t)])

    # Add data to be displayed before animation starts
    fig.add_trace(go.Heatmap(
            x=coords[0, :, 1],
            y=coords[0, :, 3],
            z=volume[0],
            type = 'heatmap',
            colorscale = 'Sunset'
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

    # Layout
    fig.update_layout(
            title='Slices in volumetric data',
            width=600,
            height=600,
            updatemenus = [
                {
                    "buttons": [
                        {
                            "args": [None, frame_args(t)],
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
