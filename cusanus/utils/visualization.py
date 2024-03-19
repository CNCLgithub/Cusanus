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
                                       motion_grids,
                                       gfield_grids)


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

class RenderGFieldVolumes(pl.Callback):
    def __init__(self,
                 n:int = 50,
                 rng =(-4., 4.),
                 ):
        super().__init__()
        self.n = n
        self.rng = rng

    def on_validation_batch_end(self, trainer, exp, outputs, batch, batch_idx,
                                data_loader_idx):

        (qs, ys) = batch
        qs = qs[0].detach().cpu()
        fig = plot_gfield_diff(qs, outputs['pred_diff'])
        path = os.path.join(exp.logger.log_dir, "volumes",
                            f"batch_{batch_idx}_pred_diff.html")
        fig.write_html(path)

    def on_test_batch_end(self, trainer, exp, outputs, batch, batch_idx,
                                data_loader_idx):
        (qs, ys) = batch
        qs = qs[0].detach().cpu()
        ys = ys[0].detach().cpu()
        # fig = plot_gfield_diff(qs, outputs['pred_diff'])
        fig = plot_gfield_diff(qs, ys)
        path = os.path.join(exp.logger.log_dir, "test_volumes",
                            f"batch_{batch_idx}_pred_diff.html")
        fig.write_html(path)
        m = outputs['mod']
        pred_qs = gfield_grids(self.n, self.rng)
        pred_qs = pred_qs.to(exp.device)
        pred_ys = exp.eval_modulation(m, pred_qs).detach().cpu()
        pred_qs = pred_qs.detach().cpu()
        fig = plot_gfield_surface(pred_qs, pred_ys)
        path = os.path.join(exp.logger.log_dir, "test_volumes",
                            f"batch_{batch_idx}_pred.html")
        fig.write_html(path)


def plot_gfield_diff(qs, ys):
    fig = go.Figure( data =
        go.Scatter(
            x = qs[:, 0],
            y = qs[:, 1],
            mode = 'markers',
            marker=dict(
                # size=12,
                color=ys.squeeze(),
                colorbar_title = 'Distance',),
            name = 'Diff: GT - PRED'))
    fig.update_layout(showlegend=True,
                      plot_bgcolor='black',
                      width=800,
                      height=800,
                      scene = dict(
                        xaxis_title='X DIM',
                        yaxis_title='Y DIM'))

    return fig

def plot_gfield_surface(qs, ys):
    fig = go.Figure(data=go.Heatmap(
        x=qs[:, 0].unique(),
        y=qs[:, 1].unique(),
        z=ys.exp().reshape(50, 50),
        # colorscale='Balance',
        ))
    fig.update_layout(showlegend=True,
                      scene = dict(
                        xaxis_title='X DIM',
                        yaxis_title='Y DIM',
                          zaxis_title='OCC'),
                      width=800,
                      height=800
                      )
    fig.update_scenes(aspectmode='cube',
                      aspectratio={'x': 1. , 'y': 1})
    return fig

class RenderKFieldVolumes(pl.Callback):
    def __init__(self,
                 nt:int=8,
                 trange=(0., 1.),
                 nx:int = 20,
                 xrange=(-3., 3.),
                 ny:int = 20,
                 yrange=(-3., 3.),
                 ):
        super().__init__()
        self.nt = nt
        self.trange = trange
        self.nx = nx
        self.xrange = xrange
        self.ny = ny
        self.yrange = yrange

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
        pred_qs = motion_grids(self.nt, self.trange,
                               self.nx, self.xrange,
                               self.ny, self.yrange)
        pred_qs = pred_qs.to(exp.device)
        pred_ys = exp.eval_modulation(m, pred_qs).detach().cpu()
        pred_qs = pred_qs.detach().cpu()
        fig = plot_motion_trace(fit_qs, fit_ys)
        path = os.path.join(exp.logger.log_dir, "test_volumes",
                            f"batch_{batch_idx}_fit.html")
        fig.write_html(path)
        fig = plot_motion_volume(pred_qs, pred_ys)
        path = os.path.join(exp.logger.log_dir, "test_volumes",
                            f"batch_{batch_idx}_pred.html")
        fig.write_html(path)


class RenderEFieldVolumes(pl.Callback):
    def __init__(self,
                 nt:int=8,
                 trange=(0., 1.),
                 nx:int = 20,
                 xrange=(-3., 3.),
                 ny:int = 20,
                 yrange=(-3., 3.),
                 ):
        super().__init__()
        self.nt = nt
        self.trange = trange
        self.nx = nx
        self.xrange = xrange
        self.ny = ny
        self.yrange = yrange

    def on_validation_batch_end(self, trainer, exp, outputs, batch, batch_idx,
                                data_loader_idx):
        (_, ys) = batch
        gt_k1 = ys[0]
        pred_k1 = outputs['pred']
        qs = motion_grids(self.nt, self.trange,
                          self.nx, self.xrange,
                          self.ny, self.yrange)
        qs = qs.to(exp.device)
        gt_ys = exp.kfield.module(qs, gt_k1).detach().cpu()
        pred_ys = exp.kfield.module(qs, pred_k1).detach().cpu()
        qs = qs.detach().cpu()

        fig = plot_motion_volume(qs, gt_ys)
        path = os.path.join(exp.logger.log_dir, "volumes",
                            f"batch_{batch_idx}_gt.html")
        fig.write_html(path)
        fig = plot_motion_volume(qs, pred_ys)
        path = os.path.join(exp.logger.log_dir, "volumes",
                            f"batch_{batch_idx}_pred.html")
        fig.write_html(path)

    def on_test_batch_end(self, trainer, exp, outputs, batch, batch_idx,
                                data_loader_idx):
        (_, ys) = batch
        gt_k1 = ys[0]
        pred_k1 = outputs['pred']
        qs = motion_grids(self.nt, self.trange,
                          self.nx, self.xrange,
                          self.ny, self.yrange)
        qs = qs.to(exp.device)
        gt_ys = exp.kfield.module(qs, gt_k1).detach().cpu()
        pred_ys = exp.kfield.module(qs, pred_k1).detach().cpu()
        qs = qs.detach().cpu()

        fig = plot_motion_volume(qs, gt_ys)
        path = os.path.join(exp.logger.log_dir, "test_volumes",
                            f"batch_{batch_idx}_gt.html")
        fig.write_html(path)
        fig = plot_motion_volume(qs, pred_ys)
        path = os.path.join(exp.logger.log_dir, "test_volumes",
                            f"batch_{batch_idx}_pred.html")
        fig.write_html(path)

def plot_motion_volume(qs, ys):
    slices = qs[:, 0].unique()
    fig = go.Figure(data=go.Volume(
        x=qs[:, 1],
        y=qs[:, 2],
        z=qs[:, 0],
        value=torch.log(ys).squeeze(),
        opacity=0.3, # needs to be small to see through all surfaces
        surface_count=20, # needs to be a large number for good volume rendering
        # autocolorscale = True,
        opacityscale = 'min',
        colorscale='Sunset',
        slices_z=dict(show=True,
                      locations=slices),
        caps= dict(x_show=False, y_show=False, z_show=False), # no caps
        # surface=dict(fill=0.50, pattern='A+B'),
        ))
    fig.update_layout(showlegend=True,
                      scene = dict(
                        xaxis_title='X DIM',
                        yaxis_title='Y DIM',
                        zaxis_title='TIME'))
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
            x = fit_qs[:, 1],
            y = fit_qs[:, 2],
            z = fit_qs[:, 0],
            mode = 'markers',
            marker=dict(
                # size=12,
                color=fit_ys.squeeze(),
                colorscale='Sunset',
                colorbar_title = 'L2 distance',),
            name = 'Fit'))
    fig.update_layout(showlegend=True,
                      scene = dict(
                        xaxis_title='X DIM',
                        yaxis_title='Y DIM',
                        zaxis_title='TIME'))

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
