import os
import numpy as np
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from cusanus.pytypes import *
from cusanus.utils.coordinates import (grids_along_depth,
                                       grids_along_axis,
                                       motion_grids,
                                       gfield_grids)

class RenderKFieldVolumes:
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
