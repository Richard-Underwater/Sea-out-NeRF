import collections
import functools
import ot
import torch.optim
import torch.nn.functional as F
from torch import Tensor

from internal import camera_utils
from internal import configs
from internal import datasets
from internal import image
from internal import math
from internal import models
from internal import ref_utils
from internal import stepfun
from internal import utils
import numpy as np
from torch.utils._pytree import tree_map, tree_flatten
from torch_scatter import segment_coo


class GradientScaler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, colors, sigmas, ray_dist):
        ctx.save_for_backward(ray_dist)
        return colors, sigmas

    @staticmethod
    def backward(ctx, grad_output_colors, grad_output_sigmas):
        (ray_dist,) = ctx.saved_tensors
        scaling = torch.square(ray_dist).clamp(0, 1)
        return grad_output_colors * scaling[..., None], grad_output_sigmas * scaling, None


def tree_reduce(fn, tree, initializer=0):
    return functools.reduce(fn, tree_flatten(tree)[0], initializer)


def tree_sum(tree):
    return tree_reduce(lambda x, y: x + y, tree, initializer=0)


def tree_norm_sq(tree):
    return tree_sum(tree_map(lambda x: torch.sum(x ** 2), tree))


def tree_norm(tree):
    return torch.sqrt(tree_norm_sq(tree))


def tree_abs_max(tree):
    return tree_reduce(
        lambda x, y: max(x, torch.abs(y).max().item()), tree, initializer=0)


def tree_len(tree):
    return tree_sum(tree_map(lambda z: np.prod(z.shape), tree))


def summarize_tree(tree, fn, ancestry=(), max_depth=3):
    """Flatten 'tree' while 'fn'-ing values and formatting keys like/this."""
    stats = {}
    for k, v in tree.items():
        name = ancestry + (k,)
        stats['/'.join(name)] = fn(v)
        if hasattr(v, 'items') and len(ancestry) < (max_depth - 1):
            stats.update(summarize_tree(v, fn, ancestry=name, max_depth=max_depth))
    return stats


def compute_data_loss(batch, renderings, config):
    """Computes data loss terms for RGB, normal, and depth outputs."""
    data_losses = []
    stats = collections.defaultdict(lambda: [])

    # lossmult can be used to apply a weight to each ray in the batch.
    # For example: masking out rays, applying the Bayer mosaic mask, upweighting
    # rays from lower resolution images and so on.
    lossmult = batch['lossmult']
    lossmult = torch.broadcast_to(lossmult, batch['rgb'][..., :3].shape)
    if config.disable_multiscale_loss:
        lossmult = torch.ones_like(lossmult)

    for rendering in renderings:
        resid_sq = (rendering['rgb'] - batch['rgb'][..., :3]) ** 2
        denom = lossmult.sum()
        stats['mses'].append(((lossmult * resid_sq).sum() / denom).item())

        if config.data_loss_type == 'mse':
            # Mean-squared error (L2) loss.
            data_loss = resid_sq
        elif config.data_loss_type == 'charb':
            # Charbonnier loss.
            data_loss = torch.sqrt(resid_sq + config.charb_padding ** 2)
        elif config.data_loss_type == 'rawnerf':
            # Clip raw values against 1 to match sensor overexposure behavior.
            rgb_render_clip = rendering['rgb'].clamp_max(1)
            resid_sq_clip = (rgb_render_clip - batch['rgb'][..., :3]) ** 2
            # Scale by gradient of log tonemapping curve.
            scaling_grad = 1. / (1e-3 + rgb_render_clip.detach())
            # Reweighted L2 loss.
            data_loss = resid_sq_clip * scaling_grad ** 2
        else:
            assert False
        data_losses.append((lossmult * data_loss).sum() / denom)

        if config.compute_disp_metrics:
            # Using mean to compute disparity, but other distance statistics can
            # be used instead.
            disp = 1 / (1 + rendering['distance_mean'])
            stats['disparity_mses'].append(((disp - batch['disps']) ** 2).mean().item())

        if config.compute_normal_metrics:
            if 'normals' in rendering:
                weights = rendering['acc'] * batch['alphas']
                normalized_normals_gt = ref_utils.l2_normalize(batch['normals'])
                normalized_normals = ref_utils.l2_normalize(rendering['normals'])
                normal_mae = ref_utils.compute_weighted_mae(weights, normalized_normals,
                                                            normalized_normals_gt)
            else:
                # If normals are not computed, set MAE to NaN.
                normal_mae = torch.nan
            stats['normal_maes'].append(normal_mae.item())

    loss = (
            config.data_coarse_loss_mult * sum(data_losses[:-1]) +
            config.data_loss_mult * data_losses[-1])

    stats = {k: np.array(stats[k]) for k in stats}
    return loss, stats

def grayworld_loss(renderings_detach):
    loss=0
    rgb=renderings_detach[-1]['rgb']
    rgb_mean=torch.mean(rgb,dim=[0,1,2])
    Drg = torch.pow(rgb_mean[0]-rgb_mean[1],2)
    Drb = torch.pow(rgb_mean[0]-rgb_mean[2],2)
    Dgb = torch.pow(rgb_mean[1]-rgb_mean[2],2)
    k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)
    return k*0.1

def interlevel_loss(ray_history, config):
    """Computes the interlevel loss defined in mip-NeRF 360."""
    # Stop the gradient from the interlevel loss onto the NeRF MLP.
    last_ray_results = ray_history[-1]
    c = last_ray_results['sdist'].detach()
    w = last_ray_results['weights'].detach()
    loss_interlevel = 0.
    for ray_results in ray_history[:-1]:
        cp = ray_results['sdist']
        wp = ray_results['weights']
        loss_interlevel += stepfun.lossfun_outer(c, w, cp, wp).mean()
    return config.interlevel_loss_mult * loss_interlevel


def anti_interlevel_loss(ray_history, config):
    """Computes the interlevel loss defined in mip-NeRF 360."""
    last_ray_results = ray_history[-1]
    c = last_ray_results['sdist'].detach()
    w = last_ray_results['weights'].detach()
    w_normalize = w / (c[..., 1:] - c[..., :-1])
    loss_anti_interlevel = 0.
    for i, ray_results in enumerate(ray_history[:-1]):
        cp = ray_results['sdist']
        wp = ray_results['weights']
        c_, w_ = stepfun.blur_stepfun(c, w_normalize, config.pulse_width[i])

        # piecewise linear pdf to piecewise quadratic cdf
        area = 0.5 * (w_[..., 1:] + w_[..., :-1]) * (c_[..., 1:] - c_[..., :-1])

        cdf = torch.cat([torch.zeros_like(area[..., :1]), torch.cumsum(area, dim=-1)], dim=-1)

        # query piecewise quadratic interpolation
        cdf_interp = math.sorted_interp_quad(cp, c_, w_, cdf)
        # difference between adjacent interpolated values
        w_s = torch.diff(cdf_interp, dim=-1)

        loss_anti_interlevel += ((w_s - wp).clamp_min(0) ** 2 / (wp + 1e-5)).mean()
    return config.anti_interlevel_loss_mult * loss_anti_interlevel

def density_loss(ray_history, threshold):
    density_all = 0
    for ray_h in ray_history:
        density = ray_h['density']
        density_all+=torch.mean(density)
    density_average=density_all/len(ray_history)
    return 0.05*torch.maximum(threshold-density_average,torch.tensor(0.0))

def distortion_loss(ray_history, config):
    """Computes the distortion loss regularizer defined in mip-NeRF 360."""
    last_ray_results = ray_history[-1]
    c = last_ray_results['sdist']
    w = last_ray_results['weights']
    loss = stepfun.lossfun_distortion(c, w).mean()
    return config.distortion_loss_mult * loss


def orientation_loss(batch, model, ray_history, config):
    """Computes the orientation loss regularizer defined in ref-NeRF."""
    total_loss = 0.
    for i, ray_results in enumerate(ray_history):
        w = ray_results['weights']
        n = ray_results[config.orientation_loss_target]
        if n is None:
            raise ValueError('Normals cannot be None if orientation loss is on.')
        # Negate viewdirs to represent normalized vectors from point to camera.
        v = -1. * batch['viewdirs']
        n_dot_v = (n * v[..., None, :]).sum(dim=-1)
        loss = (w * n_dot_v.clamp_min(0) ** 2).sum(dim=-1).mean()
        if i < model.num_levels - 1:
            total_loss += config.orientation_coarse_loss_mult * loss
        else:
            total_loss += config.orientation_loss_mult * loss
    return total_loss


def hash_decay_loss(ray_history, config):
    total_loss = 0.
    for i, ray_results in enumerate(ray_history):
        total_loss += config.hash_decay_mults * ray_results['loss_hash_decay']
    return total_loss


def opacity_loss(renderings, config):
    total_loss = 0.
    for i, rendering in enumerate(renderings):
        o = rendering['acc']
        total_loss += config.opacity_loss_mult * (-o * torch.log(o + 1e-5)).mean()
    return total_loss


def predicted_normal_loss(model, ray_history, config):
    """Computes the predicted normal supervision loss defined in ref-NeRF."""
    total_loss = 0.
    for i, ray_results in enumerate(ray_history):
        w = ray_results['weights']
        n = ray_results['normals']
        n_pred = ray_results['normals_pred']
        if n is None or n_pred is None:
            raise ValueError(
                'Predicted normals and gradient normals cannot be None if '
                'predicted normal loss is on.')
        loss = torch.mean((w * (1.0 - torch.sum(n * n_pred, dim=-1))).sum(dim=-1))
        if i < model.num_levels - 1:
            total_loss += config.predicted_normal_coarse_loss_mult * loss
        else:
            total_loss += config.predicted_normal_loss_mult * loss
    return total_loss


def clip_gradients(model, accelerator, config):
    """Clips gradients of MLP based on norm and max value."""
    if config.grad_max_norm > 0 and accelerator.sync_gradients:
        accelerator.clip_grad_norm_(model.parameters(), config.grad_max_norm)

    if config.grad_max_val > 0 and accelerator.sync_gradients:
        accelerator.clip_grad_value_(model.parameters(), config.grad_max_val)

    for param in model.parameters():
        if param.grad is not None:
            param.grad.nan_to_num_()


def create_optimizer(config: configs.Config, model):
    """Creates optax optimizer for model training."""
    adam_kwargs = {
        'betas': [config.adam_beta1, config.adam_beta2],
        'eps': config.adam_eps,
    }
    lr_kwargs = {
        'max_steps': config.max_steps,
        'lr_delay_steps': config.lr_delay_steps,
        'lr_delay_mult': config.lr_delay_mult,
    }

    lr_fn_main = lambda step: math.learning_rate_decay(
        step,
        lr_init=config.lr_init,
        lr_final=config.lr_final,
        **lr_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_init, **adam_kwargs)

    return optimizer, lr_fn_main

def density_zero_loss(ray_history,depth):
    sdist=ray_history[-1]['sdist']
    s_mids = 0.5 * (sdist[..., :-1] + sdist[..., 1:])
    density=ray_history[-1]['density']
    depth=depth.detach()
    depth_expand=depth.unsqueeze(-1).expand_as(s_mids)
    loss=(density*((s_mids<depth_expand).type(torch.float32))).mean()
    return 0.1*loss

def acc_loss(trans):
    data_loss=0
    data_loss+=torch.sqrt(torch.mean(-torch.log(torch.exp(-torch.abs(1-trans)/0.1)+torch.exp(-torch.abs(trans)/0.1))))
    return data_loss*0.001

def contrast_loss(renderings,batch):
    rendering=renderings[-1]
    I=batch['rgb'][..., :3]
    I_new=I.permute(1,2,3,0).squeeze(dim=0)
    J=rendering['rgb']
    J_new=J.permute(1,2,3,0).squeeze(dim=0)
    I_avg=torch.nn.AvgPool1d(kernel_size=16,stride=16)(I_new)
    I_avg=torch.nn.Upsample(scale_factor=16)(I_avg).unsqueeze(dim=0).permute(3,0,1,2)
    J_avg=torch.nn.AvgPool1d(kernel_size=16,stride=16)(J_new)
    J_avg = torch.nn.Upsample(scale_factor=16)(J_avg).unsqueeze(dim=0).permute(3,0,1,2)
    I_loss = torch.sqrt(((I - I_avg)**2)).sum()/I.shape[0]
    J_loss = torch.sqrt(((J - J_avg) ** 2)).sum()/J.shape[0]
    return (I_loss-J_loss)*0.0001

def entropy_min_loss(ray_history):
    weights=ray_history[-1]['weights']
    log=-torch.log(weights+1e-5)
    single=weights*log
    return 0.001*torch.sum(single,dim=-1).mean()

def blue_loss(rgb):
    rgb_loss = (rgb[...,2]-rgb[...,0]).clamp_min(0.0).mean()+(rgb[...,2]-rgb[...,1]).clamp_min(0.0).mean()
    return rgb_loss

def sliced_wasserstein_loss(x, y, num_projections=64):
    """
    x, y: [N, D] 或任意 shape（会 flatten）
    num_projections: 投影方向数量
    可微，不会生成 NxN 的矩阵，显存极低。
    """
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    N, D = x.shape
    loss = 0.0

    for _ in range(num_projections):
        # 随机方向（1D 就是 ±1）
        proj = torch.randn(D, 1, device=x.device)
        proj = proj / torch.norm(proj)

        # 投影到 1D
        xp = x @ proj
        yp = y @ proj

        # 排序后做 1D Wasserstein
        xs = torch.sort(xp.reshape(-1))[0]
        ys = torch.sort(yp.reshape(-1))[0]

        loss += torch.mean(torch.abs(xs - ys))

    return loss / num_projections


def depth_ranking_loss(pred_depth,gt_depth,num_pairs=4096,margin=0.0):
    d1 = pred_depth.view(-1)
    d2 = gt_depth.view(-1)
    d2 = -d2
    N = d1.numel()
    idx_i = torch.randint(0,N,(num_pairs,),device = d1.device)
    idx_j = torch.randint(0, N, (num_pairs,), device=d1.device)
    d1_i,d1_j = d1[idx_i],d1[idx_j]
    d2_i,d2_j = d2[idx_i],d2[idx_j]
    d1_diff = d1_i - d1_j
    d2_diff = d2_i - d2_j
    loss = F.relu(margin - d1_diff*d2_diff)
    return loss.mean()
