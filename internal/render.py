import os.path

from internal import stepfun
from internal import math
from internal import utils
import torch
import torch.nn.functional as F


def lift_gaussian(d, t_mean, t_var, r_var, diag):
    """Lift a Gaussian defined along a ray to 3D coordinates."""
    mean = d[..., None, :] * t_mean[..., None]
    eps = torch.finfo(d.dtype).eps
    # eps = 1e-3
    d_mag_sq = torch.sum(d ** 2, dim=-1, keepdim=True).clamp_min(eps)

    if diag:
        d_outer_diag = d ** 2
        null_outer_diag = 1 - d_outer_diag / d_mag_sq
        t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
        xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
        cov_diag = t_cov_diag + xy_cov_diag
        return mean, cov_diag
    else:
        d_outer = d[..., :, None] * d[..., None, :]
        eye = torch.eye(d.shape[-1], device=d.device)
        null_outer = eye - d[..., :, None] * (d / d_mag_sq)[..., None, :]
        t_cov = t_var[..., None, None] * d_outer[..., None, :, :]
        xy_cov = r_var[..., None, None] * null_outer[..., None, :, :]
        cov = t_cov + xy_cov
        return mean, cov


def conical_frustum_to_gaussian(d, t0, t1, base_radius, diag, stable=True):
    """Approximate a conical frustum as a Gaussian distribution (mean+cov).

  Assumes the ray is originating from the origin, and base_radius is the
  radius at dist=1. Doesn't assume `d` is normalized.

  Args:
    d: the axis of the cone
    t0: the starting distance of the frustum.
    t1: the ending distance of the frustum.
    base_radius: the scale of the radius as a function of distance.
    diag: whether or the Gaussian will be diagonal or full-covariance.
    stable: whether or not to use the stable computation described in
      the paper (setting this to False will cause catastrophic failure).

  Returns:
    a Gaussian (mean and covariance).
  """
    if stable:
        # Equation 7 in the paper (https://arxiv.org/abs/2103.13415).
        mu = (t0 + t1) / 2  # The average of the two `t` values.
        hw = (t1 - t0) / 2  # The half-width of the two `t` values.
        eps = torch.finfo(d.dtype).eps
        # eps = 1e-3
        t_mean = mu + (2 * mu * hw ** 2) / (3 * mu ** 2 + hw ** 2).clamp_min(eps)
        denom = (3 * mu ** 2 + hw ** 2).clamp_min(eps)
        t_var = (hw ** 2) / 3 - (4 / 15) * hw ** 4 * (12 * mu ** 2 - hw ** 2) / denom ** 2
        r_var = (mu ** 2) / 4 + (5 / 12) * hw ** 2 - (4 / 15) * (hw ** 4) / denom
    else:
        # Equations 37-39 in the paper.
        t_mean = (3 * (t1 ** 4 - t0 ** 4)) / (4 * (t1 ** 3 - t0 ** 3))
        r_var = 3 / 20 * (t1 ** 5 - t0 ** 5) / (t1 ** 3 - t0 ** 3)
        t_mosq = 3 / 5 * (t1 ** 5 - t0 ** 5) / (t1 ** 3 - t0 ** 3)
        t_var = t_mosq - t_mean ** 2
    r_var *= base_radius ** 2
    return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cylinder_to_gaussian(d, t0, t1, radius, diag):
    """Approximate a cylinder as a Gaussian distribution (mean+cov).

  Assumes the ray is originating from the origin, and radius is the
  radius. Does not renormalize `d`.

  Args:
    d: the axis of the cylinder
    t0: the starting distance of the cylinder.
    t1: the ending distance of the cylinder.
    radius: the radius of the cylinder
    diag: whether or the Gaussian will be diagonal or full-covariance.

  Returns:
    a Gaussian (mean and covariance).
  """
    t_mean = (t0 + t1) / 2
    r_var = radius ** 2 / 4
    t_var = (t1 - t0) ** 2 / 12
    return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cast_rays(tdist, origins, directions, cam_dirs, radii, rand=True, n=7, m=3, std_scale=0.5, **kwargs):
    """Cast rays (cone- or cylinder-shaped) and featurize sections of it.

  Args:
    tdist: float array, the "fencepost" distances along the ray.
    origins: float array, the ray origin coordinates.
    directions: float array, the ray direction vectors.
    radii: float array, the radii (base radii for cones) of the rays.
    ray_shape: string, the shape of the ray, must be 'cone' or 'cylinder'.
    diag: boolean, whether or not the covariance matrices should be diagonal.

  Returns:
    a tuple of arrays of means and covariances.
  """
    t0 = tdist[..., :-1, None]
    t1 = tdist[..., 1:, None]
    radii = radii[..., None]

    t_m = (t0 + t1) / 2
    t_d = (t1 - t0) / 2+1e-5

    j = torch.arange(6, device=tdist.device)
    t = t0 + t_d / (t_d ** 2 + 3 * t_m ** 2) * (t1 ** 2 + 2 * t_m ** 2 + 3 / 7 ** 0.5 * (2 * j / 5 - 1) * (
        (t_d ** 2 - t_m ** 2) ** 2 + 4 * t_m ** 4).sqrt())

    deg = torch.pi / 3 * torch.tensor([0, 2, 4, 3, 5, 1], device=tdist.device, dtype=torch.float)
    deg = torch.broadcast_to(deg, t.shape)
    if rand:
        # randomly rotate and flip
        mask = torch.rand_like(t0[..., 0]) > 0.5
        deg = deg + 2 * torch.pi * torch.rand_like(deg[..., 0])[..., None]
        deg = torch.where(mask[..., None], deg, torch.pi * 5 / 3 - deg)
    else:
        # rotate 30 degree and flip every other pattern
        mask = torch.arange(t.shape[-2], device=tdist.device) % 2 == 0
        mask = torch.broadcast_to(mask, t.shape[:-1])
        deg = torch.where(mask[..., None], deg, deg + torch.pi / 6)
        deg = torch.where(mask[..., None], deg, torch.pi * 5 / 3 - deg)
    means = torch.stack([
        radii * t * torch.cos(deg) / 2 ** 0.5,
        radii * t * torch.sin(deg) / 2 ** 0.5,
        t
    ], dim=-1)
    stds = std_scale * radii * t / 2 ** 0.5

    # two basis in parallel to the image plane
    rand_vec = torch.randn_like(cam_dirs)
    ortho1 = F.normalize(torch.cross(cam_dirs, rand_vec, dim=-1), dim=-1)
    ortho2 = F.normalize(torch.cross(cam_dirs, ortho1, dim=-1), dim=-1)

    # just use directions to be the third vector of the orthonormal basis,
    # while the cross section of cone is parallel to the image plane
    basis_matrix = torch.stack([ortho1, ortho2, directions], dim=-1)
    means = math.matmul(means, basis_matrix[..., None, :, :].transpose(-1, -2))
    means = means + origins[..., None, None, :]
    # import trimesh
    # trimesh.Trimesh(means.reshape(-1, 3).detach().cpu().numpy()).export("test.ply", "ply")

    return means, stds, t


def compute_alpha_weights(density, tdist, dirs, opaque_background=False):
    """Helper function for computing alpha compositing weights."""
    t_delta = tdist[..., 1:] - tdist[..., :-1]
    delta = t_delta * torch.norm(dirs[..., None, :], dim=-1)
    density_delta = density * delta

    if opaque_background:
        # Equivalent to making the final t-interval infinitely wide.
        density_delta = torch.cat([
            density_delta[..., :-1],
            torch.full_like(density_delta[..., -1:], torch.inf)
        ], dim=-1)

    alpha = 1 - torch.exp(-density_delta)
    trans = torch.exp(-torch.cat([
        torch.zeros_like(density_delta[..., :1]),
        torch.cumsum(density_delta[..., :-1], dim=-1)
    ], dim=-1))
    weights = alpha * trans
    return weights, alpha, trans



def volumetric_rendering(rgbs,
                         weights,
                         weights_s,
                         tdist,
                         sdist,
                         bg_rgbs,
                         t_far,
                         compute_extras,
                         extras=None):
    """Volumetric Rendering Function.

  Args:
    rgbs: color, [batch_size, num_samples, 3]
    weights: weights, [batch_size, num_samples].
    tdist: [batch_size, num_samples].
    bg_rgbs: the color(s) to use for the background.
    t_far: [batch_size, 1], the distance of the far plane.
    compute_extras: bool, if True, compute extra quantities besides color.
    extras: dict, a set of values along rays to render by alpha compositing.

  Returns:
    rendering: a dict containing an rgb image of size [batch_size, 3], and other
      visualizations if compute_extras=True.
  """
    eps = torch.finfo(rgbs.dtype).eps
    # eps = 1e-3
    rendering = {}

    acc = weights.sum(dim=-1)
    bg_w = (1 - acc[..., None]).clamp_min(0.)  # The weight of the background.
    rgb = (weights[..., None] * rgbs).sum(dim=-2) + bg_w * bg_rgbs
    t_mids = 0.5 * (tdist[..., :-1] + tdist[..., 1:])
    s_mids = 0.5 * (sdist[..., :-1] + sdist[..., 1:])
    depth = (
        torch.clip(
            torch.nan_to_num((weights * t_mids).sum(dim=-1) / acc.clamp_min(eps), torch.inf),
            tdist[..., 0], tdist[..., -1]))
    depth_s = (
        torch.clip(
            torch.nan_to_num((weights_s * s_mids).sum(dim=-1) / acc.clamp_min(eps), torch.inf),
            sdist[..., 0], sdist[..., -1]))
    rendering['rgb'] = rgb
    rendering['depth'] = depth
    rendering['acc'] = acc
    rendering['depth_s'] = depth_s
    if compute_extras:
        if extras is not None:
            for k, v in extras.items():
                if v is not None:
                    rendering[k] = (weights[..., None] * v).sum(dim=-2)

        expectation = lambda x: (weights * x).sum(dim=-1) / acc.clamp_min(eps)
        # For numerical stability this expectation is computing using log-distance.
        rendering['distance_mean'] = (
            torch.clip(
                torch.nan_to_num(torch.exp(expectation(torch.log(t_mids))), torch.inf),
                tdist[..., 0], tdist[..., -1]))

        # Add an extra fencepost with the far distance at the end of each ray, with
        # whatever weight is needed to make the new weight vector sum to exactly 1
        # (`weights` is only guaranteed to sum to <= 1, not == 1).
        t_aug = torch.cat([tdist, t_far], dim=-1)
        weights_aug = torch.cat([weights, bg_w], dim=-1)

        ps = [5, 50, 95]
        distance_percentiles = stepfun.weighted_percentile(t_aug, weights_aug, ps)

        for i, p in enumerate(ps):
            s = 'median' if p == 50 else 'percentile_' + str(p)
            rendering['distance_' + s] = distance_percentiles[..., i]

    return rendering


def volumetric_rendering_all(rgbs,
                             rgbs_new,
                             weights,
                            weights_all,
                            tdist,
                             tdist_new,
                             bg_rgbs,
                             t_far,
                             compute_extras,
                             extras=None):

    eps = torch.finfo(rgbs.dtype).eps
    # eps = 1e-3
    rendering = {}
    t_mids = 0.5 * (tdist[..., :-1] + tdist[..., 1:])
    t_mids_new = 0.5 * (tdist_new[..., :-1] + tdist_new[..., 1:])
    t_all = torch.sort(torch.cat([tdist, tdist_new], dim=-1), dim=-1).values
    t_mids_all=torch.cat([t_mids,t_mids_new],dim=-1)
    t_mids_sort=torch.sort(t_mids_all,dim=-1)
    sort_indices=t_mids_sort.indices
    t_mids_all=t_mids_sort.values
    rgbs_all=torch.cat([rgbs,rgbs_new],dim=-2)
    rgbs_all_0=torch.gather(rgbs_all[...,0],dim=-1,index=sort_indices)
    rgbs_all_1 = torch.gather(rgbs_all[..., 1], dim=-1, index=sort_indices)
    rgbs_all_2 = torch.gather(rgbs_all[..., 2], dim=-1, index=sort_indices)
    rgbs_all=torch.stack([rgbs_all_0,rgbs_all_1,rgbs_all_2],dim=-1)
    acc = weights_all.sum(dim=-2)
    bg_w = (1 - acc).clamp_min(0.)  # The weight of the background.
    rgb = (weights_all * rgbs_all).sum(dim=-2) + bg_w * bg_rgbs
    depth = (
        torch.clip(
            torch.nan_to_num((torch.mean(weights_all,dim=-1) * t_mids_all).sum(dim=-1) / torch.mean(acc,dim=-1).clamp_min(eps), torch.inf),
            t_all[..., 0], t_all[..., -1]))
    rendering['rgb'] = rgb
    rendering['depth'] = depth
    rendering['acc'] = acc

    if compute_extras:
        if extras is not None:
            for k, v in extras.items():
                if v is not None:
                    rendering[k] = (torch.mean(weights_all,dim=-1) * v).sum(dim=-2)

        expectation = lambda x: (torch.mean(weights_all,dim=-1) * x).sum(dim=-1) / torch.mean(acc,dim=-1).clamp_min(eps)
        # For numerical stability this expectation is computing using log-distance.
        rendering['distance_mean'] = (
            torch.clip(
                torch.nan_to_num(torch.exp(expectation(torch.log(t_mids_all))), torch.inf),
                t_all[..., 0], t_all[..., -1]))

        # Add an extra fencepost with the far distance at the end of each ray, with
        # whatever weight is needed to make the new weight vector sum to exactly 1
        # (`weights` is only guaranteed to sum to <= 1, not == 1).
        t_aug = torch.cat([tdist, t_far], dim=-1)
        weights_aug = torch.cat([weights, torch.mean(bg_w,dim=-1,keepdim=True)], dim=-1)

        ps = [5, 50, 95]
        distance_percentiles = stepfun.weighted_percentile(t_aug, weights_aug, ps)

        for i, p in enumerate(ps):
            s = 'median' if p == 50 else 'percentile_' + str(p)
            rendering['distance_' + s] = distance_percentiles[..., i]

    return rendering

def compute_alpha_weights_new(density, density_new, tdist, tdist_new,dirs, opaque_background=False):
    """Helper function for computing alpha compositing weights."""
    t_mids = 0.5 * (tdist[..., :-1] + tdist[..., 1:])
    t_mids_new = 0.5 * (tdist_new[..., :-1] + tdist_new[..., 1:])
    t_mids_all = torch.cat([t_mids, t_mids_new], dim=-1)
    t_mids_sort = torch.sort(t_mids_all, dim=-1)
    sort_indices = t_mids_sort.indices
    density_zero = torch.zeros_like(density_new).cuda()
    density_expand = density.unsqueeze(-1).repeat_interleave(repeats=3,dim=-1)
    density_all = torch.cat([density_expand, density_new], dim=-2)
    density_all_zero = torch.cat([density_expand, density_zero], dim=-2)
    density_all_zero_0=torch.gather(density_all_zero[...,0],index=sort_indices,dim=-1)
    density_all_zero_1 = torch.gather(density_all_zero[..., 1], index=sort_indices, dim=-1)
    density_all_zero_2 = torch.gather(density_all_zero[..., 2], index=sort_indices, dim=-1)
    density_all_zero = torch.stack([density_all_zero_0,density_all_zero_1,density_all_zero_2],dim=-1)
    tdist_all = torch.cat([tdist, tdist_new], dim=-1)
    tdist_all = torch.sort(tdist_all, dim=-1).values[..., :-1]
    t_delta = tdist_all[..., 1:] - tdist_all[..., :-1]
    delta = t_delta * torch.norm(dirs[..., None, :], dim=-1)
    delta = delta.unsqueeze(-1)
    density_delta_zero = density_all_zero * delta
    if opaque_background:
        # Equivalent to making the final t-interval infinitely wide.
        density_delta_zero = torch.cat([
            density_delta_zero[..., :-1, :],
            torch.full_like(density_delta_zero[..., -1:, :], torch.inf)
        ], dim=-2)

    alpha = 1 - torch.exp(-density_delta_zero)
    trans = torch.exp(-torch.cat([
        torch.zeros_like(density_delta_zero[..., :1, :]),
        torch.cumsum(density_delta_zero[..., :-1, :], dim=-2)
    ], dim=-2))
    weights = alpha * trans
    return weights, alpha, trans

def volumetric_rendering_oball(rgbs,
                             rgbs_new,
                            rgbs_uwob,
                             weights,
                               weights_oball,
                               weights_uwall,
                            tdist,
                             tdist_new,
                             bg_rgbs,
                             t_far,
                             compute_extras,
                             extras=None):

    eps = torch.finfo(rgbs.dtype).eps
    # eps = 1e-3
    rendering = {}
    t_mids = 0.5 * (tdist[..., :-1] + tdist[..., 1:])
    t_mids_new = 0.5 * (tdist_new[..., :-1] + tdist_new[..., 1:])
    t_all = torch.sort(torch.cat([tdist, tdist_new], dim=-1), dim=-1).values
    t_mids_all=torch.cat([t_mids,t_mids_new],dim=-1)
    t_mids_sort=torch.sort(t_mids_all,dim=-1)
    sort_indices=t_mids_sort.indices
    t_mids_all=t_mids_sort.values
    rgbs_all=torch.cat([rgbs,rgbs_new],dim=-2)
    rgbs_all_0 = torch.gather(rgbs_all[..., 0], dim=-1, index=sort_indices)
    rgbs_all_1 = torch.gather(rgbs_all[..., 1], dim=-1, index=sort_indices)
    rgbs_all_2 = torch.gather(rgbs_all[..., 2], dim=-1, index=sort_indices)
    rgbs_all = torch.stack([rgbs_all_0, rgbs_all_1, rgbs_all_2], dim=-1)
    rgbs_uw=torch.cat([rgbs_uwob,torch.zeros_like(rgbs_new)],dim=-2)
    rgbs_uw_0 = torch.gather(rgbs_uw[..., 0], dim=-1, index=sort_indices)
    rgbs_uw_1 = torch.gather(rgbs_uw[..., 1], dim=-1, index=sort_indices)
    rgbs_uw_2 = torch.gather(rgbs_uw[..., 2], dim=-1, index=sort_indices)
    rgbs_uw = torch.stack([rgbs_uw_0, rgbs_uw_1, rgbs_uw_2], dim=-1)
    acc = weights_oball.sum(dim=-2)
    bg_w = (1 - acc).clamp_min(0.)  # The weight of the background.
    rgb = ((weights_oball * rgbs_all).sum(dim=-2) + bg_w * bg_rgbs + (weights_uwall * rgbs_uw).sum(dim=-2))
    depth = (
        torch.clip(
            torch.nan_to_num((torch.mean(weights_oball,dim=-1) * t_mids_all).sum(dim=-1) / torch.mean(acc,dim=-1).clamp_min(eps), torch.inf),
            t_all[..., 0], t_all[..., -1]))
    rendering['rgb'] = rgb
    rendering['depth'] = depth
    rendering['acc'] = acc

    if compute_extras:
        if extras is not None:
            for k, v in extras.items():
                if v is not None:
                    rendering[k] = (torch.mean(weights_oball,dim=-1) * v).sum(dim=-2)

        expectation = lambda x: (torch.mean(weights_oball,dim=-1) * x).sum(dim=-1) / torch.mean(acc,dim=-1).clamp_min(eps)
        # For numerical stability this expectation is computing using log-distance.
        rendering['distance_mean'] = (
            torch.clip(
                torch.nan_to_num(torch.exp(expectation(torch.log(t_mids_all))), torch.inf),
                t_all[..., 0], t_all[..., -1]))

        # Add an extra fencepost with the far distance at the end of each ray, with
        # whatever weight is needed to make the new weight vector sum to exactly 1
        # (`weights` is only guaranteed to sum to <= 1, not == 1).
        t_aug = torch.cat([tdist, t_far], dim=-1)
        weights_aug = torch.cat([weights, torch.mean(bg_w,dim=-1,keepdim=True)], dim=-1)

        ps = [5, 50, 95]
        distance_percentiles = stepfun.weighted_percentile(t_aug, weights_aug, ps)

        for i, p in enumerate(ps):
            s = 'median' if p == 50 else 'percentile_' + str(p)
            rendering['distance_' + s] = distance_percentiles[..., i]

    return rendering

def compute_alpha_weights_oball(density_obj,density_oball, density_new, tdist, tdist_new,dirs, opaque_background=False):
    """Helper function for computing alpha compositing weights."""
    t_mids = 0.5 * (tdist[..., :-1] + tdist[..., 1:])
    t_mids_new = 0.5 * (tdist_new[..., :-1] + tdist_new[..., 1:])
    t_mids_all = torch.cat([t_mids, t_mids_new], dim=-1)
    t_mids_sort = torch.sort(t_mids_all, dim=-1)
    sort_indices = t_mids_sort.indices
    density_expand_obj=density_obj.unsqueeze(-1).repeat_interleave(repeats=3,dim=-1) \
        if density_obj.shape[-1]!=3 else density_obj
    density_expand_oball=density_oball
    density_all_obj = torch.cat([density_expand_obj, density_new], dim=-2)
    density_all_obj_0=torch.gather(density_all_obj[...,0],dim=-1,index=sort_indices)
    density_all_obj_1=torch.gather(density_all_obj[...,1],dim=-1,index=sort_indices)
    density_all_obj_2=torch.gather(density_all_obj[...,2],dim=-1,index=sort_indices)
    density_all_obj=torch.stack([density_all_obj_0,density_all_obj_1,density_all_obj_2],dim=-1)
    density_all_oball = torch.cat([density_expand_oball, density_new], dim=-2)
    density_all_oball_0 = torch.gather(density_all_oball[..., 0], dim=-1, index=sort_indices)
    density_all_oball_1 = torch.gather(density_all_oball[..., 1], dim=-1, index=sort_indices)
    density_all_oball_2 = torch.gather(density_all_oball[..., 2], dim=-1, index=sort_indices)
    density_all_oball = torch.stack([density_all_oball_0, density_all_oball_1, density_all_oball_2], dim=-1)
    tdist_all = torch.cat([tdist, tdist_new], dim=-1)
    tdist_all = torch.sort(tdist_all, dim=-1).values[...,:-1]
    t_delta = tdist_all[..., 1:] - tdist_all[..., :-1]
    delta = t_delta * torch.norm(dirs[..., None, :], dim=-1)
    delta = delta.unsqueeze(-1)
    density_delta_obj = density_all_obj * delta
    density_delta_oball =density_all_oball * delta
    if opaque_background:
        # Equivalent to making the final t-interval infinitely wide.
        density_delta_obj = torch.cat([
            density_delta_obj[..., :-1,:],
            torch.full_like(density_delta_obj[..., -1:,:], torch.inf)
        ], dim=-2)
        density_delta_oball = torch.cat([
            density_delta_oball[..., :-1, :],
            torch.full_like(density_delta_oball[..., -1:, :], torch.inf)
        ], dim=-2)

    alpha = 1 - torch.exp(-density_delta_obj)
    trans = torch.exp(-torch.cat([
        torch.zeros_like(density_delta_oball[..., :1,:]),
        torch.cumsum(density_delta_oball[..., :-1,:], dim=-2)
    ], dim=-2))
    weights = alpha * trans
    return weights, alpha, trans