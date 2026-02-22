import torch

from internal import stepfun


def compute_alpha_weights_oball(density_obj,density_oball,tdist,dirs, opaque_background=False):
    """Helper function for computing alpha compositing weights."""

    density_expand_obj=density_obj.unsqueeze(-1).repeat_interleave(repeats=3,dim=-1) \
        if density_obj.shape[-1]!=3 else density_obj
    density_expand_oball=density_oball
    t_delta = tdist[..., 1:] - tdist[..., :-1]
    delta = t_delta * torch.norm(dirs[..., None, :], dim=-1)
    delta = delta.unsqueeze(-1)
    density_delta_obj = density_expand_obj * delta
    density_delta_oball =density_expand_oball * delta
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

def volumetric_rendering_oball(rgbs,
                            rgbs_backscatter,
                             weights,
                               weights_direct,
                               weights_backscatter,
                               weights_forward,
                            tdist,
                             t_far,
                             compute_extras,
                             extras=None):

    eps = torch.finfo(rgbs.dtype).eps
    # eps = 1e-3
    rendering = {}
    t_mids = 0.5 * (tdist[..., :-1] + tdist[..., 1:])
    rgbs_all=rgbs
    acc = (weights_direct).sum(dim=-2)
    acc_all = (weights_direct+weights_backscatter+weights_forward).sum(dim=-2)
    acc_avg = (acc).mean(dim=-1,keepdim = True)
    bg_w = (1 - acc_avg).clamp_min(0.)  # The weight of the background.
    # med_bg = torch.tensor([[0.1921,0.2941,0.4588]]).cuda()
    # med_bg = torch.tensor([[0.1725,0.2471,0.306]]).cuda()
    # med_bg = torch.tensor([[0.1451, 0.2314, 0.3490]]).cuda()
    # med_bg = torch.tensor([[0.6196, 0.6, 0.6235]]).cuda()
    med_bg = torch.tensor([[0.1216,0.2902,0.498]]).cuda()
    # med_bg = torch.tensor([[0.1451, 0.1843, 0.129]]).cuda()
    rgb = ((weights_direct * rgbs_all).sum(dim=-2) +(weights_forward * rgbs_all).sum(dim=-2) +(weights_backscatter*rgbs_backscatter).sum(dim=-2))
    # rgb = ((weights_oball * rgbs_all).sum(dim=-2) + (weights_uwall * rgbs_uw).sum(dim=-2) + bg_w*med_bg)
    rgb_backscatter = (weights_backscatter*rgbs_backscatter).sum(dim=-2)
    rgb_forward = (weights_forward * rgbs_all).sum(dim=-2)
    rgb_direct = (weights_direct * rgbs_all).sum(dim=-2)
    depth = (
        torch.clip(
            torch.nan_to_num((torch.mean(weights_direct,dim=-1) * t_mids).sum(dim=-1) / torch.mean(acc_all,dim=-1).clamp_min(eps), torch.inf),
            tdist[..., 0], tdist[..., -1]))

    rendering['rgb'] = rgb
    rendering['rgb_bg_med'] = rgb_backscatter
    rendering['rgb_med'] = rgb_forward
    rendering['rgb_obj'] =rgb_direct
    rendering['rgbs_uw'] = rgbs_backscatter
    rendering['depth'] = depth
    rendering['acc'] = torch.mean(acc,dim=-1)

    if compute_extras:
        if extras is not None:
            for k, v in extras.items():
                if v is not None:
                    rendering[k] = (torch.mean(weights_direct,dim=-1) * v).sum(dim=-2)

        expectation = lambda x: (torch.mean(weights_direct,dim=-1) * x).sum(dim=-1) / torch.mean(acc,dim=-1).clamp_min(eps)
        # For numerical stability this expectation is computing using log-distance.
        rendering['distance_mean'] = (
            torch.clip(
                torch.nan_to_num(torch.exp(expectation(torch.log(t_mids))), torch.inf),
                tdist[..., 0], tdist[..., -1]))

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
