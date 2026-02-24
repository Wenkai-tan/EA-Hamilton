import torch
import torch.nn as nn
from kaolin.physics.simplicits.utils import weight_function_lbs
from kaolin.physics.utils.finite_diff import finite_diff_jac
import kaolin.physics.materials.utils as material_utils
from functools import partial
import kaolin.physics.materials.neohookean_elastic_material as neohookean_elastic_material


def compute_losses_H(model, normalized_pts, yms, prs, rhos, batch_size, num_handles, appx_vol, num_samples, le_coeff, lo_coeff):
    r""" Perform a step of the symplectic training process

    Args:
        model (nn.module): network
        normalized_pts (torch.Tensor): Spatial points in :math:`R^3`, of shape :math:`(\text{num_pts}, 3)`
        yms (torch.Tensor): Point-wise youngs modulus, of shape :math:`(\text{num_pts})` 
        prs (torch.Tensor): Point-wise poissons ratio, of shape :math:`(\text{num_pts})`
        rhos (torch.Tensor): Point-wise density, of shape :math:`(\text{num_pts})`
        batch_size (int): Number of sample deformations
        num_handles (int): Number of handles
        appx_vol (float): Approximate volume of object
        num_samples (int): Number of sample points. 
        lh_coeff (float): floating point coefficient for Hamiltonian loss 
        lo_coeff (float): floating point coefficient for orthogonal loss

    Returns:
        torch.Tensor, torch.Tensor: The elastic loss term, the orthogonality losses terms
    """
    batch_transforms = 0.1 * torch.randn(batch_size, num_handles, 3, 4,
                                         dtype=normalized_pts.dtype).to(normalized_pts.device)

    # Select num_samples from all points
    sample_indices = torch.randint(low=0, high=normalized_pts.shape[0], size=(
        num_samples,), device=normalized_pts.device)

    sample_pts = normalized_pts[sample_indices]
    sample_yms = yms[sample_indices]
    sample_prs = prs[sample_indices]
    sample_rhos = rhos[sample_indices]

    # Get current skinning weights at sample pts
    weights = model(sample_pts)

    # Calculate elastic energy for the batch of transforms
    lh = le_coeff * loss_h(model, sample_pts, sample_yms, sample_prs,
                                 sample_rhos, batch_transforms, appx_vol)

    # Calculate orthogonality of skinning weights
    lo = lo_coeff * loss_ortho(weights)

    return lh, lo

def loss_ortho(weights):
    r"""Calculate orthogonality of weights

    Args:
        weights (torch.Tensor): Tensor of weights, of shape :math:`(\text{num_samples}, \text{num_handles})`

    Returns:
        torch.Tensor: Orthogonality loss, single value tensor
    """
    return nn.MSELoss()(weights.T @ weights, torch.eye(weights.shape[1], device=weights.device))

def loss_h(model, pts, yms, prs, rhos, transforms, appx_vol):
    r"""Calculate symplectic Hamiltonian loss for training.

    Args:
        model (nn.Module): object network
        pts (torch.Tensor): Tensor of sample points in R^dim, for now dim=3, of shape :math:`(\text{num_samples}, \text{dim})`
        yms (torch.Tensor): Length pt-wise youngs modulus, of shape :math:`(\text{num_samples})`
        prs (torch.Tensor): Length pt-wise poisson ratios, of shape :math:`(\text{num_samples})`
        rhos (torch.Tensor): Length pt-wise density, of shape :math:`(\text{num_samples})`
        transforms (torch.Tensor): Batch of transformation matrix, of shape :math:`(\text{batch_size}, \text{num_handles}, \text{dim}, \text{dim}+1)`
        appx_vol (float): Approximate (or exact) volume of object (in :math:`m^3`)

    Returns:
        torch.Tensor: Hamiltonian loss, single value tensor
    """

    DT = 0.05

    mus, lams = material_utils.to_lame(yms, prs)

    # partial_weight_fcn_lbs = partial(weight_function_lbs, tfms=transforms, fcn=model)
    partial_weight_fcn_lbs = partial(weight_function_lbs, tfms=transforms, fcn=model)
    pt_wise_Fs = finite_diff_jac(partial_weight_fcn_lbs, pts)
    pt_wise_Fs = pt_wise_Fs[:, :, 0]

    # shape (N, B, 3, 3)
    N, B = pt_wise_Fs.shape[0:2]

    # shape (N, B, 1)
    mus = mus.expand(N, B).unsqueeze(-1)
    lams = lams.expand(N, B).unsqueeze(-1)

    # ramps up from 100% linear elasticity to 100% neohookean elasticity
    # lin_elastic = (1 - interp_step) * linear_elastic_material.linear_elastic_energy(mus, lams, pt_wise_Fs)
    neo_elastic = neohookean_elastic_material.neohookean_energy(mus, lams, pt_wise_Fs)

    # KINETIC ENERGY (new - from deformation)
    # kinetic = compute_kinetic_energy_from_invariants(pt_wise_Fs, rhos, DT)
    kinetic = compute_kinetic_energy_from_lbs(pts, partial_weight_fcn_lbs(pts), rhos, DT) # [N, B]


    # weighted average (since we uniformly sample, this is uniform for now)
    # Hamiltonian = elastic energy + kinetic energy (in this sinario: τ∇ + ρ(f − a) = 0)
    return (appx_vol / pts.shape[0]) * (neo_elastic.sum() + kinetic.sum())



# Use Principal Invariants of F
# def compute_kinetic_energy_from_invariants(defo_grad, rhos, dt):
#     r"""Compute kinetic energy from deformation gradient invariants.
    
#     Args:
#         defo_grad (torch.Tensor): Batched deformation gradients (denoted in literature as F) of 3 or more dimensions, :math:`(\text{batch_dims}, 3, 3)`
#         rhos (torch.Tensor): Length pt-wise density, of shape :math:`(\text{num_samples})`
#         dt (float): Time step for the simulation, used to compute velocity magnitude
    
#     Returns:
#         torch.Tensor: Kinetic energy per integration primitive, of shape :math:`(\text{
#     """
    

#     dimensions = defo_grad.shape
#     batched_dims = dimensions[:-2]
#     batched_trace = torch.vmap(torch.trace)
    
#     # Better approach: invariants of F^T·F
#     FtF = torch.matmul(torch.transpose(defo_grad, -2, -1), defo_grad)

#     cauchy_green_strains = FtF.reshape(batched_dims.numel(), 3, 3)
#     # I1 = batched_trace(cauchy_green_strains).reshape(batched_dims).unsqueeze(-1)
#     I1 = batched_trace(cauchy_green_strains).reshape(batched_dims)
    
#     # Second invariant: I2 = 0.5*((tr(F^T·F))² - tr((F^T·F)²))
#     FtF_squared = FtF @ FtF
#     I2 = 0.5 * (I1**2 - torch.diagonal(FtF_squared, dim1=-2, dim2=-1).sum(-1))
    
#     # Third invariant: I3 = det(F^T·F)
#     I3 = torch.det(FtF)  # (N, B)
    
#     # Velocity magnitude from rate of change of invariants
#     # |v|² ∝ (I1 - 3)/dt² + (I2 - 3)/dt² + (I3 - 1)/dt²
#     # (since for identity: I1=3, I2=3, I3=1)
    
#     velocity_squared = ((I1 - 3)**2 + (I2 - 3)**2 + (I3 - 1)**2) / (dt**2)
    
#     rhos_expanded = rhos.expand_as(velocity_squared)
#     kinetic_energy = 0.5 * rhos_expanded * velocity_squared
    
#     return kinetic_energy.unsqueeze(-1)

# Use x_deformed position
def compute_kinetic_energy_from_lbs(x0, x_map_x0, rhos, dt):
    r"""Compute kinetic energy from deformed positions.
    
    Args:
        x0 (torch.Tensor): Sample points in R^3, of shape :math:`(\text{num_samples}, 3)`
        x_map_x0 (torch.Tensor): Skinning weighted transform on x0, of shape :math:`(\text{num_samples}, \text{num_batches}, 3)`
        rhos (torch.Tensor): Physical parameter, of shape :math:`(\\text{num_samples}, 1)`
        dt (float): Virtual time step for training process, used to estimate velocity magnitude        

    Returns: 
        torch.Tensor: Kinetic energy per integration primitive, of shape :math:`(\text{num_samples}, \text{num_batches})`
    """
    x0_expanded = x0.unsqueeze(1)  # (N, 1, 3)
    velocities = (x_map_x0.squeeze(2) - x0_expanded) / dt  # (N, B, 3)

    velocity_sq = (velocities ** 2).sum(dim=-1)  # (N, B)

    return 0.5 * rhos * velocity_sq # (N, B)