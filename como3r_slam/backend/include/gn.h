#pragma once

#include <torch/extension.h>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DEVICE(x, y) TORCH_CHECK(x.device().type() == y.device().type(), #x " must be be on same device as " #y)

// TODO: How to remove these redundant functions?

// Forward declaration
std::vector<torch::Tensor> gauss_newton_points_cuda(
  torch::Tensor Twc, torch::Tensor Xs, torch::Tensor Cs,
  torch::Tensor ii, torch::Tensor jj, 
  torch::Tensor idx_ii2jj, torch::Tensor valid_match,
  torch::Tensor Q,
  const float sigma_point,
  const float C_thresh,
  const float Q_thresh,
  const int max_iter,
  const float delta_thresh);

std::vector<torch::Tensor> gauss_newton_points(
  torch::Tensor Twc, torch::Tensor Xs, torch::Tensor Cs,
  torch::Tensor ii, torch::Tensor jj, 
  torch::Tensor idx_ii2jj, torch::Tensor valid_match,
  torch::Tensor Q,
  const float sigma_point,
  const float C_thresh,
  const float Q_thresh,
  const int max_iter,
  const float delta_thresh);

std::vector<torch::Tensor> gauss_newton_rays_cuda(
  torch::Tensor Twc, torch::Tensor Xs, torch::Tensor Cs,
  torch::Tensor ii, torch::Tensor jj, 
  torch::Tensor idx_ii2jj, torch::Tensor valid_match,
  torch::Tensor Q,
  const float sigma_ray,
  const float sigma_dist,
  const float C_thresh,
  const float Q_thresh,
  const int max_iter,
  const float delta_thresh);

std::vector<torch::Tensor> gauss_newton_rays(
  torch::Tensor Twc, torch::Tensor Xs, torch::Tensor Cs,
  torch::Tensor ii, torch::Tensor jj, 
  torch::Tensor idx_ii2jj, torch::Tensor valid_match,
  torch::Tensor Q,
  const float sigma_ray,
  const float sigma_dist,
  const float C_thresh,
  const float Q_thresh,
  const int max_iter,
  const float delta_thresh);


  // Forward declaration
std::vector<torch::Tensor> gauss_newton_calib_cuda(
  torch::Tensor Twc, torch::Tensor Xs, torch::Tensor Cs,
  torch::Tensor K,
  torch::Tensor ii, torch::Tensor jj, 
  torch::Tensor idx_ii2jj, torch::Tensor valid_match,
  torch::Tensor Q,
  const int height, const int width,
  const int pixel_border,
  const float z_eps,
  const float sigma_pixel, const float sigma_depth,
  const float C_thresh,
  const float Q_thresh,
  const int max_iter,
  const float delta_thresh);

std::vector<torch::Tensor> gauss_newton_calib(
  torch::Tensor Twc, torch::Tensor Xs, torch::Tensor Cs,
  torch::Tensor K,
  torch::Tensor ii, torch::Tensor jj, 
  torch::Tensor idx_ii2jj, torch::Tensor valid_match,
  torch::Tensor Q,
  const int height, const int width,
  const int pixel_border,
  const float z_eps,
  const float sigma_pixel, const float sigma_depth,
  const float C_thresh,
  const float Q_thresh,
  const int max_iter,
  const float delta_thresh);

std::vector<torch::Tensor> gauss_newton_seg_depths_cuda(
  torch::Tensor poses, torch::Tensor depth_init, torch::Tensor rays,
  torch::Tensor Cs,
  torch::Tensor labels, torch::Tensor seg_offsets, torch::Tensor seg_kf,
  torch::Tensor ii, torch::Tensor jj,
  torch::Tensor idx_ii2jj, torch::Tensor valid_match,
  torch::Tensor Q,
  torch::Tensor bnd_seg_a, torch::Tensor bnd_seg_b,
  torch::Tensor bnd_log_d_diff,
  const int num_fix,
  const int M,
  const float sigma_point,
  const float lambda_prior,
  const float depth_min,
  const float C_thresh,
  const float Q_thresh,
  const float huber_radius,
  const float lambda_smooth,
  const float lambda_lm,
  const float s_step_clip,
  const int num_iters);

std::vector<torch::Tensor> gauss_newton_seg_depths(
  torch::Tensor poses, torch::Tensor depth_init, torch::Tensor rays,
  torch::Tensor Cs,
  torch::Tensor labels, torch::Tensor seg_offsets, torch::Tensor seg_kf,
  torch::Tensor ii, torch::Tensor jj,
  torch::Tensor idx_ii2jj, torch::Tensor valid_match,
  torch::Tensor Q,
  torch::Tensor bnd_seg_a, torch::Tensor bnd_seg_b,
  torch::Tensor bnd_log_d_diff,
  const int num_fix,
  const float sigma_point,
  const float lambda_prior,
  const float depth_min,
  const float C_thresh,
  const float Q_thresh,
  const float huber_radius,
  const float lambda_smooth,
  const float lambda_lm,
  const float s_step_clip,
  const int num_iters);

std::vector<torch::Tensor> iter_proj(
  torch::Tensor rays_img_with_grad, 
  torch::Tensor pts_3d_norm, 
  torch::Tensor p_init,
  const int max_iter,
  const float lambda_init,
  const float cost_thresh);

std::vector<torch::Tensor> iter_proj_cuda(
  torch::Tensor rays_img_with_grad,
  torch::Tensor pts_3d_norm,
  torch::Tensor p_init,
  const int max_iter,
  const float lambda_init,
  const float cost_thresh);

std::vector<torch::Tensor> refine_matches_cuda(
  torch::Tensor D11, 
  torch::Tensor D21,
  torch::Tensor p1,
  const int radius,
  const int dilation_max);

std::vector<torch::Tensor> refine_matches(
  torch::Tensor D11, 
  torch::Tensor D21,
  torch::Tensor p1, 
  const int radius,
  const int dilation_max);