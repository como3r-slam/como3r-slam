#include "gn.h"

std::vector<torch::Tensor> gauss_newton_points(
  torch::Tensor Twc, torch::Tensor Xs, torch::Tensor Cs,
  torch::Tensor ii, torch::Tensor jj, 
  torch::Tensor idx_ii2jj, torch::Tensor valid_match,
  torch::Tensor Q,
  const float sigma_point,
  const float C_thresh,
  const float Q_thresh,
  const int max_iter,
  const float delta_thresh) {

  CHECK_CONTIGUOUS(Twc);
  CHECK_CONTIGUOUS(Xs);
  CHECK_CONTIGUOUS(Cs);
  CHECK_CONTIGUOUS(ii);
  CHECK_CONTIGUOUS(jj);
  CHECK_CONTIGUOUS(idx_ii2jj);
  CHECK_CONTIGUOUS(valid_match);
  CHECK_CONTIGUOUS(Q);

  // const at::cuda::OptionalCUDAGuard device_guard(device_of(x1));
  return gauss_newton_points_cuda(Twc, Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q,
      sigma_point, C_thresh, Q_thresh, max_iter, delta_thresh);
}

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
  const float delta_thresh) {

  CHECK_CONTIGUOUS(Twc);
  CHECK_CONTIGUOUS(Xs);
  CHECK_CONTIGUOUS(Cs);
  CHECK_CONTIGUOUS(ii);
  CHECK_CONTIGUOUS(jj);
  CHECK_CONTIGUOUS(idx_ii2jj);
  CHECK_CONTIGUOUS(valid_match);
  CHECK_CONTIGUOUS(Q);

  // const at::cuda::OptionalCUDAGuard device_guard(device_of(x1));
  return gauss_newton_rays_cuda(Twc, Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q,
      sigma_ray, sigma_dist, C_thresh, Q_thresh, max_iter, delta_thresh);
}

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
  const float delta_thresh) {

  CHECK_CONTIGUOUS(Twc);
  CHECK_CONTIGUOUS(Xs);
  CHECK_CONTIGUOUS(Cs);
  CHECK_CONTIGUOUS(K);
  CHECK_CONTIGUOUS(ii);
  CHECK_CONTIGUOUS(jj);
  CHECK_CONTIGUOUS(idx_ii2jj);
  CHECK_CONTIGUOUS(valid_match);
  CHECK_CONTIGUOUS(Q);

  // const at::cuda::OptionalCUDAGuard device_guard(device_of(x1));
  return gauss_newton_calib_cuda(Twc, Xs, Cs, K, ii, jj, idx_ii2jj, valid_match, Q,
      height, width, pixel_border, z_eps, sigma_pixel, sigma_depth, C_thresh, Q_thresh, max_iter, delta_thresh);
}

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
  const int num_iters) {

  CHECK_CONTIGUOUS(poses);
  CHECK_CONTIGUOUS(depth_init);
  CHECK_CONTIGUOUS(rays);
  CHECK_CONTIGUOUS(Cs);
  CHECK_CONTIGUOUS(labels);
  CHECK_CONTIGUOUS(seg_offsets);
  CHECK_CONTIGUOUS(seg_kf);
  CHECK_CONTIGUOUS(ii);
  CHECK_CONTIGUOUS(jj);
  CHECK_CONTIGUOUS(idx_ii2jj);
  CHECK_CONTIGUOUS(valid_match);
  CHECK_CONTIGUOUS(Q);
  CHECK_CONTIGUOUS(bnd_seg_a);
  CHECK_CONTIGUOUS(bnd_seg_b);
  CHECK_CONTIGUOUS(bnd_log_d_diff);

  TORCH_CHECK(depth_init.dim() == 2,
      "depth_init must be (N, P), got ", depth_init.sizes());
  TORCH_CHECK(labels.dim() == 2 && labels.sizes() == depth_init.sizes(),
      "labels must match depth_init shape (N, P), got labels=",
      labels.sizes(), " depth_init=", depth_init.sizes());
  TORCH_CHECK(labels.scalar_type() == torch::kInt32,
      "labels must be int32");
  TORCH_CHECK(seg_offsets.dim() == 1 &&
              seg_offsets.size(0) == depth_init.size(0) + 1,
      "seg_offsets must be (N+1,), got ", seg_offsets.sizes(),
      " for N=", depth_init.size(0));
  TORCH_CHECK(seg_offsets.scalar_type() == torch::kInt32 &&
              seg_kf.scalar_type() == torch::kInt32 &&
              bnd_seg_a.scalar_type() == torch::kInt32 &&
              bnd_seg_b.scalar_type() == torch::kInt32,
      "seg_offsets / seg_kf / bnd_seg_* must be int32");
  TORCH_CHECK(bnd_seg_a.sizes() == bnd_seg_b.sizes() &&
              bnd_seg_a.sizes() == bnd_log_d_diff.sizes(),
      "boundary tensors must all share length B");

  // Resolve M (total segment count) on the host so the CUDA orchestrator
  // can size its scratch buffers without an in-kernel sync. seg_offsets
  // is CSR-style so M = seg_offsets[N].
  const int N = depth_init.size(0);
  const int M = seg_offsets.cpu().data_ptr<int32_t>()[N];
  TORCH_CHECK(seg_kf.size(0) == M,
      "seg_kf length (", seg_kf.size(0), ") must equal M (", M, ")");

  return gauss_newton_seg_depths_cuda(
      poses, depth_init, rays, Cs,
      labels, seg_offsets, seg_kf,
      ii, jj, idx_ii2jj, valid_match, Q,
      bnd_seg_a, bnd_seg_b, bnd_log_d_diff,
      num_fix, M,
      sigma_point, lambda_prior, depth_min,
      C_thresh, Q_thresh, huber_radius,
      lambda_smooth, lambda_lm, s_step_clip,
      num_iters);
}

std::vector<torch::Tensor> iter_proj(
  torch::Tensor rays_img_with_grad, 
  torch::Tensor pts_3d_norm, 
  torch::Tensor p_init,
  const int max_iter,
  const float lambda_init,
  const float cost_thresh) {

  CHECK_CONTIGUOUS(rays_img_with_grad);
  CHECK_CONTIGUOUS(pts_3d_norm);
  CHECK_CONTIGUOUS(p_init);

  // const at::cuda::OptionalCUDAGuard device_guard(device_of(x1));
  return iter_proj_cuda(rays_img_with_grad, pts_3d_norm, p_init, 
      max_iter, lambda_init, cost_thresh);
}

std::vector<torch::Tensor> refine_matches(
  torch::Tensor D11, 
  torch::Tensor D21,
  torch::Tensor p1, 
  const int window_size,
  const int dilation_max) {

  CHECK_CONTIGUOUS(D11);
  CHECK_CONTIGUOUS(D21);
  CHECK_CONTIGUOUS(p1);

  // const at::cuda::OptionalCUDAGuard device_guard(device_of(x1));
  return refine_matches_cuda(D11, D21, p1, window_size, dilation_max);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gauss_newton_points", &gauss_newton_points, "gauss_newton point adjustment");
  m.def("gauss_newton_rays", &gauss_newton_rays, "gauss_newton ray adjustment");
  m.def("gauss_newton_calib", &gauss_newton_calib, "gauss_newton calib adjustment");
  m.def("gauss_newton_seg_depths", &gauss_newton_seg_depths,
        "super-primitive depth refinement: per-segment log-scale "
        "parameterization along frozen MASt3R rays (poses frozen), with "
        "boundary smoothness across adjacent segments");

  m.def("iter_proj", &iter_proj, "iterative projection with generic camera");
  m.def("refine_matches", &refine_matches, "refine match in local neighborhood");
}