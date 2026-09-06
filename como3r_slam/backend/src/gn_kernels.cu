// Part of this source code is derived from DROID-SLAM (https://github.com/princeton-vl/DROID-SLAM)
// Copyright (c) 2021, Princeton Vision & Learning Lab, licensed under the BSD 3-Clause License
//
// Any modifications made are licensed under the CC BY-NC-SA 4.0 License.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>

#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

#include <c10/cuda/CUDAException.h>

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;
typedef std::vector<std::vector<long>> graph_t;
typedef std::vector<torch::Tensor> tensor_list_t;


#define THREADS 256
#define NUM_BLOCKS(batch_size) ((batch_size + THREADS - 1) / THREADS)

#define GPU_1D_KERNEL_LOOP(k, n) \
  for (size_t k = threadIdx.x; k<n; k += blockDim.x)

#define EPS 1e-6

__device__ void warpReduce(volatile float *sdata, unsigned int tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid +  8];
  sdata[tid] += sdata[tid +  4];
  sdata[tid] += sdata[tid +  2];
  sdata[tid] += sdata[tid +  1];
}

__device__ void blockReduce(volatile float *sdata) {
  unsigned int tid = threadIdx.x;
  __syncthreads();

  // if (threadIdx.x < 256) {sdata[tid] += sdata[tid + 256]; } __syncthreads();
  if (threadIdx.x < 128) {sdata[tid] += sdata[tid + 128]; } __syncthreads();
  if (threadIdx.x <  64) {sdata[tid] += sdata[tid +  64]; } __syncthreads();

  if (tid < 32) warpReduce(sdata, tid);
  __syncthreads();
}

class SparseBlock {
  public:

    Eigen::SparseMatrix<double> A;
    Eigen::VectorX<double> b;

    SparseBlock(int N, int M) : N(N), M(M) {
      A = Eigen::SparseMatrix<double>(N*M, N*M);
      b = Eigen::VectorXd::Zero(N*M);
    }

    SparseBlock(Eigen::SparseMatrix<double> const& A, Eigen::VectorX<double> const& b, 
        int N, int M) : A(A), b(b), N(N), M(M) {}

    void update_lhs(torch::Tensor As, torch::Tensor ii, torch::Tensor jj) {

      auto As_cpu = As.to(torch::kCPU).to(torch::kFloat64);
      auto ii_cpu = ii.to(torch::kCPU).to(torch::kInt64);
      auto jj_cpu = jj.to(torch::kCPU).to(torch::kInt64);

      auto As_acc = As_cpu.accessor<double,3>();
      auto ii_acc = ii_cpu.accessor<long,1>();
      auto jj_acc = jj_cpu.accessor<long,1>();

      std::vector<T> tripletList;
      for (int n=0; n<ii.size(0); n++) {
        const int i = ii_acc[n];
        const int j = jj_acc[n];

        if (i >= 0 && j >= 0) {
          for (int k=0; k<M; k++) {
            for (int l=0; l<M; l++) {
              double val = As_acc[n][k][l];
              tripletList.push_back(T(M*i + k, M*j + l, val));
            }
          }
        }
      }
      A.setFromTriplets(tripletList.begin(), tripletList.end());
    }

    void update_rhs(torch::Tensor bs, torch::Tensor ii) {
      auto bs_cpu = bs.to(torch::kCPU).to(torch::kFloat64);
      auto ii_cpu = ii.to(torch::kCPU).to(torch::kInt64);

      auto bs_acc = bs_cpu.accessor<double,2>();
      auto ii_acc = ii_cpu.accessor<long,1>();

      for (int n=0; n<ii.size(0); n++) {
        const int i = ii_acc[n];
        if (i >= 0) {
          for (int j=0; j<M; j++) {
            b(i*M + j) += bs_acc[n][j];
          }
        }
      }
    }

    SparseBlock operator-(const SparseBlock& S) {
      return SparseBlock(A - S.A, b - S.b, N, M);
    }

    std::tuple<torch::Tensor, torch::Tensor> get_dense() {
      Eigen::MatrixXd Ad = Eigen::MatrixXd(A);

      torch::Tensor H = torch::from_blob(Ad.data(), {N*M, N*M}, torch::TensorOptions()
        .dtype(torch::kFloat64)).to(torch::kCUDA).to(torch::kFloat32);

      torch::Tensor v = torch::from_blob(b.data(), {N*M, 1}, torch::TensorOptions()
        .dtype(torch::kFloat64)).to(torch::kCUDA).to(torch::kFloat32);

      return std::make_tuple(H, v);

    }

    torch::Tensor solve(const float lm=0.0, const float ep=1e-6) {

      torch::Tensor dx;

      // Multi-agent / loop-closure scenarios can produce ill-conditioned
      // Hessians (e.g. when residuals are near zero or some DOFs couple
      // weakly through long chains). A pure LLT with zero regularization
      // silently fails in that case and returns dx = 0, which used to make
      // the Gauss-Newton loop a no-op. We retry with escalating Tikhonov
      // regularization on the diagonal until LLT succeeds.
      const int n_attempts = 6;
      for (int attempt = 0; attempt < n_attempts; attempt++) {
        // Scale ep by 100^attempt: 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4.
        double scale = 1.0;
        for (int s = 0; s < attempt; s++) scale *= 100.0;
        double cur_ep = (double)ep * scale;

        Eigen::SparseMatrix<double> L(A);
        L.diagonal().array() += cur_ep + (double)lm * L.diagonal().array();

        Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(L);

        if (solver.info() == Eigen::Success) {
          Eigen::VectorXd x = solver.solve(b);
          dx = torch::from_blob(x.data(), {N, M}, torch::TensorOptions()
            .dtype(torch::kFloat64)).to(torch::kCUDA).to(torch::kFloat32);
          return dx;
        }
      }

      // Even with heavy regularization the system is unsolvable; fall back
      // to a zero update so the outer GN loop can terminate gracefully.
      dx = torch::zeros({N, M}, torch::TensorOptions()
        .device(torch::kCUDA).dtype(torch::kFloat32));
      return dx;
    }

  private:
    const int N;
    const int M;

};

torch::Tensor get_unique_kf_idx(torch::Tensor ii, torch::Tensor jj) {
  std::tuple<torch::Tensor, torch::Tensor> unique_kf_idx = torch::_unique(torch::cat({ii,jj}), /*sorted=*/ true);
  return std::get<0>(unique_kf_idx);
}

std::vector<torch::Tensor> create_inds(torch::Tensor unique_kf_idx, const int pin, torch::Tensor ii, torch::Tensor jj) {
  torch::Tensor ii_ind = torch::searchsorted(unique_kf_idx, ii) - pin;
  torch::Tensor jj_ind = torch::searchsorted(unique_kf_idx, jj) - pin;
  return {ii_ind, jj_ind};
}

__forceinline__ __device__ float huber(float r) {
  const float r_abs = fabs(r);
  return r_abs < 1.345 ? 1.0 : 1.345 / r_abs;
}

// Returns qi * qj
__device__ void 
quat_comp(const float *qi, const float *qj, float *out) {
  out[0] = qi[3] * qj[0] + qi[0] * qj[3] + qi[1] * qj[2] - qi[2] * qj[1];
  out[1] = qi[3] * qj[1] - qi[0] * qj[2] + qi[1] * qj[3] + qi[2] * qj[0];
  out[2] = qi[3] * qj[2] + qi[0] * qj[1] - qi[1] * qj[0] + qi[2] * qj[3];
  out[3] = qi[3] * qj[3] - qi[0] * qj[0] - qi[1] * qj[1] - qi[2] * qj[2];
}

// Inverts quat
__device__ void 
quat_inv(const float *q, float *out) {
  out[0] = -q[0];
  out[1] = -q[1];
  out[2] = -q[2];
  out[3] =  q[3];
}

__device__ void
actSO3(const float *q, const float *X, float *Y) {
  float uv[3];
  uv[0] = 2.0 * (q[1]*X[2] - q[2]*X[1]);
  uv[1] = 2.0 * (q[2]*X[0] - q[0]*X[2]);
  uv[2] = 2.0 * (q[0]*X[1] - q[1]*X[0]);

  Y[0] = X[0] + q[3]*uv[0] + (q[1]*uv[2] - q[2]*uv[1]);
  Y[1] = X[1] + q[3]*uv[1] + (q[2]*uv[0] - q[0]*uv[2]);
  Y[2] = X[2] + q[3]*uv[2] + (q[0]*uv[1] - q[1]*uv[0]);
}

__device__  void
actSim3(const float *t, const float *q, const float *s, const float *X, float *Y) {
  // Rotation
  actSO3(q, X, Y);
  // Scale
  Y[0] *= s[0];
  Y[1] *= s[0];
  Y[2] *= s[0];
  // Translation
  Y[0] += t[0];
  Y[1] += t[1];
  Y[2] += t[2];
}

// Inverts quat
__device__ void 
scale_vec3_inplace(float *t, float s) {
  t[0] *= s;
  t[1] *= s;
  t[2] *= s;
}

__device__ void
crossInplace(const float* a, float *b) {
  float x[3] = {
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0], 
  };

  b[0] = x[0];
  b[1] = x[1];
  b[2] = x[2];
}

__forceinline__ __device__ float
dot3(const float *t, const float *s) {
  return t[0]*s[0] + t[1]*s[1] + t[2]*s[2];
}

__forceinline__ __device__ float
squared_norm3(const float *v) {
  return v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
}

__device__ void 
relSim3(const float *ti, const float *qi, const float* si,
        const float *tj, const float *qj, const float* sj,
        float *tij, float *qij, float *sij) {
  
  // 1. Setup scale
  float si_inv = 1.0/si[0];
  sij[0] = si_inv * sj[0];

  // 2. Relative rotation
  float qi_inv[4];
  quat_inv(qi, qi_inv);
  quat_comp(qi_inv, qj, qij);

  // 3. Translation
  tij[0] = tj[0] - ti[0];
  tij[1] = tj[1] - ti[1];
  tij[2] = tj[2] - ti[2];
  actSO3(qi_inv, tij, tij);
  scale_vec3_inplace(tij, si_inv);
}

// Order of X,Y is tau, omega, s 
// NOTE: This is applying adj inv on the right to a row vector on the left,
// The equivalent is transposing the adjoint and multiplying a column vector
__device__ void
apply_Sim3_adj_inv(const float *t, const float *q, const float *s, const float *X, float *Y) {
  // float qinv[4] = {-q[0], -q[1], -q[2], q[3]};
  const float s_inv = 1.0/s[0];

  // First component = s_inv R a
  float Ra[3];
  actSO3(q, &X[0], Ra);
  Y[0] = s_inv * Ra[0];
  Y[1] = s_inv * Ra[1];
  Y[2] = s_inv * Ra[2];
  
  // Second component = s_inv [t]x Ra + Rb
  actSO3(q, &X[3], &Y[3]); // Init to Rb
  Y[3] += s_inv*(t[1]*Ra[2] - t[2]*Ra[1]);
  Y[4] += s_inv*(t[2]*Ra[0] - t[0]*Ra[2]);
  Y[5] += s_inv*(t[0]*Ra[1] - t[1]*Ra[0]);

  // Third component = s_inv t^T R a + c
  Y[6] = X[6] + ( s_inv * dot3(t, Ra) );
}

__device__ void
expSO3(const float *phi, float* q) {
  // SO3 exponential map
  float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];

  float imag, real;

  if (theta_sq < EPS) {
    float theta_p4 = theta_sq * theta_sq;
    imag = 0.5 - (1.0/48.0)*theta_sq + (1.0/3840.0)*theta_p4;
    real = 1.0 - (1.0/ 8.0)*theta_sq + (1.0/ 384.0)*theta_p4;
  } else {
    float theta = sqrtf(theta_sq);
    imag = sinf(0.5 * theta) / theta;
    real = cosf(0.5 * theta);
  }

  q[0] = imag * phi[0];
  q[1] = imag * phi[1];
  q[2] = imag * phi[2];
  q[3] = real;

}

__device__ void
expSim3(const float *xi, float* t, float* q, float* s) {
  float tau[3] = {xi[0], xi[1], xi[2]};
  float phi[3] = {xi[3], xi[4], xi[5]};
  float sigma = xi[6];

  // New for sim3
  float scale = expf(sigma);

  // 1. Rotation
  expSO3(phi, q);
  // 2. Scale
  s[0] = scale;

  // 3. Translation

  // TODO: Reuse this from expSO3?
  float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
  float theta = sqrtf(theta_sq);

  // Coefficients for W
  https://github.com/princeton-vl/lietorch/blob/0fa9ce8ffca86d985eca9e189a99690d6f3d4df6/lietorch/include/rxso3.h#L190
  
  // TODO: Does this really match equations? Where is scale-1
  float A, B, C;
  const float one = 1.0;
  const float half = 0.5;
  if (fabs(sigma) < EPS) {
    C = one;
    if (fabs(theta) < EPS) {
      A = half;
      B = 1.0/6.0;
    } else {
      A = (one - cosf(theta)) / theta_sq;
      B = (theta - sinf(theta)) / (theta_sq * theta);
    }
  } else {
    C = (scale - one) / sigma;
    if (fabs(theta) < EPS) {
      float sigma_sq = sigma * sigma;
      A = ((sigma - one) * scale + one) / sigma_sq;
      B = (scale * half * sigma_sq + scale - one - sigma * scale) /
          (sigma_sq * sigma);
    } else {
      float a = scale * sinf(theta);
      float b = scale * cosf(theta);
      float c = theta_sq + sigma * sigma;
      A = (a * sigma + (one - b) * theta) / (theta * c);
      B = (C - ((b - one) * sigma + a * theta) / (c)) / (theta_sq); // Why is it C - ????? not +?
    }
  }

  // W = C * I + A * Phi + B * Phi2;
  // t = W tau
  t[0] = C * tau[0]; 
  t[1] = C * tau[1]; 
  t[2] = C * tau[2];

  crossInplace(phi, tau);
  t[0] += A * tau[0];
  t[1] += A * tau[1];
  t[2] += A * tau[2];

  crossInplace(phi, tau);
  t[0] += B * tau[0];
  t[1] += B * tau[1];
  t[2] += B * tau[2];
}

__device__ void
retrSim3(const float *xi, const float* t, const float* q, const float* s, float* t1, float* q1, float* s1) {
  
  // retraction on Sim3 manifold
  float dt[3] = {0, 0, 0};
  float dq[4] = {0, 0, 0, 1};
  float ds[1] = {0};
  
  expSim3(xi, dt, dq, ds);

  // Compose transformation from left
  // R
  quat_comp(dq, q, q1);
  // t = ds dR R + ds
  actSO3(dq, t, t1);
  scale_vec3_inplace(t1, ds[0]);
  t1[0] += dt[0];
  t1[1] += dt[1];
  t1[2] += dt[2];
  // s
  s1[0] = ds[0] * s[0];
}

__global__ void pose_retr_kernel(
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> dx,
    const int num_fix) 
{
  const int num_poses = poses.size(0);

  for (int k=num_fix+threadIdx.x; k<num_poses; k+=blockDim.x) {
    float xi[7], q[4], q1[4], t[3], t1[3], s[1], s1[1];

    t[0] = poses[k][0];
    t[1] = poses[k][1];
    t[2] = poses[k][2];

    q[0] = poses[k][3];
    q[1] = poses[k][4];
    q[2] = poses[k][5];
    q[3] = poses[k][6];

    s[0] = poses[k][7];
    
    for (int n=0; n<7; n++) {
      xi[n] = dx[k-num_fix][n];
    }

    retrSim3(xi, t, q, s, t1, q1, s1);

    poses[k][0] = t1[0];
    poses[k][1] = t1[1];
    poses[k][2] = t1[2];

    poses[k][3] = q1[0];
    poses[k][4] = q1[1];
    poses[k][5] = q1[2];
    poses[k][6] = q1[3];

    poses[k][7] = s1[0];
  }
}

__global__ void point_align_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Twc,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Xs,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Cs,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    const torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> idx_ii2_jj,
    const torch::PackedTensorAccessor32<bool,3,torch::RestrictPtrTraits> valid_match,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Q,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> Hs,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gs,
    const float sigma_point,
    const float C_thresh,
    const float Q_thresh)
{
 
  // Twc and Xs first dim is number of poses
  // ii, jj, Cii, Cjj, Q first dim is number of edges
 
  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;
 
  const int num_points = Xs.size(1);
 
  int ix = static_cast<int>(ii[block_id]);
  int jx = static_cast<int>(jj[block_id]);
 
  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];
  __shared__ float si[1], sj[1], sij[1];
 
  __syncthreads();
 
  // load poses from global memory
  if (thread_id < 3) {
    ti[thread_id] = Twc[ix][thread_id];
    tj[thread_id] = Twc[jx][thread_id];
  }
 
  if (thread_id < 4) {
    qi[thread_id] = Twc[ix][thread_id+3];
    qj[thread_id] = Twc[jx][thread_id+3];
  }
 
  if (thread_id < 1) {
    si[thread_id] = Twc[ix][thread_id+7];
    sj[thread_id] = Twc[jx][thread_id+7];
  }
 
  __syncthreads();
 
  // Calculate relative poses
  if (thread_id == 0) {
    relSim3(ti, qi, si, tj, qj, sj, tij, qij, sij);
  }
 
  __syncthreads();
 
  // //points
  float Xi[3];
  float Xj[3];
  float Xj_Ci[3];
 
  // residuals
  float err[3];
  float w[3];
 
  // // jacobians
  float Jx[14];
  // float Jz;
 
  float* Ji = &Jx[0];
  float* Jj = &Jx[7];
 
  // hessians
  const int h_dim = 14*(14+1)/2;
  float hij[h_dim];
 
  float vi[7], vj[7];
 
  int l; // We reuse this variable later for Hessian fill-in
  for (l=0; l<h_dim; l++) {
    hij[l] = 0;
  }
 
  for (int n=0; n<7; n++) {
    vi[n] = 0;
    vj[n] = 0;
  }
 
    // Parameters
  const float sigma_point_inv = 1.0/sigma_point;
 
  __syncthreads();
 
  GPU_1D_KERNEL_LOOP(k, num_points) {
 
    // Get points
    const bool valid_match_ind = valid_match[block_id][k][0]; 
    const int64_t ind_Xi = valid_match_ind ? idx_ii2_jj[block_id][k] : 0;

    Xi[0] = Xs[ix][ind_Xi][0];
    Xi[1] = Xs[ix][ind_Xi][1];
    Xi[2] = Xs[ix][ind_Xi][2];
 
    Xj[0] = Xs[jx][k][0];
    Xj[1] = Xs[jx][k][1];
    Xj[2] = Xs[jx][k][2];
 
    // Transform point
    actSim3(tij, qij, sij, Xj, Xj_Ci);
 
    // Error (difference in camera rays)
    err[0] = Xj_Ci[0] - Xi[0];
    err[1] = Xj_Ci[1] - Xi[1];
    err[2] = Xj_Ci[2] - Xi[2];
 
    // Weights (Huber)
    const float q = Q[block_id][k][0];
    const float ci = Cs[ix][ind_Xi][0];
    const float cj = Cs[jx][k][0];
    const bool valid = 
      valid_match_ind
      & (q > Q_thresh)
      & (ci > C_thresh)
      & (cj > C_thresh);

    // Weight using confidences
    const float conf_weight = q;
    // const float conf_weight = q * ci * cj;
    
    const float sqrt_w_point = valid ? sigma_point_inv * sqrtf(conf_weight) : 0;
 
    // Robust weight
    w[0] = huber(sqrt_w_point * err[0]);
    w[1] = huber(sqrt_w_point * err[1]);
    w[2] = huber(sqrt_w_point * err[2]);
    
    // Add back in sigma
    const float w_const_point = sqrt_w_point * sqrt_w_point;
    w[0] *= w_const_point;
    w[1] *= w_const_point;
    w[2] *= w_const_point;
 
    // Jacobians
    
    // x coordinate
    Ji[0] = 1.0;
    Ji[1] = 0.0;
    Ji[2] = 0.0;
    Ji[3] = 0.0;
    Ji[4] = Xj_Ci[2]; // z
    Ji[5] = -Xj_Ci[1]; // -y
    Ji[6] = Xj_Ci[0]; // x

    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[0] * Jx[n] * Jx[m];
        l++;
      }
    }
 
    for (int n=0; n<7; n++) {
      vi[n] += w[0] * err[0] * Ji[n];
      vj[n] += w[0] * err[0] * Jj[n];
    }
 
    // y coordinate
    Ji[0] = 0.0;
    Ji[1] = 1.0;
    Ji[2] = 0.0;
    Ji[3] = -Xj_Ci[2]; // -z
    Ji[4] = 0; 
    Ji[5] = Xj_Ci[0]; // x
    Ji[6] = Xj_Ci[1]; // y
 
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[1] * Jx[n] * Jx[m];
        l++;
      }
    }
 
    for (int n=0; n<7; n++) {
      vi[n] += w[1] * err[1] * Ji[n];
      vj[n] += w[1] * err[1] * Jj[n];
    }
 
    // z coordinate
    Ji[0] = 0.0;
    Ji[1] = 0.0;
    Ji[2] = 1.0;
    Ji[3] = Xj_Ci[1]; // y
    Ji[4] = -Xj_Ci[0]; // -x 
    Ji[5] = 0;
    Ji[6] = Xj_Ci[2]; // z
 
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[2] * Jx[n] * Jx[m];
        l++;
      }
    }
 
    for (int n=0; n<7; n++) {
      vi[n] += w[2] * err[2] * Ji[n];
      vj[n] += w[2] * err[2] * Jj[n];
    }
 
 
  }
 
  __syncthreads();
 
  __shared__ float sdata[THREADS];
  for (int n=0; n<7; n++) {
    sdata[threadIdx.x] = vi[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      gs[0][block_id][n] = sdata[0];
    }
 
    __syncthreads();
 
    sdata[threadIdx.x] = vj[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      gs[1][block_id][n] = sdata[0];
    }
 
  }
 
  l=0;
  for (int n=0; n<14; n++) {
    for (int m=0; m<=n; m++) {
      sdata[threadIdx.x] = hij[l];
      blockReduce(sdata);
 
      if (threadIdx.x == 0) {
        if (n<7 && m<7) {
          Hs[0][block_id][n][m] = sdata[0];
          Hs[0][block_id][m][n] = sdata[0];
        }
        else if (n >=7 && m<7) {
          Hs[1][block_id][m][n-7] = sdata[0];
          Hs[2][block_id][n-7][m] = sdata[0];
        }
        else {
          Hs[3][block_id][n-7][m-7] = sdata[0];
          Hs[3][block_id][m-7][n-7] = sdata[0];
        }
      }
 
      l++;
    }
  }
}

std::vector<torch::Tensor> gauss_newton_points_cuda(
  torch::Tensor Twc, torch::Tensor Xs, torch::Tensor Cs,
  torch::Tensor ii, torch::Tensor jj, 
  torch::Tensor idx_ii2jj, torch::Tensor valid_match,
  torch::Tensor Q,
  const float sigma_point,
  const float C_thresh,
  const float Q_thresh,
  const int max_iter,
  const float delta_thresh)
{
  auto opts = Twc.options();
  const int num_edges = ii.size(0);
  const int num_poses = Xs.size(0);
  const int n = Xs.size(1);

  const int num_fix = 1;

  // Setup indexing
  torch::Tensor unique_kf_idx = get_unique_kf_idx(ii, jj);
  // For edge construction
  std::vector<torch::Tensor> inds = create_inds(unique_kf_idx, 0, ii, jj);
  torch::Tensor ii_edge = inds[0];
  torch::Tensor jj_edge = inds[1];
  // For linear system indexing (pin=2 because fixing first two poses)
  std::vector<torch::Tensor> inds_opt = create_inds(unique_kf_idx, num_fix, ii, jj);
  torch::Tensor ii_opt = inds_opt[0];
  torch::Tensor jj_opt = inds_opt[1];

  const int pose_dim = 7; // sim3

  // initialize buffers
  torch::Tensor Hs = torch::zeros({4, num_edges, pose_dim, pose_dim}, opts);
  torch::Tensor gs = torch::zeros({2, num_edges, pose_dim}, opts);

  // For debugging outputs
  torch::Tensor dx;

  torch::Tensor delta_norm;

  for (int itr=0; itr<max_iter; itr++) {

    point_align_kernel<<<num_edges, THREADS>>>(
      Twc.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      Xs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Cs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      ii_edge.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      jj_edge.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      idx_ii2jj.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
      valid_match.packed_accessor32<bool,3,torch::RestrictPtrTraits>(),
      Q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Hs.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      gs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      sigma_point, C_thresh, Q_thresh
    );


    // pose x pose block
    SparseBlock A(num_poses - num_fix, pose_dim);

    A.update_lhs(Hs.reshape({-1, pose_dim, pose_dim}), 
        torch::cat({ii_opt, ii_opt, jj_opt, jj_opt}), 
        torch::cat({ii_opt, jj_opt, ii_opt, jj_opt}));

    A.update_rhs(gs.reshape({-1, pose_dim}), 
        torch::cat({ii_opt, jj_opt}));

    // NOTE: Accounting for negative here!
    dx = -A.solve();
    
    pose_retr_kernel<<<1, THREADS>>>(
      Twc.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      num_fix);

    // Termination criteria
    // Need to specify this second argument otherwise ambiguous function call...
    delta_norm = torch::linalg::linalg_norm(dx, std::optional<c10::Scalar>(), {}, false, {});
    if (delta_norm.item<float>() < delta_thresh) {
      break;
    }
        

  }

  return {dx}; // For debugging
}

__global__ void ray_align_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Twc,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Xs,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Cs,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    const torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> idx_ii2_jj,
    const torch::PackedTensorAccessor32<bool,3,torch::RestrictPtrTraits> valid_match,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Q,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> Hs,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gs,
    const float sigma_ray,
    const float sigma_dist,
    const float C_thresh,
    const float Q_thresh)
{
 
  // Twc and Xs first dim is number of poses
  // ii, jj, Cii, Cjj, Q first dim is number of edges
 
  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;
 
  const int num_points = Xs.size(1);
 
  int ix = static_cast<int>(ii[block_id]);
  int jx = static_cast<int>(jj[block_id]);
 
  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];
  __shared__ float si[1], sj[1], sij[1];
 
  __syncthreads();
 
  // load poses from global memory
  if (thread_id < 3) {
    ti[thread_id] = Twc[ix][thread_id];
    tj[thread_id] = Twc[jx][thread_id];
  }
 
  if (thread_id < 4) {
    qi[thread_id] = Twc[ix][thread_id+3];
    qj[thread_id] = Twc[jx][thread_id+3];
  }
 
  if (thread_id < 1) {
    si[thread_id] = Twc[ix][thread_id+7];
    sj[thread_id] = Twc[jx][thread_id+7];
  }
 
  __syncthreads();
 
  // Calculate relative poses
  if (thread_id == 0) {
    relSim3(ti, qi, si, tj, qj, sj, tij, qij, sij);
  }
 
  __syncthreads();
 
  // //points
  float Xi[3];
  float Xj[3];
  float Xj_Ci[3];
 
  // residuals
  float err[4];
  float w[4];
 
  // // jacobians
  float Jx[14];
  // float Jz;
 
  float* Ji = &Jx[0];
  float* Jj = &Jx[7];
 
  // hessians
  const int h_dim = 14*(14+1)/2;
  float hij[h_dim];
 
  float vi[7], vj[7];
 
  int l; // We reuse this variable later for Hessian fill-in
  for (l=0; l<h_dim; l++) {
    hij[l] = 0;
  }
 
  for (int n=0; n<7; n++) {
    vi[n] = 0;
    vj[n] = 0;
  }
 
    // Parameters
  const float sigma_ray_inv = 1.0/sigma_ray;
  const float sigma_dist_inv = 1.0/sigma_dist;
 
  __syncthreads();
 
  GPU_1D_KERNEL_LOOP(k, num_points) {
 
    // Get points
    const bool valid_match_ind = valid_match[block_id][k][0]; 
    const int64_t ind_Xi = valid_match_ind ? idx_ii2_jj[block_id][k] : 0;

    Xi[0] = Xs[ix][ind_Xi][0];
    Xi[1] = Xs[ix][ind_Xi][1];
    Xi[2] = Xs[ix][ind_Xi][2];
 
    Xj[0] = Xs[jx][k][0];
    Xj[1] = Xs[jx][k][1];
    Xj[2] = Xs[jx][k][2];
 
    // Normalize measurement point
    const float norm2_i = squared_norm3(Xi);
    const float norm1_i = sqrtf(norm2_i);
    const float norm1_i_inv = 1.0/norm1_i;    
    
    float ri[3];
    for (int i=0; i<3; i++) ri[i] = norm1_i_inv * Xi[i];
 
    // Transform point
    actSim3(tij, qij, sij, Xj, Xj_Ci);
 
    // Get predicted point norm
    const float norm2_j = squared_norm3(Xj_Ci);
    const float norm1_j = sqrtf(norm2_j);
    const float norm1_j_inv = 1.0/norm1_j;

    float rj_Ci[3];
    for (int i=0; i<3; i++) rj_Ci[i] = norm1_j_inv * Xj_Ci[i];
 
    // Error (difference in camera rays)
    err[0] = rj_Ci[0] - ri[0];
    err[1] = rj_Ci[1] - ri[1];
    err[2] = rj_Ci[2] - ri[2];
    err[3] = norm1_j - norm1_i; // Distance
 
    // Weights (Huber)
    const float q = Q[block_id][k][0];
    const float ci = Cs[ix][ind_Xi][0];
    const float cj = Cs[jx][k][0];
    const bool valid = 
      valid_match_ind
      & (q > Q_thresh)
      & (ci > C_thresh)
      & (cj > C_thresh);

    // Weight using confidences
    const float conf_weight = q;
    // const float conf_weight = q * ci * cj;
    
    const float sqrt_w_ray = valid ? sigma_ray_inv * sqrtf(conf_weight) : 0;
    const float sqrt_w_dist = valid ? sigma_dist_inv * sqrtf(conf_weight) : 0;
 
    // Robust weight
    w[0] = huber(sqrt_w_ray * err[0]);
    w[1] = huber(sqrt_w_ray * err[1]);
    w[2] = huber(sqrt_w_ray * err[2]);
    w[3] = huber(sqrt_w_dist * err[3]);
    
    // Add back in sigma
    const float w_const_ray = sqrt_w_ray * sqrt_w_ray;
    const float w_const_dist = sqrt_w_dist * sqrt_w_dist;
    w[0] *= w_const_ray;
    w[1] *= w_const_ray;
    w[2] *= w_const_ray;
    w[3] *= w_const_dist;
 
    // Jacobians
    
    const float norm3_j_inv = norm1_j_inv / norm2_j;
    const float drx_dPx = norm1_j_inv - Xj_Ci[0]*Xj_Ci[0]*norm3_j_inv;
    const float dry_dPy = norm1_j_inv - Xj_Ci[1]*Xj_Ci[1]*norm3_j_inv;
    const float drz_dPz = norm1_j_inv - Xj_Ci[2]*Xj_Ci[2]*norm3_j_inv;
    const float drx_dPy = - Xj_Ci[0]*Xj_Ci[1]*norm3_j_inv;
    const float drx_dPz = - Xj_Ci[0]*Xj_Ci[2]*norm3_j_inv;
    const float dry_dPz = - Xj_Ci[1]*Xj_Ci[2]*norm3_j_inv;
 
    // rx coordinate
    Ji[0] = drx_dPx;
    Ji[1] = drx_dPy;
    Ji[2] = drx_dPz;
    Ji[3] = 0.0;
    Ji[4] = rj_Ci[2]; // z
    Ji[5] = -rj_Ci[1]; // -y
    Ji[6] = 0.0; // x

    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[0] * Jx[n] * Jx[m];
        l++;
      }
    }
 
    for (int n=0; n<7; n++) {
      vi[n] += w[0] * err[0] * Ji[n];
      vj[n] += w[0] * err[0] * Jj[n];
    }
 
    // ry coordinate
    Ji[0] = drx_dPy; // same as drx_dPy
    Ji[1] = dry_dPy;
    Ji[2] = dry_dPz;
    Ji[3] = -rj_Ci[2]; // -z
    Ji[4] = 0.0;
    Ji[5] = rj_Ci[0]; // x
    Ji[6] = 0.0; // y
 
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[1] * Jx[n] * Jx[m];
        l++;
      }
    }
 
    for (int n=0; n<7; n++) {
      vi[n] += w[1] * err[1] * Ji[n];
      vj[n] += w[1] * err[1] * Jj[n];
    }
 
    // rz coordinate
    Ji[0] = drx_dPz; // same as drz_dPX
    Ji[1] = dry_dPz; // same as drz_dPy
    Ji[2] = drz_dPz;
    Ji[3] = rj_Ci[1]; // y
    Ji[4] = -rj_Ci[0]; // -x
    Ji[5] = 0.0;
    Ji[6] = 0.0; // z
 
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[2] * Jx[n] * Jx[m];
        l++;
      }
    }
 
    for (int n=0; n<7; n++) {
      vi[n] += w[2] * err[2] * Ji[n];
      vj[n] += w[2] * err[2] * Jj[n];
    }


    // dist coordinate
    Ji[0] = rj_Ci[0];
    Ji[1] = rj_Ci[1]; 
    Ji[2] = rj_Ci[2];
    Ji[3] = 0.0; 
    Ji[4] = 0.0; 
    Ji[5] = 0.0;
    Ji[6] = norm1_j;
 
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[3] * Jx[n] * Jx[m];
        l++;
      }
    }
 
    for (int n=0; n<7; n++) {
      vi[n] += w[3] * err[3] * Ji[n];
      vj[n] += w[3] * err[3] * Jj[n];
    }
 
 
  }
 
  __syncthreads();
 
  __shared__ float sdata[THREADS];
  for (int n=0; n<7; n++) {
    sdata[threadIdx.x] = vi[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      gs[0][block_id][n] = sdata[0];
    }
 
    __syncthreads();
 
    sdata[threadIdx.x] = vj[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      gs[1][block_id][n] = sdata[0];
    }
 
  }
 
  l=0;
  for (int n=0; n<14; n++) {
    for (int m=0; m<=n; m++) {
      sdata[threadIdx.x] = hij[l];
      blockReduce(sdata);
 
      if (threadIdx.x == 0) {
        if (n<7 && m<7) {
          Hs[0][block_id][n][m] = sdata[0];
          Hs[0][block_id][m][n] = sdata[0];
        }
        else if (n >=7 && m<7) {
          Hs[1][block_id][m][n-7] = sdata[0];
          Hs[2][block_id][n-7][m] = sdata[0];
        }
        else {
          Hs[3][block_id][n-7][m-7] = sdata[0];
          Hs[3][block_id][m-7][n-7] = sdata[0];
        }
      }
 
      l++;
    }
  }
}

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
  const float delta_thresh)
{
  auto opts = Twc.options();
  const int num_edges = ii.size(0);
  const int num_poses = Xs.size(0);
  const int n = Xs.size(1);

  const int num_fix = 1;

  // Setup indexing
  torch::Tensor unique_kf_idx = get_unique_kf_idx(ii, jj);
  // For edge construction
  std::vector<torch::Tensor> inds = create_inds(unique_kf_idx, 0, ii, jj);
  torch::Tensor ii_edge = inds[0];
  torch::Tensor jj_edge = inds[1];
  // For linear system indexing (pin=2 because fixing first two poses)
  std::vector<torch::Tensor> inds_opt = create_inds(unique_kf_idx, num_fix, ii, jj);
  torch::Tensor ii_opt = inds_opt[0];
  torch::Tensor jj_opt = inds_opt[1];

  const int pose_dim = 7; // sim3

  // initialize buffers
  torch::Tensor Hs = torch::zeros({4, num_edges, pose_dim, pose_dim}, opts);
  torch::Tensor gs = torch::zeros({2, num_edges, pose_dim}, opts);

  // For debugging outputs
  torch::Tensor dx;

  torch::Tensor delta_norm;

  for (int itr=0; itr<max_iter; itr++) {

    ray_align_kernel<<<num_edges, THREADS>>>(
      Twc.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      Xs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Cs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      ii_edge.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      jj_edge.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      idx_ii2jj.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
      valid_match.packed_accessor32<bool,3,torch::RestrictPtrTraits>(),
      Q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Hs.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      gs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      sigma_ray, sigma_dist, C_thresh, Q_thresh
    );


    // pose x pose block
    SparseBlock A(num_poses - num_fix, pose_dim);

    A.update_lhs(Hs.reshape({-1, pose_dim, pose_dim}), 
        torch::cat({ii_opt, ii_opt, jj_opt, jj_opt}), 
        torch::cat({ii_opt, jj_opt, ii_opt, jj_opt}));

    A.update_rhs(gs.reshape({-1, pose_dim}), 
        torch::cat({ii_opt, jj_opt}));

    // NOTE: Accounting for negative here!
    dx = -A.solve();

    //
    pose_retr_kernel<<<1, THREADS>>>(
      Twc.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      num_fix);

    // Termination criteria
    // Need to specify this second argument otherwise ambiguous function call...
    delta_norm = torch::linalg::linalg_norm(dx, std::optional<c10::Scalar>(), {}, false, {});
    if (delta_norm.item<float>() < delta_thresh) {
      break;
    }
        

  }

  return {dx}; // For debugging
}


__global__ void calib_proj_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Twc,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Xs,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Cs,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> K,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    const torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> idx_ii2_jj,
    const torch::PackedTensorAccessor32<bool,3,torch::RestrictPtrTraits> valid_match,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Q,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> Hs,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gs,
    const int height,
    const int width,
    const int pixel_border,
    const float z_eps,
    const float sigma_pixel,
    const float sigma_depth,
    const float C_thresh,
    const float Q_thresh)
{
 
  // Twc and Xs first dim is number of poses
  // ii, jj, Cii, Cjj, Q first dim is number of edges
 
  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;
 
  const int num_points = Xs.size(1);
 
  int ix = static_cast<int>(ii[block_id]);
  int jx = static_cast<int>(jj[block_id]);

  __shared__ float fx;
  __shared__ float fy;
  __shared__ float cx;
  __shared__ float cy;
 
  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];
  __shared__ float si[1], sj[1], sij[1];

  // load intrinsics from global memory
  if (thread_id == 0) {
    fx = K[0][0];
    fy = K[1][1];
    cx = K[0][2];
    cy = K[1][2];
  }
 
  __syncthreads();
 
  // load poses from global memory
  if (thread_id < 3) {
    ti[thread_id] = Twc[ix][thread_id];
    tj[thread_id] = Twc[jx][thread_id];
  }
 
  if (thread_id < 4) {
    qi[thread_id] = Twc[ix][thread_id+3];
    qj[thread_id] = Twc[jx][thread_id+3];
  }
 
  if (thread_id < 1) {
    si[thread_id] = Twc[ix][thread_id+7];
    sj[thread_id] = Twc[jx][thread_id+7];
  }
 
  __syncthreads();
 
  // Calculate relative poses
  if (thread_id == 0) {
    relSim3(ti, qi, si, tj, qj, sj, tij, qij, sij);
  }
 
  __syncthreads();
 
  // //points
  float Xi[3];
  float Xj[3];
  float Xj_Ci[3];
 
  // residuals
  float err[3];
  float w[3];
 
  // // jacobians
  float Jx[14];
  // float Jz;
 
  float* Ji = &Jx[0];
  float* Jj = &Jx[7];
 
  // hessians
  const int h_dim = 14*(14+1)/2;
  float hij[h_dim];
 
  float vi[7], vj[7];
 
  int l; // We reuse this variable later for Hessian fill-in
  for (l=0; l<h_dim; l++) {
    hij[l] = 0;
  }
 
  for (int n=0; n<7; n++) {
    vi[n] = 0;
    vj[n] = 0;
  }
 
  // Parameters
  const float sigma_pixel_inv = 1.0/sigma_pixel;
  const float sigma_depth_inv = 1.0/sigma_depth;
 
  __syncthreads();
 
  GPU_1D_KERNEL_LOOP(k, num_points) {
 
    // Get points
    const bool valid_match_ind = valid_match[block_id][k][0]; 
    const int64_t ind_Xi = valid_match_ind ? idx_ii2_jj[block_id][k] : 0;

    Xi[0] = Xs[ix][ind_Xi][0];
    Xi[1] = Xs[ix][ind_Xi][1];
    Xi[2] = Xs[ix][ind_Xi][2];
 
    Xj[0] = Xs[jx][k][0];
    Xj[1] = Xs[jx][k][1];
    Xj[2] = Xs[jx][k][2];

    // Get measurement pixel
    const int u_target = ind_Xi % width; 
    const int v_target = ind_Xi / width;
 
    // Transform point
    actSim3(tij, qij, sij, Xj, Xj_Ci);

    // // Check if in front of camera
    const bool valid_z = ((Xj_Ci[2] > z_eps) && (Xi[2] > z_eps));

    // Handle depth related vars
    const float zj_inv = valid_z ? 1.0/Xj_Ci[2] : 0.0;
    const float zj_log = valid_z ? logf(Xj_Ci[2]) : 0.0;
    const float zi_log = valid_z ? logf(Xi[2]) : 0.0; 

    // Project point
    const float x_div_z = Xj_Ci[0] * zj_inv;
    const float y_div_z = Xj_Ci[1] * zj_inv;
    const float u = fx * x_div_z + cx;
    const float v = fy * y_div_z + cy;

    // Handle proj
    const bool valid_u = ((u > pixel_border) && (u < width - 1 - pixel_border));
    const bool valid_v = ((v > pixel_border) && (v < height - 1 - pixel_border));

    // Error (difference in camera rays)
    err[0] = u - u_target;
    err[1] = v - v_target;
    err[2] = zj_log - zi_log; // Log-depth

    // Weights (Huber)
    const float q = Q[block_id][k][0];
    const float ci = Cs[ix][ind_Xi][0];
    const float cj = Cs[jx][k][0];
    const bool valid =
      valid_match_ind
      & (q > Q_thresh)
      & (ci > C_thresh)
      & (cj > C_thresh)
      & valid_u & valid_v & valid_z; // Check for valid image and depth
    
    // Weight using confidences
    const float conf_weight = q;
    
    const float sqrt_w_pixel = valid ? sigma_pixel_inv * sqrtf(conf_weight) : 0;
    const float sqrt_w_depth = valid ? sigma_depth_inv * sqrtf(conf_weight) : 0;

    // Robust weight
    w[0] = huber(sqrt_w_pixel * err[0]);
    w[1] = huber(sqrt_w_pixel * err[1]);
    w[2] = huber(sqrt_w_depth * err[2]);
    
    // Add back in sigma
    const float w_const_pixel = sqrt_w_pixel * sqrt_w_pixel;
    const float w_const_depth = sqrt_w_depth * sqrt_w_depth;
    w[0] *= w_const_pixel;
    w[1] *= w_const_pixel;
    w[2] *= w_const_depth;

    // Jacobians    

    // x coordinate
    Ji[0] = fx * zj_inv;
    Ji[1] = 0.0;
    Ji[2] = -fx * x_div_z * zj_inv;
    Ji[3] = -fx * x_div_z * y_div_z;
    Ji[4] = fx * (1 + x_div_z*x_div_z);
    Ji[5] = -fx * y_div_z; 
    Ji[6] = 0.0;

    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];


    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[0] * Jx[n] * Jx[m];
        l++;
      }
    }

    for (int n=0; n<7; n++) {
      vi[n] += w[0] * err[0] * Ji[n];
      vj[n] += w[0] * err[0] * Jj[n];
    }

    // y coordinate
    Ji[0] = 0.0; 
    Ji[1] = fy * zj_inv;
    Ji[2] = -fy * y_div_z * zj_inv;
    Ji[3] = -fy * (1 + y_div_z*y_div_z);
    Ji[4] = fy * x_div_z * y_div_z;
    Ji[5] = fy * x_div_z; 
    Ji[6] = 0.0;

    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[1] * Jx[n] * Jx[m];
        l++;
      }
    }

    for (int n=0; n<7; n++) {
      vi[n] += w[1] * err[1] * Ji[n];
      vj[n] += w[1] * err[1] * Jj[n];
    }

    // z coordinate
    Ji[0] = 0.0; 
    Ji[1] = 0.0; 
    Ji[2] = zj_inv;
    Ji[3] = y_div_z; // y
    Ji[4] = -x_div_z; // -x
    Ji[5] = 0.0;
    Ji[6] = 1.0; // z

    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[2] * Jx[n] * Jx[m];
        l++;
      }
    }

    for (int n=0; n<7; n++) {
      vi[n] += w[2] * err[2] * Ji[n];
      vj[n] += w[2] * err[2] * Jj[n];
    }

  }
 
  __syncthreads();
 
  __shared__ float sdata[THREADS];
  for (int n=0; n<7; n++) {
    sdata[threadIdx.x] = vi[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      gs[0][block_id][n] = sdata[0];
    }
 
    __syncthreads();
 
    sdata[threadIdx.x] = vj[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      gs[1][block_id][n] = sdata[0];
    }
 
  }
 
  l=0;
  for (int n=0; n<14; n++) {
    for (int m=0; m<=n; m++) {
      sdata[threadIdx.x] = hij[l];
      blockReduce(sdata);
 
      if (threadIdx.x == 0) {
        if (n<7 && m<7) {
          Hs[0][block_id][n][m] = sdata[0];
          Hs[0][block_id][m][n] = sdata[0];
        }
        else if (n >=7 && m<7) {
          Hs[1][block_id][m][n-7] = sdata[0];
          Hs[2][block_id][n-7][m] = sdata[0];
        }
        else {
          Hs[3][block_id][n-7][m-7] = sdata[0];
          Hs[3][block_id][m-7][n-7] = sdata[0];
        }
      }
 
      l++;
    }
  }
}


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
  const float delta_thresh)
{
  auto opts = Twc.options();
  const int num_edges = ii.size(0);
  const int num_poses = Xs.size(0);
  const int n = Xs.size(1);

  const int num_fix = 1;

  // Setup indexing
  torch::Tensor unique_kf_idx = get_unique_kf_idx(ii, jj);
  // For edge construction
  std::vector<torch::Tensor> inds = create_inds(unique_kf_idx, 0, ii, jj);
  torch::Tensor ii_edge = inds[0];
  torch::Tensor jj_edge = inds[1];
  // For linear system indexing (pin=2 because fixing first two poses)
  std::vector<torch::Tensor> inds_opt = create_inds(unique_kf_idx, num_fix, ii, jj);
  torch::Tensor ii_opt = inds_opt[0];
  torch::Tensor jj_opt = inds_opt[1];

  const int pose_dim = 7; // sim3

  // initialize buffers
  torch::Tensor Hs = torch::zeros({4, num_edges, pose_dim, pose_dim}, opts);
  torch::Tensor gs = torch::zeros({2, num_edges, pose_dim}, opts);

  // For debugging outputs
  torch::Tensor dx;

  torch::Tensor delta_norm;

  for (int itr=0; itr<max_iter; itr++) {

    calib_proj_kernel<<<num_edges, THREADS>>>(
      Twc.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      Xs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Cs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      K.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      ii_edge.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      jj_edge.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      idx_ii2jj.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
      valid_match.packed_accessor32<bool,3,torch::RestrictPtrTraits>(),
      Q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Hs.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      gs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      height, width, pixel_border, z_eps, sigma_pixel, sigma_depth, C_thresh, Q_thresh
    );


    // pose x pose block
    SparseBlock A(num_poses - num_fix, pose_dim);

    A.update_lhs(Hs.reshape({-1, pose_dim, pose_dim}), 
        torch::cat({ii_opt, ii_opt, jj_opt, jj_opt}), 
        torch::cat({ii_opt, jj_opt, ii_opt, jj_opt}));

    A.update_rhs(gs.reshape({-1, pose_dim}), 
        torch::cat({ii_opt, jj_opt}));

    // NOTE: Accounting for negative here!
    dx = -A.solve();

    
    pose_retr_kernel<<<1, THREADS>>>(
      Twc.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      num_fix);

    // Termination criteria
    // Need to specify this second argument otherwise ambiguous function call...
    delta_norm = torch::linalg::linalg_norm(dx, std::optional<c10::Scalar>(), {}, false, {});
    if (delta_norm.item<float>() < delta_thresh) {
      break;
    }
        

  }

  return {dx}; // For debugging
}

// =======================================================================
// Super-primitive depth refinement (poses frozen). Inspired by
// https://arxiv.org/pdf/2312.05889. Each keyframe is oversegmented (see
// como3r_slam/depth_refinement.py) and each segment owns ONE scalar
// log-scale s. Per-pixel depth is
//
//   d_p = d_init_p * exp(s_{seg(p)})
//
// so the Hessian is M x M (M = total number of segments) instead of N*P,
// which makes the system tiny and well-conditioned. The data residual
// is unchanged (3D point error in frame i, Huber + Q weighting); only
// the Jacobian chain-rules through ``d_p = d_init_p * exp(s)``:
//
//   dr/dd_p   = +/- ray_p
//   dd_p/ds   = d_p
//   => dr/ds  = +/- ray_p * d_p
//
// Per-segment H/g get atomicAdd'd because many pixels share one segment.
// A separate boundary-smoothness term adds a 1D residual per unique pair
// of adjacent segments (d_a * exp(s_a) - d_b * exp(s_b) -> 0), giving
// the M x M Hessian its only off-diagonal coupling. The solver is the
// same diagonal Jacobi sweep used elsewhere -- with the cross-segment
// coupling visited from BOTH sides of each pair, the result is the
// diagonal of the proper data + smoothness Hessian.
// =======================================================================

// Recompute current depths from segment log-scales:
//   d_p = d_init_p * exp(s_{seg(p)})
// Invalid pixels (labels[k][p] < 0) pass through with d = d_init unchanged.
__global__ void seg_compute_depths_kernel(
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> depths,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> depth_init,
    const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits>   labels,
    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits>   seg_offsets,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> seg_scale)
{
  const int N = depths.size(0);
  const int P = depths.size(1);
  const int total = N * P;

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += gridDim.x * blockDim.x) {
    const int k = idx / P;
    const int p = idx % P;
    const float d0 = depth_init[k][p];
    const int lab = labels[k][p];
    if (lab < 0) {
      depths[k][p] = d0;
    } else {
      const int g = seg_offsets[k] + lab;
      depths[k][p] = d0 * __expf(seg_scale[g]);
    }
  }
}

// Data-term residual accumulation: same 3D point error and weighting as
// the pose branch, but each pixel's contribution is funneled into the
// owning segment's scalar via the d_p chain-rule factor.
__global__ void seg_data_residual_accumulate_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Twc,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> depths,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> rays,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Cs,
    const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits>   labels,
    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits>   seg_offsets,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits>  ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits>  jj,
    const torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits>  idx_ii2_jj,
    const torch::PackedTensorAccessor32<bool,3,torch::RestrictPtrTraits>  valid_match,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Q,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits>       H_seg,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits>       g_seg,
    const float sigma_point,
    const float C_thresh,
    const float Q_thresh,
    const float huber_radius)
{
  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;

  const int P = depths.size(1);

  const int ix = static_cast<int>(ii[block_id]);
  const int jx = static_cast<int>(jj[block_id]);

  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];
  __shared__ float si[1], sj[1], sij[1];

  if (thread_id < 3) {
    ti[thread_id] = Twc[ix][thread_id];
    tj[thread_id] = Twc[jx][thread_id];
  }
  if (thread_id < 4) {
    qi[thread_id] = Twc[ix][thread_id + 3];
    qj[thread_id] = Twc[jx][thread_id + 3];
  }
  if (thread_id < 1) {
    si[thread_id] = Twc[ix][thread_id + 7];
    sj[thread_id] = Twc[jx][thread_id + 7];
  }
  __syncthreads();

  if (thread_id == 0) {
    relSim3(ti, qi, si, tj, qj, sj, tij, qij, sij);
  }
  __syncthreads();

  const float sigma_point_inv = 1.0f / sigma_point;
  const int seg_base_i = seg_offsets[ix];
  const int seg_base_j = seg_offsets[jx];

  GPU_1D_KERNEL_LOOP(k, P) {
    // Mirror existing "invalid -> weight=0" pattern: safe gather then mask.
    const bool vm = valid_match[block_id][k][0];
    long pi_raw = idx_ii2_jj[block_id][k];
    bool pi_in_range = (pi_raw >= 0) && (pi_raw < (long)P);
    const int pi = (vm && pi_in_range) ? (int)pi_raw : 0;

    float ray_i[3] = {rays[ix][pi][0], rays[ix][pi][1], rays[ix][pi][2]};
    float ray_j[3] = {rays[jx][k][0],  rays[jx][k][1],  rays[jx][k][2]};
    const float d_i = depths[ix][pi];
    const float d_j = depths[jx][k];

    // X_j_in_i = sR_ij * (d_j * ray_j) + t_ij
    float Xj[3] = {d_j * ray_j[0], d_j * ray_j[1], d_j * ray_j[2]};
    float Xj_Ci[3];
    actSim3(tij, qij, sij, Xj, Xj_Ci);

    // J_j = sR_ij * ray_j (rotation+scale, no translation)
    float v_j[3];
    actSO3(qij, ray_j, v_j);
    v_j[0] *= sij[0];
    v_j[1] *= sij[0];
    v_j[2] *= sij[0];

    // Residual (3D in frame i)
    float r[3] = {
      Xj_Ci[0] - d_i * ray_i[0],
      Xj_Ci[1] - d_i * ray_i[1],
      Xj_Ci[2] - d_i * ray_i[2],
    };

    // Segment IDs (with -1 = invalid pixel -> drop residual entirely).
    const int lab_i = labels[ix][pi];
    const int lab_j = labels[jx][k];

    const float q   = Q[block_id][k][0];
    const float ci  = Cs[ix][pi][0];
    const float cj  = Cs[jx][k][0];
    const bool valid =
        vm
      & pi_in_range
      & (q  > Q_thresh)
      & (ci > C_thresh)
      & (cj > C_thresh)
      & (lab_i >= 0)
      & (lab_j >= 0);

    const float sqrt_w = valid ? (sigma_point_inv * sqrtf(fmaxf(q, 0.0f))) : 0.0f;
    const float sqrt_w2 = sqrt_w * sqrt_w;

    float w[3];
    #pragma unroll
    for (int c = 0; c < 3; c++) {
      const float a = fabsf(sqrt_w * r[c]);
      const float hw = (a < huber_radius) ? 1.0f : (huber_radius / fmaxf(a, 1e-12f));
      w[c] = hw * sqrt_w2;
    }

    // Per-pixel Hessian / gradient contributions, chain-ruled to the
    // segment scalar via dd/ds = d:
    //   J_i_d = -ray_i,  J_j_d = v_j
    //   J_i_s = -ray_i * d_i,  J_j_s = v_j * d_j
    //   H_s += d^2 * (J_d^T W J_d),  g_s += d * (J_d^T W r) * (-1 for i, +1 for j)
    const float H_i_e_d =
        w[0] * ray_i[0] * ray_i[0]
      + w[1] * ray_i[1] * ray_i[1]
      + w[2] * ray_i[2] * ray_i[2];
    const float H_j_e_d =
        w[0] * v_j[0] * v_j[0]
      + w[1] * v_j[1] * v_j[1]
      + w[2] * v_j[2] * v_j[2];
    const float g_i_e_d = -(
        w[0] * ray_i[0] * r[0]
      + w[1] * ray_i[1] * r[1]
      + w[2] * ray_i[2] * r[2]);
    const float g_j_e_d =
        w[0] * v_j[0] * r[0]
      + w[1] * v_j[1] * r[1]
      + w[2] * v_j[2] * r[2];

    if (!valid) continue;

    const int gseg_i = seg_base_i + lab_i;
    const int gseg_j = seg_base_j + lab_j;

    atomicAdd(&H_seg[gseg_i], H_i_e_d * d_i * d_i);
    atomicAdd(&g_seg[gseg_i], g_i_e_d * d_i);
    atomicAdd(&H_seg[gseg_j], H_j_e_d * d_j * d_j);
    atomicAdd(&g_seg[gseg_j], g_j_e_d * d_j);
  }
}

// Boundary smoothness in LOG-SPACE: per pair (a, b) of adjacent segments,
// penalize the ratio d_a/d_b deviating from the MASt3R-suggested ratio
// d_init_a/d_init_b. Equivalently, log d_a - log d_b should match
// log d_init_a - log d_init_b. With d = d_init * exp(s), log d = log d_init
// + s, so the residual collapses to
//
//   r = (s_a - s_b) + log_d_init_diff_ab
//
// which is LINEAR in s -- no exp(), so no risk of overflow blowing up the
// Newton step into NaN, no matter how far s drifts. The Jacobians are
// constant (+1, -1) so H and g additions are trivially bounded.
__global__ void seg_boundary_smoothness_kernel(
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> seg_scale,
    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits>   bnd_seg_a,
    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits>   bnd_seg_b,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> bnd_log_d_diff,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits>       H_seg,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits>       g_seg,
    const float lambda_smooth)
{
  const int B = bnd_seg_a.size(0);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < B;
       idx += gridDim.x * blockDim.x) {
    const int a = bnd_seg_a[idx];
    const int b = bnd_seg_b[idx];
    const float r = (seg_scale[a] - seg_scale[b]) + bnd_log_d_diff[idx];
    // H += lambda, g += +/- lambda * r (J_s = +/-1)
    atomicAdd(&H_seg[a], lambda_smooth);
    atomicAdd(&g_seg[a], lambda_smooth * r);
    atomicAdd(&H_seg[b], lambda_smooth);
    atomicAdd(&g_seg[b], -lambda_smooth * r);
  }
}

// Levenberg-Marquardt + prior + step clamp Newton update on the segment
// scalar. Skips pinned KFs' segments (seg_kf[m] < num_fix).
//
//   prior:  H += lambda_prior, g += lambda_prior * s (anchors s -> 0,
//           i.e. depth = MASt3R prediction)
//   LM:     H_damped = (H + lambda_prior) * (1 + lambda_lm)
//           -- shrinks the Newton step under uncertainty, prevents the
//           pure-GN overshoot that would otherwise feed inf back through
//           exp(s) into the next iteration.
//   clamp:  |delta_s| <= s_step_clip (per-iter step bound).
__global__ void seg_update_kernel(
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> seg_scale,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> H_seg,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> g_seg,
    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits>   seg_kf,
    const int num_fix,
    const float lambda_prior,
    const float lambda_lm,
    const float s_step_clip)
{
  const int M = seg_scale.size(0);
  for (int m = blockIdx.x * blockDim.x + threadIdx.x;
       m < M;
       m += gridDim.x * blockDim.x) {
    if (seg_kf[m] < num_fix) continue;
    const float s = seg_scale[m];
    const float H = (H_seg[m] + lambda_prior) * (1.0f + lambda_lm);
    const float g = g_seg[m] + lambda_prior * s;
    float ds = -g / fmaxf(H, 1e-12f);
    // Per-iter step clamp -- bounds |s| growth and immunizes the next
    // exp(s) in seg_compute_depths_kernel from blowing past float32 range.
    if (ds >  s_step_clip) ds =  s_step_clip;
    if (ds < -s_step_clip) ds = -s_step_clip;
    seg_scale[m] = s + ds;
  }
}


std::vector<torch::Tensor> gauss_newton_seg_depths_cuda(
    torch::Tensor poses,
    torch::Tensor depth_init,
    torch::Tensor rays,
    torch::Tensor Cs,
    torch::Tensor labels,
    torch::Tensor seg_offsets,
    torch::Tensor seg_kf,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor idx_ii2jj,
    torch::Tensor valid_match,
    torch::Tensor Q,
    torch::Tensor bnd_seg_a,
    torch::Tensor bnd_seg_b,
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
    const int num_iters)
{
  (void)depth_min;   // depth floor not needed; s is unbounded above 0.
  auto opts_f = depth_init.options();
  const int N = depth_init.size(0);
  const int P = depth_init.size(1);
  const int E = ii.size(0);
  const int B = bnd_seg_a.size(0);

  auto seg_scale = torch::zeros({M}, opts_f);
  auto depths    = depth_init.clone();
  auto H_seg     = torch::zeros({M}, opts_f);
  auto g_seg     = torch::zeros({M}, opts_f);

  const int compute_blocks = (N * P + THREADS - 1) / THREADS;
  const int update_blocks  = M > 0 ? (M + THREADS - 1) / THREADS : 1;
  const int bnd_blocks     = B > 0 ? (B + THREADS - 1) / THREADS : 0;
  const bool smooth_on     = (lambda_smooth > 0.0f) && (B > 0);

  for (int it = 0; it < num_iters; it++) {
    // Refresh depths from current seg_scale.
    seg_compute_depths_kernel<<<compute_blocks, THREADS>>>(
        depths.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        depth_init.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        labels.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
        seg_offsets.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
        seg_scale.packed_accessor32<float,1,torch::RestrictPtrTraits>()
    );

    H_seg.zero_();
    g_seg.zero_();

    if (E > 0 && M > 0) {
      seg_data_residual_accumulate_kernel<<<E, THREADS>>>(
          poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
          depths.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
          rays.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
          Cs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
          labels.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
          seg_offsets.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
          ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
          jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
          idx_ii2jj.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
          valid_match.packed_accessor32<bool,3,torch::RestrictPtrTraits>(),
          Q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
          H_seg.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
          g_seg.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
          sigma_point, C_thresh, Q_thresh, huber_radius
      );
    }

    if (smooth_on) {
      seg_boundary_smoothness_kernel<<<bnd_blocks, THREADS>>>(
          seg_scale.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
          bnd_seg_a.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
          bnd_seg_b.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
          bnd_log_d_diff.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
          H_seg.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
          g_seg.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
          lambda_smooth
      );
    }

    if (M > 0) {
      seg_update_kernel<<<update_blocks, THREADS>>>(
          seg_scale.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
          H_seg.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
          g_seg.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
          seg_kf.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
          num_fix, lambda_prior, lambda_lm, s_step_clip
      );
    }
  }

  // Final refresh so the returned depths reflect the converged scales.
  seg_compute_depths_kernel<<<compute_blocks, THREADS>>>(
      depths.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      depth_init.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      labels.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
      seg_offsets.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
      seg_scale.packed_accessor32<float,1,torch::RestrictPtrTraits>()
  );

  // Caching-allocator alias safety: H_seg / g_seg go out of scope at return;
  // sync first so in-flight kernel writes finish before their storage is
  // returned to the pool (see feedback_cuda_kernel_temporary_buffers note).
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  return {depths, seg_scale};
}
