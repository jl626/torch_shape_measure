// torch library headers
#include <torch/extension.h>
#include <THC/THC.h>

// C++ standard header
#include <algorithm>
#include <vector>
#include <math.h>
#include <omp.h>
#include <cstdio>
#include <iostream>
#include <memory>

// CUDA and/or cuBLAS header
#include "cuda_helper.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolver_common.h>
#include <cusolverDn.h>

// Mark: CUDA EMD primal form (through sinkhorn iteration) from Optas github
void approxmatchLauncher(int b,
    int n,
    int m,
    const float * xyz1,
    const float * xyz2,
    float * match,
    float * temp);

void matchcostLauncher(int b,
    int n,
    int m,
    const float * xyz1,
    const float * xyz2,
    const float * match,
    float * out);

void matchcostgradLauncher(int b,
    int n,
    int m,
    const float * xyz1,
    const float * xyz2,
    const float * match,
    float * grad1,
    float * grad2);

// Mark: CUDA Chamfer distance
int ChamferDistanceKernelLauncher(
    const int b, const int n,
    const float* xyz,
    const int m,
    const float* xyz2,
    float* result,
    int* result_i,
    float* result2,
    int* result2_i);

int ChamferDistanceGradKernelLauncher(
    const int b, const int n,
    const float* xyz1,
    const int m,
    const float* xyz2,
    const float* grad_dist1,
    const int* idx1,
    const float* grad_dist2,
    const int* idx2,
    float* grad_xyz1,
    float* grad_xyz2);

// Mark: Varifold Kernel 
void CudaKernelLauncher(int b,
    int n,
    int m,
    const float * xyz1,
    const float * xyz2,
	const float * nor1,
    const float * nor2,
	float * match);

void CudaKernelGradLauncher(int b,
    int n,
    int m,
    const float * xyz1,
    const float * xyz2,
	const float * nor1,
    const float * nor2,
    float * grad_xy,
    float * grad_yx,
	float * grad_x,
    float * grad_y);

cublasHandle_t getCurrentCUDABlasHandle() {
    return THCState_getCurrentBlasHandle(at::globalContext().getTHCState());
}

template<int success = CUSOLVER_STATUS_SUCCESS, class T, class Status> // , class A = Status(*)(P), class D = Status(*)(T)>
std::unique_ptr<T, Status(*)(T*)> unique_allocate(Status(allocator)(T**),  Status(deleter)(T*))
{
    T* ptr;
    auto stat = allocator(&ptr);
    AT_CHECK(stat == success);
    return {ptr, deleter};
}

template <class T>
std::unique_ptr<T, decltype(&cudaFree)> unique_cuda_ptr(size_t len) {
    T* ptr;
    auto stat = cudaMalloc(&ptr, sizeof(T) * len);
    AT_CHECK(stat == cudaSuccess);
    return {ptr, cudaFree};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
batch_svd_forward(at::Tensor a, bool is_sort, double tol=1e-7, int max_sweeps=100)
{
    AT_CHECK(a.is_cuda(), "only cuda tensor is supported");
    AT_CHECK(a.dtype() == at::kFloat, "only float is supported");

    auto handle_ptr = unique_allocate(cusolverDnCreate, cusolverDnDestroy);
    const auto A = a.contiguous().clone().transpose(1, 2).contiguous().transpose(1, 2);
    // const auto A = a;
    const auto batch_size = A.size(0);
    const auto m = A.size(1);
    AT_CHECK(m <= 32, "matrix row should be <= 32");
    const auto n = A.size(2);
    AT_CHECK(n <= 32, "matrix col should be <= 32");
    const auto lda = m;
    const auto d_A = A.data<float>();
    const auto minmn = std::min(m, n);
    auto s = at::empty({batch_size, minmn}, a.type());
    auto d_s = s.data<float>();
    auto U = at::empty({batch_size, m, m}, a.type());
    const auto d_U = U.data<float>();
    const auto ldu = m;
    auto V = at::empty({batch_size, n, n}, a.type());
    const auto d_V = V.data<float>();
    const auto ldv = n;

    auto params = unique_allocate(cusolverDnCreateGesvdjInfo, cusolverDnDestroyGesvdjInfo);
    auto status = cusolverDnXgesvdjSetTolerance(params.get(), tol);
    AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);
    status = cusolverDnXgesvdjSetMaxSweeps(params.get(), max_sweeps);
    AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);
    status = cusolverDnXgesvdjSetSortEig(params.get(), is_sort);
    AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);

    auto jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors
    int lwork;
    auto status_buffer = cusolverDnSgesvdjBatched_bufferSize(
        handle_ptr.get(),
        jobz,
        m,
        n,
        d_A,
        lda,
        d_s,
        d_U,
        ldu,
        d_V,
        ldv,
        &lwork,
        params.get(),
        batch_size);
    AT_CHECK(CUSOLVER_STATUS_SUCCESS == status_buffer);
    auto work_ptr = unique_cuda_ptr<float>(lwork);
    auto info_ptr = unique_cuda_ptr<int>(batch_size);
    status = cusolverDnSgesvdjBatched(
        handle_ptr.get(),
        jobz,
        m,
        n,
        d_A,
        lda,
        d_s,
        d_U,
        ldu,
        d_V,
        ldv,
        work_ptr.get(),
        lwork,
        info_ptr.get(),
        params.get(),
        batch_size
        );
    AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);

    std::vector<int> hinfo(batch_size);
    auto status_memcpy = cudaMemcpy(hinfo.data(), info_ptr.get(), sizeof(int) * batch_size, cudaMemcpyDeviceToHost);
    AT_CHECK(cudaSuccess == status_memcpy);

    for(int i = 0 ; i < batch_size; ++i)
    {
        if ( 0 == hinfo[i] )
        {
            continue;
        }
        else if ( 0 > hinfo[i] )
        {
            printf("Error: %d-th parameter is wrong \n", -hinfo[i]);
            AT_CHECK(false);
        }
        else
        {
            printf("WARNING: matrix %d, info = %d : Jacobi method does not converge \n", i, hinfo[i] );
        }
    }

    // U = U.contiguous().transpose(1, 2).contiguous().transpose(1, 2);
    // s = s.contiguous().transpose(0, 1).contiguous().transpose(0, 1);
    // V = V.contiguous().transpose(1, 2).contiguous().transpose(1, 2);
    U = U.contiguous().transpose(1, 2).contiguous();
    s = s.contiguous();
    V = V.contiguous().transpose(1, 2).contiguous();

    return std::make_tuple(U, s, V);
}



// https://j-towns.github.io/papers/svd-derivative.pdf
//
// This makes no assumption on the signs of sigma.
at::Tensor batch_svd_backward(const std::vector<at::Tensor> &grads, const at::Tensor& self,
          bool some, bool compute_uv, const at::Tensor& raw_u, const at::Tensor& sigma, const at::Tensor& raw_v) {
  AT_CHECK(compute_uv,
           "svd_backward: Setting compute_uv to false in torch.svd doesn't compute singular matrices, ",
           "and hence we cannot compute backward. Please use torch.svd(compute_uv=True)");

  // A [b, m, n]
  // auto b = self.size(0);
  auto m = self.size(1);
  auto n = self.size(2);
  auto k = sigma.size(1);
  auto gsigma = grads[1];

  auto u = raw_u;
  auto v = raw_v;
  auto gu = grads[0];
  auto gv = grads[2];

  if (!some) {
    // We ignore the free subspace here because possible base vectors cancel
    // each other, e.g., both -v and +v are valid base for a dimension.
    // Don't assume behavior of any particular implementation of svd.
    u = raw_u.narrow(2, 0, k);
    v = raw_v.narrow(2, 0, k);
    if (gu.defined()) {
      gu = gu.narrow(2, 0, k);
    }
    if (gv.defined()) {
      gv = gv.narrow(2, 0, k);
    }
  }
  auto vt = v.transpose(1, 2);

  at::Tensor sigma_term;
  if (gsigma.defined()) {
    sigma_term = u.bmm(gsigma.diag_embed()).bmm(vt);
  } else {
    sigma_term = at::zeros({1}, self.options()).expand_as(self);
  }
  // in case that there are no gu and gv, we can avoid the series of kernel
  // calls below
  if (!gv.defined() && !gu.defined()) {
    return sigma_term;
  }

  auto ut = u.transpose(1, 2);
  auto im = at::eye(m, self.options());  // work if broadcast
  auto in = at::eye(n, self.options());
  auto sigma_mat = sigma.diag_embed();
  auto sigma_mat_inv = sigma.pow(-1).diag_embed();
  auto sigma_expanded_sq = sigma.pow(2).unsqueeze(1).expand_as(sigma_mat);
  auto F = sigma_expanded_sq - sigma_expanded_sq.transpose(1, 2);
  // The following two lines invert values of F, and fills the diagonal with 0s.
  // Notice that F currently has 0s on diagonal. So we fill diagonal with +inf
  // first to prevent nan from appearing in backward of this function.
  F.diagonal(0, -2, -1).fill_(INFINITY);
  F = F.pow(-1);

  at::Tensor u_term, v_term;

  if (gu.defined()) {
    u_term = u.bmm(F.mul(ut.bmm(gu) - gu.transpose(1, 2).bmm(u))).bmm(sigma_mat);
    if (m > k) {
      u_term = u_term + (im - u.bmm(ut)).bmm(gu).bmm(sigma_mat_inv);
    }
    u_term = u_term.bmm(vt);
  } else {
    u_term = at::zeros({1}, self.options()).expand_as(self);
  }

  if (gv.defined()) {
    auto gvt = gv.transpose(1, 2);
    v_term = sigma_mat.bmm(F.mul(vt.bmm(gv) - gvt.bmm(v))).bmm(vt);
    if (n > k) {
      v_term = v_term + sigma_mat_inv.bmm(gvt.bmm(in - v.bmm(vt)));
    }
    v_term = u.bmm(v_term);
  } else {
    v_term = at::zeros({1}, self.options()).expand_as(self);
  }

  return u_term + sigma_term + v_term;
}

at::Tensor Varifold_Kernel_forward_cuda(const at::Tensor xyz1, 
    const at::Tensor xyz2,
	const at::Tensor nor1, 
	const at::Tensor nor2) 
{
	// Allocate necessary data structures
	at::Tensor match_x = at::zeros({xyz1.size(0), xyz1.size(1), xyz1.size(1)}, 
		xyz1.options());
	at::Tensor match_y = at::zeros({xyz1.size(0), xyz2.size(1), xyz2.size(1)}, 
		xyz2.options());
	at::Tensor match_xy = at::zeros({xyz1.size(0), xyz1.size(1), xyz2.size(1)}, 
		xyz1.options());
		

	// Find the approximate matching 
	CudaKernelLauncher(xyz1.size(0), 
    xyz1.size(1), 
    xyz1.size(1),
    xyz1.data<float>(),
    xyz1.data<float>(),
	nor1.data<float>(),
	nor1.data<float>(),
	match_x.data<float>());

	CudaKernelLauncher(xyz1.size(0), 
    xyz2.size(1), 
    xyz2.size(1),
    xyz2.data<float>(),
    xyz2.data<float>(),
	nor2.data<float>(),
	nor2.data<float>(),
	match_y.data<float>());

	CudaKernelLauncher(xyz1.size(0), 
    xyz1.size(1), 
    xyz2.size(1),
    xyz1.data<float>(),
    xyz2.data<float>(),
	nor1.data<float>(),
	nor2.data<float>(),
	match_xy.data<float>());

	auto match = match_x.sum() + match_y.sum() - 2.0 * match_xy.sum();
	// return output
	return match;
}

std::vector<at::Tensor> Varifold_Kernel_backward_cuda(const at::Tensor xyz1,
    const at::Tensor xyz2,
	const at::Tensor nor1,
	const at::Tensor nor2)
{
	// Allocate necessary data structures
	at::Tensor grad_x = at::zeros_like(xyz1);
	at::Tensor grad_xy = at::zeros_like(xyz1);
	at::Tensor grad_y = at::zeros_like(xyz2);
	at::Tensor grad_yx = at::zeros_like(xyz2);
	
    CudaKernelGradLauncher(xyz1.size(0), 
    xyz1.size(1), 
    xyz2.size(1), 
    xyz1.data<float>(),
    xyz2.data<float>(), 
    nor1.data<float>(),
    nor2.data<float>(),
    grad_xy.data<float>(), 
    grad_yx.data<float>(),
	grad_x.data<float>(),
	grad_y.data<float>());

	// get output
	auto gradxyz1 = 4*(grad_x - grad_xy);
	auto gradxyz2 = 4*(grad_y - grad_yx);

    // return gradients
    return {gradxyz1, gradxyz2};
}

std::vector<at::Tensor> emd_distance_forward_cuda(const at::Tensor xyz1, 
    const at::Tensor xyz2) 
{
	// Allocate necessary data structures
	at::Tensor match = at::zeros({xyz1.size(0), xyz1.size(1), xyz2.size(1)}, 
		xyz1.options());
	at::Tensor cost = at::zeros({xyz1.size(0)}, xyz1.options());
	at::Tensor temp = at::zeros({xyz1.size(0), 2 * (xyz1.size(1) + xyz2.size(1))}, 
		xyz1.options());

	// Find the approximate matching 
	approxmatchLauncher(xyz1.size(0), 
    xyz1.size(1), 
    xyz2.size(1),
    xyz1.data<float>(),
    xyz2.data<float>(),
	match.data<float>(),
    temp.data<float>());

	// Compute the matching cost
	matchcostLauncher(xyz1.size(0), 
    xyz1.size(1), 
    xyz2.size(1),
    xyz1.data<float>(),
    xyz2.data<float>(),
	match.data<float>(),
	cost.data<float>());

    return {cost, match};
}

// CUDA 

std::vector<at::Tensor> emd_distance_backward_cuda(const at::Tensor xyz1,
    const at::Tensor xyz2,
    const at::Tensor match)
{
	// Allocate necessary data structures
	at::Tensor gradxyz1 = at::zeros_like(xyz1);
	at::Tensor gradxyz2 = at::zeros_like(xyz2);

    matchcostgradLauncher(xyz1.size(0), 
    xyz1.size(1), 
    xyz2.size(1), 
    xyz1.data<float>(),
    xyz2.data<float>(),
    match.data<float>(), 
    gradxyz1.data<float>(), 
    gradxyz2.data<float>());
    
    // return gradients
    return {gradxyz1, gradxyz2};
}

void chamfer_distance_forward_cuda(
    const at::Tensor xyz1, 
    const at::Tensor xyz2, 
    const at::Tensor dist1, 
    const at::Tensor dist2, 
    const at::Tensor idx1, 
    const at::Tensor idx2) 
{
    ChamferDistanceKernelLauncher(xyz1.size(0),
    xyz1.size(1),
    xyz1.data<float>(),
    xyz2.size(1),
    xyz2.data<float>(),
    dist1.data<float>(), 
    idx1.data<int>(),
    dist2.data<float>(), 
    idx2.data<int>());
}

void chamfer_distance_backward_cuda(
    const at::Tensor xyz1,
    const at::Tensor xyz2, 
    at::Tensor gradxyz1, 
    at::Tensor gradxyz2, 
    at::Tensor graddist1, 
    at::Tensor graddist2, 
    at::Tensor idx1, 
    at::Tensor idx2)
{
    ChamferDistanceGradKernelLauncher(xyz1.size(0),
    xyz1.size(1),
    xyz1.data<float>(),
    xyz2.size(1),
    xyz2.data<float>(),
    graddist1.data<float>(), 
    idx1.data<int>(),
    graddist2.data<float>(), 
    idx2.data<int>(),
    gradxyz1.data<float>(), gradxyz2.data<float>());
}



//'''
//
//CPU function wrappers!!!
//'''


// Mark: Wasserstein (EMD) cpu function

void approxmatch_cpu(int b,int n,int m,const float * xyz1,const float * xyz2,float * match){
	for (int i=0;i<b;i++){
		int factorl=std::max(n,m)/n;
		int factorr=std::max(n,m)/m;
		std::vector<double> saturatedl(n,double(factorl)),saturatedr(m,double(factorr));
		std::vector<double> weight(n*m);
		for (int j=0;j<n*m;j++)
			match[j]=0;
		for (int j=8;j>=-2;j--){
			//printf("i=%d j=%d\n",i,j);
			double level=-powf(4.0,j);
			if (j==-2)
				level=0;
			for (int k=0;k<n;k++){
				double x1=xyz1[k*3+0];
				double y1=xyz1[k*3+1];
				double z1=xyz1[k*3+2];
				for (int l=0;l<m;l++){
					double x2=xyz2[l*3+0];
					double y2=xyz2[l*3+1];
					double z2=xyz2[l*3+2];
					weight[k*m+l]=expf(level*((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2)))*saturatedr[l];
				}
			}
			std::vector<double> ss(m,1e-9);
			for (int k=0;k<n;k++){
				double s=1e-9;
				for (int l=0;l<m;l++){
					s+=weight[k*m+l];
				}
				for (int l=0;l<m;l++){
					weight[k*m+l]=weight[k*m+l]/s*saturatedl[k];
				}
				for (int l=0;l<m;l++)
					ss[l]+=weight[k*m+l];
			}
			for (int l=0;l<m;l++){
				double s=ss[l];
				double r=std::min(saturatedr[l]/s,1.0);
				ss[l]=r;
			}
			std::vector<double> ss2(m,0);
			for (int k=0;k<n;k++){
				double s=0;
				for (int l=0;l<m;l++){
					weight[k*m+l]*=ss[l];
					s+=weight[k*m+l];
					ss2[l]+=weight[k*m+l];
				}
				saturatedl[k]=std::max(saturatedl[k]-s,0.0);
			}
			for (int k=0;k<n*m;k++)
				match[k]+=weight[k];
			for (int l=0;l<m;l++){
				saturatedr[l]=std::max(saturatedr[l]-ss2[l],0.0);
			}
		}
		xyz1+=n*3;
		xyz2+=m*3;
		match+=n*m;
	}
}

void matchcost_cpu(int b,int n,int m,const float * xyz1,const float * xyz2,const float * match,float * cost){
	for (int i=0;i<b;i++){
		double s=0;
		for (int j=0;j<n;j++)
			for (int k=0;k<m;k++){
				float x1=xyz1[j*3+0];
				float y1=xyz1[j*3+1];
				float z1=xyz1[j*3+2];
				float x2=xyz2[k*3+0];
				float y2=xyz2[k*3+1];
				float z2=xyz2[k*3+2];
				float d=sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1))*match[j*m+k];
				s+=d;
			}
		cost[0]=s;
		xyz1+=n*3;
		xyz2+=m*3;
		match+=n*m;
		cost+=1;
	}
}

void matchcostgrad_cpu(int b,int n,int m,const float * xyz1,const float * xyz2,const float * match,float * grad1,float * grad2){
	for (int i=0;i<b;i++){
		for (int j=0;j<n;j++)
			grad1[j*3+0]=0;
		for (int j=0;j<m;j++){
			float sx=0,sy=0,sz=0;
			for (int k=0;k<n;k++){
				float x2=xyz2[j*3+0];
				float y2=xyz2[j*3+1];
				float z2=xyz2[j*3+2];
				float x1=xyz1[k*3+0];
				float y1=xyz1[k*3+1];
				float z1=xyz1[k*3+2];
				float d=std::max(sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)),1e-20f);
				float dx=match[k*m+j]*((x2-x1)/d);
				float dy=match[k*m+j]*((y2-y1)/d);
				float dz=match[k*m+j]*((z2-z1)/d);
				grad1[k*3+0]-=dx;
				grad1[k*3+1]-=dy;
				grad1[k*3+2]-=dz;
				sx+=dx;
				sy+=dy;
				sz+=dz;
			}
			grad2[j*3+0]=sx;
			grad2[j*3+1]=sy;
			grad2[j*3+2]=sz;
		}
		xyz1+=n*3;
		xyz2+=m*3;
		match+=n*m;
		grad1+=n*3;
		grad2+=m*3;
	}
}

// Mark: Chamfer distance cpu function

void nnsearch(
    const int b, const int n, const int m,
    const float* xyz1,
    const float* xyz2,
    float* dist,
    int* idx)
{
    for (int i = 0; i < b; i++) {
        for (int j = 0; j < n; j++) {
            const float x1 = xyz1[(i*n+j)*3+0];
            const float y1 = xyz1[(i*n+j)*3+1];
            const float z1 = xyz1[(i*n+j)*3+2];
            double best = 0;
            int besti = 0;
            for (int k = 0; k < m; k++) {
                const float x2 = xyz2[(i*m+k)*3+0] - x1;
                const float y2 = xyz2[(i*m+k)*3+1] - y1;
                const float z2 = xyz2[(i*m+k)*3+2] - z1;
                const double d=x2*x2+y2*y2+z2*z2;
                if (k==0 || d < best){
                    best = d;
                    besti = k;
                }
            }
            dist[i*n+j] = best;
            idx[i*n+j] = besti;
        }
    }
}



//* batch Euclidean grid cpu
void varifold_kernel_cpu(int b,
	int n,
	int m,
	const float * xyz1,
	const float * xyz2, 
	const float * nor1, 
	const float * nor2,
	float * match){
	for (int i=0;i<b;i++){
		std::vector<double> weight(n*m);
		for (int k=0;k<n;k++){
			double x1 =xyz1[k*3+0];
			double y1 =xyz1[k*3+1];
			double z1 =xyz1[k*3+2];
			double nx1=nor1[k*3+0];
			double ny1=nor1[k*3+1];
			double nz1=nor1[k*3+2];
			for (int l=0;l<m;l++){
				double x2 =xyz2[l*3+0];
				double y2 =xyz2[l*3+1];
				double z2 =xyz2[l*3+2];
				double nx2=nor2[l*3+0];
				double ny2=nor2[l*3+1];
				double nz2=nor2[l*3+2];
				match[k*m+l]=std::max(expf(-1.0*((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2)))*powf(nx1*nx2+ny1*ny2+nz1*nz2,2),1e-10f);
			}
		}
		xyz1+=n*3;
		xyz2+=m*3;
		nor1+=n*3;
		nor2+=m*3;
		match+=n*m;
	}
}

void varifold_grad_cpu(int b,
	int n,
	int m,
	const float * xyz1,
	const float * xyz2,
	const float * nor1,
	const float * nor2,
	float * grad1,
	float * grad2){
	for (int i=0;i<b;i++){
		for (int j=0;j<n;j++)
			grad1[j*3+0]=0;
		for (int j=0;j<m;j++){
			float sx=0,sy=0,sz=0;
			for (int k=0;k<n;k++){
				float  x2=xyz2[j*3+0];
				float  y2=xyz2[j*3+1];
				float  z2=xyz2[j*3+2];
				float nx2=nor2[j*3+0];
				float ny2=nor2[j*3+1];
				float nz2=nor2[j*3+2];

				float  x1=xyz1[k*3+0];
				float  y1=xyz1[k*3+1];
				float  z1=xyz1[k*3+2];
				float nx1=nor1[k*3+0];
				float ny1=nor1[k*3+1];
				float nz1=nor1[k*3+2];
				
				float d=-1.0*std::max(expf(-1.0*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1))),1e-20f);
				float g=std::max(powf(nx1*nx2+ny1*ny2+nz1*nz2,2),1e-20f);
				float dx=d*g*(x2-x1);
				float dy=d*g*(y2-y1);
				float dz=d*g*(z2-z1);
				grad1[k*3+0]-=dx;
				grad1[k*3+1]-=dy;
				grad1[k*3+2]-=dz;
				sx+=dx;
				sy+=dy;
				sz+=dz;
			}
			grad2[j*3+0]=sx;
			grad2[j*3+1]=sy;
			grad2[j*3+2]=sz;
		}
		xyz1+=n*3;
		xyz2+=m*3;
		nor1+=n*3;
		nor2+=m*3;
		grad1+=n*3;
		grad2+=m*3;
	}
}

void varifold_grad_cpu2(int b,
	int n,
	int m,
	const float * xyz1,
	const float * xyz2,
	const float * nor1,
	const float * nor2,
	float * grad1){
	for (int i=0;i<b;i++){
		for (int j=0;j<m;j++){
			float sx=0,sy=0,sz=0;
			for (int k=0;k<n;k++){
				float  x2=xyz2[j*3+0];
				float  y2=xyz2[j*3+1];
				float  z2=xyz2[j*3+2];
				float nx2=nor2[j*3+0];
				float ny2=nor2[j*3+1];
				float nz2=nor2[j*3+2];

				float  x1=xyz1[k*3+0];
				float  y1=xyz1[k*3+1];
				float  z1=xyz1[k*3+2];
				float nx1=nor1[k*3+0];
				float ny1=nor1[k*3+1];
				float nz1=nor1[k*3+2];
				
				float d=-1.0*std::max(expf(-1.0*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1))),1e-20f);
				float g=std::max(powf(nx1*nx2+ny1*ny2+nz1*nz2,2),1e-20f);
				float dx=d*g*(x2-x1);
				float dy=d*g*(y2-y1);
				float dz=d*g*(z2-z1);
				sx+=dx;
				sy+=dy;
				sz+=dz;
			}
			grad1[j*3+0]=sx;
			grad1[j*3+1]=sy;
			grad1[j*3+2]=sz;
		}
		xyz1+=n*3;
		xyz2+=m*3;
		nor1+=n*3;
		nor2+=m*3;
		grad1+=n*3;
	}
}


std::vector<at::Tensor>  emd_distance_forward(
    const at::Tensor xyz1, 
    const at::Tensor xyz2){

	// Allocate necessary data structures
	at::Tensor match = at::zeros({xyz1.size(0), xyz1.size(1), xyz2.size(1)}, 
		xyz1.options());
	at::Tensor cost = at::zeros({xyz1.size(0)}, xyz1.options());
	// Find the approximate matching 

	approxmatch_cpu(xyz1.size(0), 
    xyz1.size(1), 
    xyz2.size(1),
    xyz1.data<float>(),
    xyz2.data<float>(),
	match.data<float>());

	// Compute the matching cost
	matchcost_cpu(xyz1.size(0), 
    xyz1.size(1), 
    xyz2.size(1),
    xyz1.data<float>(),
    xyz2.data<float>(),
	match.data<float>(),
	cost.data<float>());

    // return output
	return {cost, match};
}

std::vector<at::Tensor>  emd_distance_backward(
    const at::Tensor xyz1, 
    const at::Tensor xyz2,
    const at::Tensor match){

	// Allocate necessary data structures
	at::Tensor gradxyz1 = at::zeros_like(xyz1);
	at::Tensor gradxyz2 = at::zeros_like(xyz2);

    matchcostgrad_cpu(xyz1.size(0), 
    xyz1.size(1), 
    xyz2.size(1), 
    xyz1.data<float>(),
    xyz2.data<float>(),
    match.data<float>(), 
    gradxyz1.data<float>(), 
    gradxyz2.data<float>());    

    // return gradients
    return {gradxyz1, gradxyz2};

}

void chamfer_distance_forward(
    const at::Tensor xyz1, 
    const at::Tensor xyz2, 
    const at::Tensor dist1, 
    const at::Tensor dist2, 
    const at::Tensor idx1, 
    const at::Tensor idx2) 
{
    const int batchsize = xyz1.size(0);
    const int n = xyz1.size(1);
    const int m = xyz2.size(1);

    const float* xyz1_data = xyz1.data<float>();
    const float* xyz2_data = xyz2.data<float>();
    float* dist1_data = dist1.data<float>();
    float* dist2_data = dist2.data<float>();
    int* idx1_data = idx1.data<int>();
    int* idx2_data = idx2.data<int>();

    nnsearch(batchsize, n, m, xyz1_data, xyz2_data, dist1_data, idx1_data);
    nnsearch(batchsize, m, n, xyz2_data, xyz1_data, dist2_data, idx2_data);
}


void chamfer_distance_backward(
    const at::Tensor xyz1, 
    const at::Tensor xyz2, 
    at::Tensor gradxyz1, 
    at::Tensor gradxyz2, 
    at::Tensor graddist1, 
    at::Tensor graddist2, 
    at::Tensor idx1, 
    at::Tensor idx2) 
{
    const int b = xyz1.size(0);
    const int n = xyz1.size(1);
    const int m = xyz2.size(1);

    const float* xyz1_data = xyz1.data<float>();
    const float* xyz2_data = xyz2.data<float>();
    float* gradxyz1_data = gradxyz1.data<float>();
    float* gradxyz2_data = gradxyz2.data<float>();
    float* graddist1_data = graddist1.data<float>();
    float* graddist2_data = graddist2.data<float>();
    const int* idx1_data = idx1.data<int>();
    const int* idx2_data = idx2.data<int>();

    for (int i = 0; i < b*n*3; i++)
        gradxyz1_data[i] = 0;
    for (int i = 0; i < b*m*3; i++)
        gradxyz2_data[i] = 0;
    for (int i = 0;i < b; i++) {
        for (int j = 0; j < n; j++) {
            const float x1 = xyz1_data[(i*n+j)*3+0];
            const float y1 = xyz1_data[(i*n+j)*3+1];
            const float z1 = xyz1_data[(i*n+j)*3+2];
            const int j2 = idx1_data[i*n+j];

            const float x2 = xyz2_data[(i*m+j2)*3+0];
            const float y2 = xyz2_data[(i*m+j2)*3+1];
            const float z2 = xyz2_data[(i*m+j2)*3+2];
            const float g = graddist1_data[i*n+j]*2;

            gradxyz1_data[(i*n+j)*3+0] += g*(x1-x2);
            gradxyz1_data[(i*n+j)*3+1] += g*(y1-y2);
            gradxyz1_data[(i*n+j)*3+2] += g*(z1-z2);
            gradxyz2_data[(i*m+j2)*3+0] -= (g*(x1-x2));
            gradxyz2_data[(i*m+j2)*3+1] -= (g*(y1-y2));
            gradxyz2_data[(i*m+j2)*3+2] -= (g*(z1-z2));
        }
        for (int j = 0; j < m; j++) {
            const float x1 = xyz2_data[(i*m+j)*3+0];
            const float y1 = xyz2_data[(i*m+j)*3+1];
            const float z1 = xyz2_data[(i*m+j)*3+2];
            const int j2 = idx2_data[i*m+j];
            const float x2 = xyz1_data[(i*n+j2)*3+0];
            const float y2 = xyz1_data[(i*n+j2)*3+1];
            const float z2 = xyz1_data[(i*n+j2)*3+2];
            const float g = graddist2_data[i*m+j]*2;
            gradxyz2_data[(i*m+j)*3+0] += g*(x1-x2);
            gradxyz2_data[(i*m+j)*3+1] += g*(y1-y2);
            gradxyz2_data[(i*m+j)*3+2] += g*(z1-z2);
            gradxyz1_data[(i*n+j2)*3+0] -= (g*(x1-x2));
            gradxyz1_data[(i*n+j2)*3+1] -= (g*(y1-y2));
            gradxyz1_data[(i*n+j2)*3+2] -= (g*(z1-z2));
        }
    }
}

at::Tensor  Varifold_Kernel_forward(
    const at::Tensor xyz1, 
    const at::Tensor xyz2,
	const at::Tensor nor1,
	const at::Tensor nor2){

	// Allocate necessary data structures
	at::Tensor match_x = at::zeros({xyz1.size(0), xyz1.size(1), xyz1.size(1)}, 
		xyz1.options());
	at::Tensor match_y = at::zeros({xyz1.size(0), xyz2.size(1), xyz2.size(1)}, 
		xyz1.options());
	at::Tensor match_xy = at::zeros({xyz1.size(0), xyz1.size(1), xyz2.size(1)}, 
		xyz1.options());


	varifold_kernel_cpu(xyz1.size(0), 
    xyz1.size(1), 
    xyz1.size(1),
    xyz1.data<float>(),
    xyz1.data<float>(),
	nor1.data<float>(),
	nor1.data<float>(),
	match_x.data<float>());

	varifold_kernel_cpu(xyz1.size(0), 
    xyz2.size(1), 
    xyz2.size(1),
    xyz2.data<float>(),
    xyz2.data<float>(),
	nor2.data<float>(),
	nor2.data<float>(),
	match_y.data<float>());

	varifold_kernel_cpu(xyz1.size(0), 
    xyz1.size(1), 
    xyz2.size(1),
    xyz1.data<float>(),
    xyz2.data<float>(),
	nor1.data<float>(),
	nor2.data<float>(),
	match_xy.data<float>());

	auto match = match_x.sum() + match_y.sum() - 2.0 * match_xy.sum();
    // return output
	return match;
}

std::vector<at::Tensor>  Varifold_Kernel_backward(
    const at::Tensor xyz1, 
    const at::Tensor xyz2,
    const at::Tensor nor1,
	const at::Tensor nor2){

	// Allocate necessary data structures
	at::Tensor grad_x = at::zeros_like(xyz1);
	at::Tensor grad_xy = at::zeros_like(xyz1);
	at::Tensor grad_y = at::zeros_like(xyz2);
	at::Tensor grad_yx = at::zeros_like(xyz2);

    varifold_grad_cpu(xyz1.size(0), 
    xyz1.size(1), 
    xyz2.size(1), 
    xyz1.data<float>(),
    xyz2.data<float>(),
	nor1.data<float>(),
	nor2.data<float>(),
    grad_xy.data<float>(), 
    grad_yx.data<float>());    

    varifold_grad_cpu2(xyz1.size(0), 
    xyz1.size(1), 
    xyz1.size(1), 
    xyz1.data<float>(),
    xyz1.data<float>(),
	nor1.data<float>(),
	nor1.data<float>(),
    grad_x.data<float>());  

    varifold_grad_cpu2(xyz1.size(0), 
    xyz2.size(1), 
    xyz2.size(1), 
    xyz2.data<float>(),
    xyz2.data<float>(),
	nor2.data<float>(),
	nor2.data<float>(),
    grad_y.data<float>());

	// get output
	auto gradxyz1 = 4*(grad_x - grad_xy);
	auto gradxyz2 = 4*(grad_y - grad_yx);

    // return gradients
    return {gradxyz1, gradxyz2};

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batch_svd_forward", &batch_svd_forward,"cusolver based batch svd implementation");
    m.def("batch_svd_backward", &batch_svd_backward,"batch svd backward");
    m.def("cd_forward", &chamfer_distance_forward, "Chamfer Distance forward");
    m.def("cd_forward_cuda", &chamfer_distance_forward_cuda, "ChamferDistance forward (CUDA)");
    m.def("cd_backward", &chamfer_distance_backward, "Chamfer Distance backward");
    m.def("cd_backward_cuda", &chamfer_distance_backward_cuda, "ChamferDistance backward (CUDA)");
    m.def("emd_distance_forward", &emd_distance_forward, "Wasserstein (Earth Mover's) Distance forward");
    m.def("emd_distance_forward_cuda", &emd_distance_forward_cuda, "Wasserstein (Earth Mover's) Distance forward (CUDA)");
    m.def("emd_distance_backward", &emd_distance_backward, "Wasserstein (Earth Mover's) Distance backward");
    m.def("emd_distance_backward_cuda", &emd_distance_backward_cuda, "Wasserstein (Earth Mover's) Distance backward (CUDA)");
    m.def("varifold_kernel_forward", &Varifold_Kernel_forward, "Varifold Kernel forward cpu");
    m.def("varifold_kernel_forward_cuda", &Varifold_Kernel_forward_cuda, "Kernel forward cuda");
	m.def("varifold_kernel_backward", &Varifold_Kernel_backward, "Varifold Kernel backward cpu");
    m.def("varifold_kernel_backward_cuda", &Varifold_Kernel_backward_cuda, "Varifold Kernel backward cuda");
}
