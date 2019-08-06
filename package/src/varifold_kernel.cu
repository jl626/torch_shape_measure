#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kernel_cuda(int b,
	int n,
	int m,
	const float * __restrict__ xyz1,
	const float * __restrict__ xyz2,
	const float * __restrict__ nor1,
	const float * __restrict__ nor2,	
	float * __restrict__ match){

	for (int i = blockIdx.x; i < b; i += gridDim.x){
		for (int l = threadIdx.x; l < n; l += blockDim.x){
			float x1  = xyz1[i*n*3+l*3+0];
			float y1  = xyz1[i*n*3+l*3+1];
			float z1  = xyz1[i*n*3+l*3+2];
			float nx1 = nor1[i*n*3+l*3+0];
			float ny1 = nor1[i*n*3+l*3+1];
			float nz1 = nor1[i*n*3+l*3+2];
			
			for (int k = 0; k < m; k++){
				float x2  =  x1 - xyz2[i*m*3+k*3+0];
				float y2  =  y1 - xyz2[i*m*3+k*3+1];
				float z2  =  z1 - xyz2[i*m*3+k*3+2];
				float nx2 = nx1 * nor2[i*m*3+k*3+0];
				float ny2 = ny1 * nor2[i*m*3+k*3+1];
				float nz2 = nz1 * nor2[i*m*3+k*3+2];

				match[i*m*n + k*n + l]  = fmaxf(expf(-1.0f*(x2*x2+y2*y2+z2*z2))*powf(nx2+ny2+nz2,2),1e-10f);
			}
		}
	}
}
void CudaKernelLauncher(int b,
	int n,
	int m,
	const float * xyz1,
	const float * xyz2,
	const float * nor1,
	const float * nor2,	
	float * match){
	kernel_cuda<<<32,512>>>(b,n,m,xyz1,xyz2,nor1,nor2,match);
}

__global__ void CudaKernelGrad2(int b,
	int n,
	int m,
	const float * __restrict__ xyz1,
	const float * __restrict__ xyz2,
	const float * __restrict__ nor1,
	const float * __restrict__ nor2,
	float * grad2)
{
	__shared__ float sum_grad[256*3];
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		int kbeg=m*blockIdx.y/gridDim.y;
		int kend=m*(blockIdx.y+1)/gridDim.y;
		for (int k=kbeg;k<kend;k++){
			float  x2=xyz2[(i*m+k)*3+0];
			float  y2=xyz2[(i*m+k)*3+1];
			float  z2=xyz2[(i*m+k)*3+2];
			float nx2=nor2[(i*m+k)*3+0];
			float ny2=nor2[(i*m+k)*3+1];
			float nz2=nor2[(i*m+k)*3+2];
			float subsumx=0,subsumy=0,subsumz=0;
			for (int j=threadIdx.x;j<n;j+=blockDim.x){
				float  x1= x2-xyz1[(i*n+j)*3+0];
				float  y1= y2-xyz1[(i*n+j)*3+1];
				float  z1= z2-xyz1[(i*n+j)*3+2];
				float nx1=nx2-nor1[(i*n+j)*3+0];
				float ny1=ny2-nor1[(i*n+j)*3+1];
				float nz1=nz2-nor1[(i*n+j)*3+2];
				
				float d = -1.0f * fmaxf(expf(x1*x1+y1*y1+z1*z1),1e-20f);
				float g = fmaxf(powf(nx1+ny1+nz1,2),1e-20f);
				subsumx+=x1*d*g;
				subsumy+=y1*d*g;
				subsumz+=z1*d*g;
			}
			sum_grad[threadIdx.x*3+0]=subsumx;
			sum_grad[threadIdx.x*3+1]=subsumy;
			sum_grad[threadIdx.x*3+2]=subsumz;
			for (int j=1;j<blockDim.x;j<<=1){
				__syncthreads();
				int j1=threadIdx.x;
				int j2=threadIdx.x+j;
				if ((j1&j)==0 && j2<blockDim.x){
					sum_grad[j1*3+0]+=sum_grad[j2*3+0];
					sum_grad[j1*3+1]+=sum_grad[j2*3+1];
					sum_grad[j1*3+2]+=sum_grad[j2*3+2];
				}
			}
			if (threadIdx.x==0){
				grad2[(i*m+k)*3+0]=sum_grad[0];
				grad2[(i*m+k)*3+1]=sum_grad[1];
				grad2[(i*m+k)*3+2]=sum_grad[2];
			}
			__syncthreads();
		}
	}

}

__global__ void CudaKernelGrad1(int b,
	int n,
	int m,
	const float * __restrict__ xyz1,
	const float * __restrict__ xyz2,
	const float * __restrict__ nor1,
	const float * __restrict__ nor2,
	float * grad1)
{
	for (int i = blockIdx.x; i < b; i += gridDim.x){
		for (int l = threadIdx.x; l < n; l += blockDim.x){
			float x1  = xyz1[i*n*3+l*3+0];
			float y1  = xyz1[i*n*3+l*3+1];
			float z1  = xyz1[i*n*3+l*3+2];
			float nx1 = nor1[i*n*3+l*3+0];
			float ny1 = nor1[i*n*3+l*3+1];
			float nz1 = nor1[i*n*3+l*3+2];
						
			float dx=0,dy=0,dz=0;
			for (int k = 0; k < m; k++){
				float x2  =  x1 - xyz2[i*m*3+k*3+0];
				float y2  =  y1 - xyz2[i*m*3+k*3+1];
				float z2  =  z1 - xyz2[i*m*3+k*3+2];
				float nx2 = nx1 * nor2[i*m*3+k*3+0];
				float ny2 = ny1 * nor2[i*m*3+k*3+1];
				float nz2 = nz1 * nor2[i*m*3+k*3+2];
				float d  = -1.0f * fmaxf(expf(-1.0f*(x2*x2+y2*y2+z2*z2)),1e-20f);
				float g  = fmaxf(powf(nx2+ny2+nz2,2),1e-20f);
				dx += x2 * d * g; 
				dy += y2 * d * g;
				dz += z2 * d * g;
			}
			grad1[i*n*3+l*3+0] = dx;
			grad1[i*n*3+l*3+1] = dy;
			grad1[i*n*3+l*3+2] = dz;
		}
	}
}

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
    float * grad_y)
	{
	CudaKernelGrad1<<<32,512>>>(b,n,m,xyz1,xyz2,nor1,nor2,grad_xy);
	CudaKernelGrad1<<<32,512>>>(b,n,m,xyz1,xyz1,nor1,nor1,grad_x);
	CudaKernelGrad1<<<32,512>>>(b,n,m,xyz2,xyz2,nor2,nor2,grad_y);
	CudaKernelGrad1<<<32,512>>>(b,m,n,xyz2,xyz1,nor2,nor1,grad_yx);
    
	//CudaKernelGrad2<<<dim3(32,32),256>>>(b,n,m,xyz1,xyz2,nor1,nor2,grad_yx);
    }