import torch
import torch.nn as nn

import torch_geometric
from torch_geometric.nn import knn_graph
from torch_geometric.utils import scatter_ 

import _shape_ext._metric as metric


class BatchSVDFunction(torch.autograd.Function):

    @staticmethod
    def forward(self, x):
        U0, S, V0 = metric.batch_svd_forward(x, True, 1e-7, 100)
        k = S.size(1)
        U = U0[:, :, :k]
        V = V0[:, :, :k]
        self.save_for_backward(x, U, S, V)

        return U, S, V

    @staticmethod
    def backward(self, grad_u, grad_s, grad_v):
        x, U, S, V = self.saved_variables

        grad_out = metric.batch_svd_backward(
            [grad_u, grad_s, grad_v],
            x, True, True, U, S, V
        )

        return grad_out


def batch_svd(x):
    """
    input:
        x --- shape of [B, M, N]
    return:
        U, S, V = batch_svd(x) where x = USV^T
    """
    if not x.is_cuda:
        raise Exception('batch SVD only works with CUDA ') 
    B,M,N =x.size()
    if M >= 33 or N >=33:
        raise Exception('cuBLAS only support batch SVD up to 32 batches')
        
    return BatchSVDFunction.apply(x)

def get_normal(inputs,batch_size,num_points,k =10):
    x = inputs.reshape(-1,3) # Mark: PyTorch_Geometric uses a large disconnected sparse graph 
    batch = torch.arange(batch_size).repeat_interleave(num_points).cuda()
    edge_index = knn_graph(x, k, batch=batch, loop=True)
    row, col = edge_index
    x = x.unsqueeze(-1) if x.dim() == 1 else x
    
    # compute centroids
    knn_row = x.index_select(0,row) # nearest neighbor coordinates
    knn_col = x.index_select(0,col) # reference coordinates
    mean_v = scatter_('mean', knn_row, col, dim_size=x.size(0)) # geometric mean
    out = knn_row - mean_v.index_select(0,col)

    # reshape to B X N X k X 3
    out = out.reshape(batch_size, num_points, k, 3)
    
    # Covariance computation
    Cmat = torch.sum(torch.matmul(out.unsqueeze(-1), out[:,:,:,None,:]),2)/k

    # get SVD (size of f must be less than 32)
    Cmat = Cmat.reshape(batch_size*num_points,3,3)
    [U,_,_] = batch_svd(Cmat)
    nor = U[:,:,2] # normal
    nor = nor.reshape(batch_size,num_points,3)
    return nor

class VarifoldKernelFunctionTorch(torch.autograd.Function):
    @staticmethod
    def forward(self,xyz1,xyz2,nor1=None,nor2=None):
        B, N1, f1 = xyz1.size()
        B, N2, f2 = xyz2.size()

        # get normals if normals are not given
        if nor1 is None:
            with torch.no_grad():
                nor1 = get_normal(xyz1,B,N1)
        if nor2 is None:
            with torch.no_grad():
                nor2 = get_normal(xyz1,B,N2)

        # transport metric between two 3D discretisation
        Sx  = torch.pow(xyz1[:,:,0].repeat_interleave(N1).reshape(B,N1,N1) - xyz1[:,:,0].repeat(1,N1).reshape(B,N1,N1),2)\
            + torch.pow(xyz1[:,:,1].repeat_interleave(N1).reshape(B,N1,N1) - xyz1[:,:,1].repeat(1,N1).reshape(B,N1,N1),2)\
            + torch.pow(xyz1[:,:,2].repeat_interleave(N1).reshape(B,N1,N1) - xyz1[:,:,2].repeat(1,N1).reshape(B,N1,N1),2) 

        Sxy = torch.pow(xyz1[:,:,0].repeat_interleave(N2).reshape(B,N1,N2) - xyz2[:,:,0].repeat(1,N1).reshape(B,N1,N2),2)\
            + torch.pow(xyz1[:,:,1].repeat_interleave(N2).reshape(B,N1,N2) - xyz2[:,:,1].repeat(1,N1).reshape(B,N1,N2),2)\
            + torch.pow(xyz1[:,:,2].repeat_interleave(N2).reshape(B,N1,N2) - xyz2[:,:,2].repeat(1,N1).reshape(B,N1,N2),2)

        Sy  = torch.pow(xyz2[:,:,0].repeat_interleave(N1).reshape(B,N1,N1) - xyz2[:,:,0].repeat(1,N1).reshape(B,N1,N1),2) \
            + torch.pow(xyz2[:,:,1].repeat_interleave(N1).reshape(B,N1,N1) - xyz2[:,:,1].repeat(1,N1).reshape(B,N1,N1),2) \
            + torch.pow(xyz2[:,:,2].repeat_interleave(N1).reshape(B,N1,N1) - xyz2[:,:,2].repeat(1,N1).reshape(B,N1,N1),2)

        Dx  = nor1[:,:,0].repeat_interleave(N1).reshape(B,N1,N1) * nor1[:,:,0].repeat(1,N1).reshape(B,N1,N1)\
            + nor1[:,:,1].repeat_interleave(N1).reshape(B,N1,N1) * nor1[:,:,1].repeat(1,N1).reshape(B,N1,N1)\
            + nor1[:,:,2].repeat_interleave(N1).reshape(B,N1,N1) * nor1[:,:,2].repeat(1,N1).reshape(B,N1,N1) 

        Dxy = nor1[:,:,0].repeat_interleave(N2).reshape(B,N1,N2) * nor2[:,:,0].repeat(1,N1).reshape(B,N1,N2)\
            + nor1[:,:,1].repeat_interleave(N2).reshape(B,N1,N2) * nor2[:,:,1].repeat(1,N1).reshape(B,N1,N2)\
            + nor1[:,:,2].repeat_interleave(N2).reshape(B,N1,N2) * nor2[:,:,2].repeat(1,N1).reshape(B,N1,N2)

        Dy  = nor2[:,:,0].repeat_interleave(N1).reshape(B,N1,N1) * nor2[:,:,0].repeat(1,N1).reshape(B,N1,N1)\
            + nor2[:,:,1].repeat_interleave(N1).reshape(B,N1,N1) * nor2[:,:,1].repeat(1,N1).reshape(B,N1,N1)\
            + nor2[:,:,2].repeat_interleave(N1).reshape(B,N1,N1) * nor2[:,:,2].repeat(1,N1).reshape(B,N1,N1)          

        Ax  = torch.exp(-1.0*Sx)
        Axy = torch.exp(-1.0*Sxy)
        Ay  = torch.exp(-1.0*Sy)

        Dx  = torch.pow(Dx ,2)
        Dxy = torch.pow(Dxy,2)
        Dy  = torch.pow(Dy ,2)

        varifold = torch.sum(Ax*Dx) + torch.sum(Ay*Dy) - 2.0*torch.sum(Axy*Dxy)

        self.save_for_backward(xyz1, xyz2, nor1, nor2)
        return varifold

    @staticmethod
    def backward(self, grad_output):
        xyz1, xyz2, nor1, nor2 = self.saved_tensors

        B, N1, f1 = xyz1.size()
        B, N2, f2 = xyz2.size()

        grad_xyz1 = torch.zeros(xyz1.size())
        grad_xyz2 = torch.zeros(xyz2.size())

        if xyz1.is_cuda:
            grad_xyz1 = grad_xyz1.cuda()
            grad_xyz2 = grad_xyz2.cuda()

        # transport metric between two 3D discretisation
        Sx  = torch.pow(xyz1[:,:,0].repeat_interleave(N1).reshape(B,N1,N1) - xyz1[:,:,0].repeat(1,N1).reshape(B,N1,N1),2)\
            + torch.pow(xyz1[:,:,1].repeat_interleave(N1).reshape(B,N1,N1) - xyz1[:,:,1].repeat(1,N1).reshape(B,N1,N1),2)\
            + torch.pow(xyz1[:,:,2].repeat_interleave(N1).reshape(B,N1,N1) - xyz1[:,:,2].repeat(1,N1).reshape(B,N1,N1),2) 

        Sxy = torch.pow(xyz1[:,:,0].repeat_interleave(N2).reshape(B,N1,N2) - xyz2[:,:,0].repeat(1,N1).reshape(B,N1,N2),2)\
            + torch.pow(xyz1[:,:,1].repeat_interleave(N2).reshape(B,N1,N2) - xyz2[:,:,1].repeat(1,N1).reshape(B,N1,N2),2)\
            + torch.pow(xyz1[:,:,2].repeat_interleave(N2).reshape(B,N1,N2) - xyz2[:,:,2].repeat(1,N1).reshape(B,N1,N2),2)

        Sy  = torch.pow(xyz2[:,:,0].repeat_interleave(N2).reshape(B,N2,N2) - xyz2[:,:,0].repeat(1,N2).reshape(B,N2,N2),2) \
            + torch.pow(xyz2[:,:,1].repeat_interleave(N2).reshape(B,N2,N2) - xyz2[:,:,1].repeat(1,N2).reshape(B,N2,N2),2) \
            + torch.pow(xyz2[:,:,2].repeat_interleave(N2).reshape(B,N2,N2) - xyz2[:,:,2].repeat(1,N2).reshape(B,N2,N2),2)

        Dx  = nor1[:,:,0].repeat_interleave(N1).reshape(B,N1,N1) * nor1[:,:,0].repeat(1,N1).reshape(B,N1,N1)\
            + nor1[:,:,1].repeat_interleave(N1).reshape(B,N1,N1) * nor1[:,:,1].repeat(1,N1).reshape(B,N1,N1)\
            + nor1[:,:,2].repeat_interleave(N1).reshape(B,N1,N1) * nor1[:,:,2].repeat(1,N1).reshape(B,N1,N1) 

        Dxy = nor1[:,:,0].repeat_interleave(N2).reshape(B,N1,N2) * nor2[:,:,0].repeat(1,N1).reshape(B,N1,N2)\
            + nor1[:,:,1].repeat_interleave(N2).reshape(B,N1,N2) * nor2[:,:,1].repeat(1,N1).reshape(B,N1,N2)\
            + nor1[:,:,2].repeat_interleave(N2).reshape(B,N1,N2) * nor2[:,:,2].repeat(1,N1).reshape(B,N1,N2)

        Dy  = nor2[:,:,0].repeat_interleave(N2).reshape(B,N2,N2) * nor2[:,:,0].repeat(1,N2).reshape(B,N2,N2)\
            + nor2[:,:,1].repeat_interleave(N2).reshape(B,N2,N2) * nor2[:,:,1].repeat(1,N2).reshape(B,N2,N2)\
            + nor2[:,:,2].repeat_interleave(N2).reshape(B,N2,N2) * nor2[:,:,2].repeat(1,N2).reshape(B,N2,N2)          

        Ax  = torch.exp(-1.0*Sx)
        Axy = torch.exp(-1.0*Sxy)
        Ay  = torch.exp(-1.0*Sy)

        Apx  = -1.0*torch.exp(-1.0*Sx)
        Apxy = -1.0*torch.exp(-1.0*Sxy)
        Apy  = -1.0*torch.exp(-1.0*Sy)

        Dx  = torch.pow(Dx ,2)
        Dxy = torch.pow(Dxy,2)
        Dy  = torch.pow(Dy ,2)

        grad_xyz1[:,:,0] = 4*(torch.sum(Apx * Dx * (xyz1[:,:,0].repeat_interleave(N1).reshape(B,N1,N1) - xyz1[:,:,0].repeat(1,N1).reshape(B,N1,N1)),2)\
                            - torch.sum(Apxy* Dxy* (xyz1[:,:,0].repeat_interleave(N2).reshape(B,N1,N2) - xyz2[:,:,0].repeat(1,N1).reshape(B,N1,N2)),2))
        grad_xyz1[:,:,1] = 4*(torch.sum(Apx * Dx * (xyz1[:,:,1].repeat_interleave(N1).reshape(B,N1,N1) - xyz1[:,:,1].repeat(1,N1).reshape(B,N1,N1)),2)\
                            - torch.sum(Apxy* Dxy* (xyz1[:,:,1].repeat_interleave(N2).reshape(B,N1,N2) - xyz2[:,:,1].repeat(1,N1).reshape(B,N1,N2)),2))
        grad_xyz1[:,:,2] = 4*(torch.sum(Apx * Dx * (xyz1[:,:,2].repeat_interleave(N1).reshape(B,N1,N1) - xyz1[:,:,2].repeat(1,N1).reshape(B,N1,N1)),2)\
                            - torch.sum(Apxy* Dxy* (xyz1[:,:,2].repeat_interleave(N2).reshape(B,N1,N2) - xyz2[:,:,2].repeat(1,N1).reshape(B,N1,N2)),2))


        grad_xyz2[:,:,0] = 4*(-1.0*torch.sum(Apy * Dy * (xyz2[:,:,0].repeat_interleave(N2).reshape(B,N2,N2) - xyz2[:,:,0].repeat(1,N2).reshape(B,N2,N2)),1)\
                            - -1.0*torch.sum(Apxy* Dxy* (xyz1[:,:,0].repeat_interleave(N2).reshape(B,N1,N2) - xyz2[:,:,0].repeat(1,N1).reshape(B,N1,N2)),1))
        grad_xyz2[:,:,1] = 4*(-1.0*torch.sum(Apy * Dy * (xyz2[:,:,1].repeat_interleave(N2).reshape(B,N2,N2) - xyz2[:,:,1].repeat(1,N2).reshape(B,N2,N2)),1)\
                            - -1.0*torch.sum(Apxy* Dxy* (xyz1[:,:,1].repeat_interleave(N2).reshape(B,N1,N2) - xyz2[:,:,1].repeat(1,N1).reshape(B,N1,N2)),1))
        grad_xyz2[:,:,2] = 4*(-1.0*torch.sum(Apy * Dy * (xyz2[:,:,2].repeat_interleave(N2).reshape(B,N2,N2) - xyz2[:,:,2].repeat(1,N2).reshape(B,N2,N2)),1)\
                            - -1.0*torch.sum(Apxy* Dxy* (xyz1[:,:,2].repeat_interleave(N2).reshape(B,N1,N2) - xyz2[:,:,2].repeat(1,N1).reshape(B,N1,N2)),1)) 

        return grad_xyz1, grad_xyz2

class VarifoldLossTorch(nn.Module):
    def __init__(self):
        super(VarifoldLossTorch,self).__init__()
    def forward(self, xyz1, xyz2, nor1=None, nor2=None):
        return VarifoldKernelFunctionTorch.apply(xyz1, xyz2, nor1, nor2)   

class VarifoldKernelFunction(torch.autograd.Function):
    @staticmethod
    def forward(self,xyz1,xyz2,nor1=None,nor2=None):
        B, N1, f1 = xyz1.size()
        B, N2, f2 = xyz2.size()

        # get normals if normals are not given
        if nor1 is None:
            with torch.no_grad():
                nor1 = get_normal(xyz1,B,N1)
            if xyz1.is_cuda:
                nor1 = nor1.cuda()

        if nor2 is None:
            with torch.no_grad():
                nor2 = get_normal(xyz2,B,N2)
            if xyz1.is_cuda:
                nor2 = nor2.cuda()

        if xyz1.is_cuda:
            varifold = metric.varifold_kernel_forward_cuda(xyz1,xyz2,nor1,nor2)
        else:
            varifold = metric.varifold_kernel_forward(xyz1,xyz2,nor1,nor2)
        self.save_for_backward(xyz1, xyz2, nor1, nor2)

        return varifold

    @staticmethod
    def backward(self, grad_output):
        xyz1, xyz2, nor1, nor2 = self.saved_tensors

        B, N1, f1 = xyz1.size()
        B, N2, f2 = xyz2.size()

        if xyz1.is_cuda:
            grad_xyz1, grad_xyz2 = metric.varifold_kernel_backward_cuda(xyz1,xyz2,nor1,nor2)
        else:
            grad_xyz1, grad_xyz2 = metric.varifold_kernel_backward(xyz1,xyz2,nor1,nor2)

        return grad_xyz1, grad_xyz2, None, None

class VarifoldLoss(nn.Module):
    def __init__(self):
        super(VarifoldLoss,self).__init__()
    def forward(self, xyz1, xyz2, nor1=None, nor2=None):
        return VarifoldKernelFunction.apply(xyz1, xyz2, nor1, nor2)

# Wasserstein distance function
class EarthMoverDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, xyz1, xyz2):
        if not xyz1.is_cuda:
            cost, match = metric.emd_distance_forward(xyz1, xyz2)
        else:
            cost, match = metric.emd_distance_forward_cuda(xyz1, xyz2)
        self.save_for_backward(xyz1, xyz2, match)
        return cost
        
    @staticmethod
    def backward(self, grad_output):
        xyz1, xyz2, match = self.saved_tensors
        if not xyz1.is_cuda:
            grad_xyz1, grad_xyz2 = metric.emd_distance_backward(xyz1, xyz2, match)
        else:
            grad_xyz1, grad_xyz2 = metric.emd_distance_backward_cuda(xyz1, xyz2, match)
        return grad_xyz1, grad_xyz2

# Chamfer distance function
class ChamferDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n, dtype=torch.int)
        idx2 = torch.zeros(batchsize, m, dtype=torch.int)

        if not xyz1.is_cuda:
            metric.cd_forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        else:
            dist1 = dist1.cuda()
            dist2 = dist2.cuda()
            idx1 = idx1.cuda()
            idx2 = idx2.cuda()
            metric.cd_forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)

        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2

    @staticmethod
    def backward(ctx, graddist1, graddist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors

        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        if not graddist1.is_cuda:
            metric.cd_backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        else:
            gradxyz1 = gradxyz1.cuda()
            gradxyz2 = gradxyz2.cuda()
            metric.cd_backward_cuda(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)

        return gradxyz1, gradxyz2

class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss,self).__init__()
    def forward(self, xyz1, xyz2):
        return ChamferDistanceFunction.apply(xyz1, xyz2)

class EMDLoss(nn.Module):
	'''
	Computes the (approximate) Earth Mover's Distance between two point sets (from optas's github). 
	'''
	def __init__(self):
		super(EMDLoss, self).__init__()

	def forward(self, xyz1, xyz2):
		return EarthMoverDistanceFunction.apply(xyz1, xyz2)