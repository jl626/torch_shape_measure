import torch
import time

from shape_metric import EMDLoss, ChamferLoss, VarifoldLoss


'''
Test Wasserstein(EMD) Loss!
'''

dist01 = EMDLoss()

print('compute EMDLoss ALG1')
p1 = torch.rand(32,1024,3).cuda()#.double()
p2 = torch.rand(32,1024,3).cuda()#.double()

p1.requires_grad = True
p2.requires_grad = True

s = time.time()

cost  = dist01(p1, p2)
print('Wasserstein (EMD) cost from ALG1:')
print(cost)

loss1 = torch.sum(cost)
print('Wasserstein (EMD) loss from ALG1: %.05f'%loss1)
loss1.backward()

emd_time = time.time() - s
print('Time: ', emd_time)

'''
Test Chamfer Loss!
'''

print('compute ChamferLoss')
dist02 = ChamferLoss()

s2 = time.time()

cost1, cost2 = dist02(p1, p2)

print('chamfer cost1')
print(cost1)
print('chamfer cost2')
print(cost2) 

loss = (torch.mean(cost1)) + (torch.mean(cost2))
print('chamfer loss: %.05f' %loss.data.cpu().numpy())

cd_time = time.time() - s2
print('Time: ', cd_time)


'''
Test Varifold pseudo-metric Loss!
'''
print('compute VarifoldLoss')
dist03 = VarifoldLoss()

s3 = time.time()
print(loss)
cost3 = dist03(p1,p2)
print('Varifold loss: %.05f' %cost3.data.cpu().numpy())

varifold_time = time.time() - s3
print('Time: ', varifold_time)



import scipy.io as sio
d = sio.loadmat('bunny.mat')
print('test on a real 3D object - bunny vs sphere')

# points and normals are provided
pts1 = torch.unsqueeze( torch.FloatTensor(d['pts']),0)
pts2 = torch.unsqueeze(torch.FloatTensor(d['pts2']),0)
nor1 = torch.unsqueeze( torch.FloatTensor(d['normal']),0)
nor2 = torch.unsqueeze(torch.FloatTensor(d['normal2']),0)

nor1_norm = torch.unsqueeze(torch.sqrt(torch.sum(torch.pow(nor1,2),2)),-1)
nor2_norm = torch.unsqueeze(torch.sqrt(torch.sum(torch.pow(nor2,2),2)),-1)

pts1 = pts1.cuda()
pts2 = pts1.cuda()
nor1 = nor1/nor1_norm
nor1 = nor1.cuda()
nor2 = nor2/nor2_norm
nor2 = nor2.cuda()

pts1.requires_grad = True
pts2.requires_grad = True

emd  = dist01(pts1, pts2)
print('Wasserstein (EMD) cost from ALG1:')
print(emd)

loss1 = torch.sum(emd)
print('Wasserstein (EMD) loss from ALG1: %.05f'%loss1)
loss1.backward()

emd_time = time.time() - s
print('Time: ', emd_time)

'''
Test Chamfer Loss!
'''

print('compute ChamferLoss')
dist02 = ChamferLoss()

s2 = time.time()

cd1, cd2 = dist02(pts1, pts2)

print('chamfer cost1')
print(cd1)
print('chamfer cost2')
print(cd2) 

loss2 = (torch.mean(cd1)) + (torch.mean(cd2))
print('chamfer loss: %.05f' %loss2.data.cpu().numpy())
loss2.backward()

cd_time = time.time() - s2
print('Time: ', cd_time)


'''
Test Varifold pseudo-metric Loss!
'''
print('compute VarifoldLoss')
dist03 = VarifoldLoss()

s3 = time.time()

loss3 = dist03(pts1,pts2,nor1,nor2)

print('Varifold loss: %.05f' %loss3.data.cpu().numpy())
loss3.backward()

varifold_time = time.time() - s3
print('Time: ', varifold_time)
