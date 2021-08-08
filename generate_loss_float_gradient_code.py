import numpy as np


cube4 = np.full([4,4,4,3],-1,np.int32) # x, y, z, [x,y,z]

cube4[0,0,0] = [-1,-1,-1]
cube4[1,0,0] = [24,-1,-1]
cube4[2,0,0] = [24,-1,-1]
cube4[3,0,0] = [-1,-1,-1]
cube4[0,1,0] = [-1,25,-1]
cube4[1,1,0] = [27,28,-1]
cube4[2,1,0] = [29,30,-1]
cube4[3,1,0] = [-1,25,-1]
cube4[0,2,0] = [-1,25,-1]
cube4[1,2,0] = [33,34,-1]
cube4[2,2,0] = [31,32,-1]
cube4[3,2,0] = [-1,25,-1]
cube4[0,3,0] = [-1,-1,-1]
cube4[1,3,0] = [24,-1,-1]
cube4[2,3,0] = [24,-1,-1]
cube4[3,3,0] = [-1,-1,-1]

cube4[0,0,1] = [-1,-1,26]
cube4[1,0,1] = [43,-1,44]
cube4[2,0,1] = [45,-1,46]
cube4[3,0,1] = [-1,-1,26]
cube4[0,1,1] = [-1,35,36]
cube4[1,1,1] = [0,1,2]
cube4[2,1,1] = [3,4,5]
cube4[3,1,1] = [-1,35,36]
cube4[0,2,1] = [-1,37,38]
cube4[1,2,1] = [9,10,11]
cube4[2,2,1] = [6,7,8]
cube4[3,2,1] = [-1,37,38]
cube4[0,3,1] = [-1,-1,26]
cube4[1,3,1] = [43,-1,44]
cube4[2,3,1] = [45,-1,46]
cube4[3,3,1] = [-1,-1,26]

cube4[0,0,2] = [-1,-1,26]
cube4[1,0,2] = [49,-1,50]
cube4[2,0,2] = [47,-1,48]
cube4[3,0,2] = [-1,-1,26]
cube4[0,1,2] = [-1,41,42]
cube4[1,1,2] = [12,13,14]
cube4[2,1,2] = [15,16,17]
cube4[3,1,2] = [-1,41,42]
cube4[0,2,2] = [-1,39,40]
cube4[1,2,2] = [21,22,23]
cube4[2,2,2] = [18,19,20]
cube4[3,2,2] = [-1,39,40]
cube4[0,3,2] = [-1,-1,26]
cube4[1,3,2] = [49,-1,50]
cube4[2,3,2] = [47,-1,48]
cube4[3,3,2] = [-1,-1,26]

cube4[0,0,3] = [-1,-1,-1]
cube4[1,0,3] = [24,-1,-1]
cube4[2,0,3] = [24,-1,-1]
cube4[3,0,3] = [-1,-1,-1]
cube4[0,1,3] = [-1,25,-1]
cube4[1,1,3] = [27,28,-1]
cube4[2,1,3] = [29,30,-1]
cube4[3,1,3] = [-1,25,-1]
cube4[0,2,3] = [-1,25,-1]
cube4[1,2,3] = [33,34,-1]
cube4[2,2,3] = [31,32,-1]
cube4[3,2,3] = [-1,25,-1]
cube4[0,3,3] = [-1,-1,-1]
cube4[1,3,3] = [24,-1,-1]
cube4[2,3,3] = [24,-1,-1]
cube4[3,3,3] = [-1,-1,-1]

#print(np.transpose(cube4[1,:,:,0])) #X_slices
#print(np.transpose(cube4[2,:,:,0]))
#print(np.transpose(cube4[:,1,:,1])) #Y_slices
#print(np.transpose(cube4[:,2,:,1]))
#print(np.transpose(cube4[:,:,1,2])) #Z_slices
#print(np.transpose(cube4[:,:,2,2]))



def get_slice(i,j,k,dim):
	idx = cube4[i,j,k,dim]
	if idx<0:
		if dim==0:
			if i==0:
				return 0
			if i==3:
				return 1
		if dim==1:
			if j==0:
				return 0
			if j==3:
				return 1
		if dim==2:
			if k==0:
				return 0
			if k==3:
				return 1
	else:
		sss = "[:,"+str(idx)+","
		if i==3:
			sss += "1:,"
		else:
			sss += ":-1,"
		if j==3:
			sss += "1:,"
		else:
			sss += ":-1,"
		if k==3:
			sss += "1:]"
		else:
			sss += ":-1]"
		return sss



out_str = ""
counter = 0



#X_slices
for i in [1,2]:
	for k in range(3):
		for j in range(3):
			v00_idx_str = get_slice(i,j,k,0)
			v10_idx_str = get_slice(i,j,k+1,0)
			v11_idx_str = get_slice(i,j+1,k+1,0)
			v01_idx_str = get_slice(i,j+1,k,0)
			
			counter += 1
			out_str += "torch.sum( ( (pred_output_float"+v00_idx_str+"-pred_output_float"+v10_idx_str+")**2 )*torch.min(gt_output_float_mask"+v00_idx_str+",gt_output_float_mask"+v10_idx_str+" )*( torch.abs(gt_output_float"+v00_idx_str+"-gt_output_float"+v10_idx_str+")<2e-4 ).float() ) + "
			counter += 1
			out_str += "torch.sum( ( (pred_output_float"+v00_idx_str+"-pred_output_float"+v01_idx_str+")**2 )*torch.min(gt_output_float_mask"+v00_idx_str+",gt_output_float_mask"+v01_idx_str+" )*( torch.abs(gt_output_float"+v00_idx_str+"-gt_output_float"+v01_idx_str+")<2e-4 ).float() ) + "

#Y_slices
for j in [1,2]:
	for k in range(3):
		for i in range(3):
			v00_idx_str = get_slice(i,j,k,1)
			v10_idx_str = get_slice(i,j,k+1,1)
			v11_idx_str = get_slice(i+1,j,k+1,1)
			v01_idx_str = get_slice(i+1,j,k,1)
			
			counter += 1
			out_str += "torch.sum( ( (pred_output_float"+v00_idx_str+"-pred_output_float"+v10_idx_str+")**2 )*torch.min(gt_output_float_mask"+v00_idx_str+",gt_output_float_mask"+v10_idx_str+" )*( torch.abs(gt_output_float"+v00_idx_str+"-gt_output_float"+v10_idx_str+")<2e-4 ).float() ) + "
			counter += 1
			out_str += "torch.sum( ( (pred_output_float"+v00_idx_str+"-pred_output_float"+v01_idx_str+")**2 )*torch.min(gt_output_float_mask"+v00_idx_str+",gt_output_float_mask"+v01_idx_str+" )*( torch.abs(gt_output_float"+v00_idx_str+"-gt_output_float"+v01_idx_str+")<2e-4 ).float() ) + "

#Z_slices
for k in [1,2]:
	for j in range(3):
		for i in range(3):
			v00_idx_str = get_slice(i,j,k,2)
			v10_idx_str = get_slice(i,j+1,k,2)
			v11_idx_str = get_slice(i+1,j+1,k,2)
			v01_idx_str = get_slice(i+1,j,k,2)
			
			counter += 1
			out_str += "torch.sum( ( (pred_output_float"+v00_idx_str+"-pred_output_float"+v10_idx_str+")**2 )*torch.min(gt_output_float_mask"+v00_idx_str+",gt_output_float_mask"+v10_idx_str+" )*( torch.abs(gt_output_float"+v00_idx_str+"-gt_output_float"+v10_idx_str+")<2e-4 ).float() ) + "
			counter += 1
			out_str += "torch.sum( ( (pred_output_float"+v00_idx_str+"-pred_output_float"+v01_idx_str+")**2 )*torch.min(gt_output_float_mask"+v00_idx_str+",gt_output_float_mask"+v01_idx_str+" )*( torch.abs(gt_output_float"+v00_idx_str+"-gt_output_float"+v01_idx_str+")<2e-4 ).float() ) + "




fout = open("loss_float_gradient_code.py",'w')
fout.write("import torch\ndef get_loss_float_gradient(pred_output_float,gt_output_float,gt_output_float_mask):\n    return ")
fout.write(out_str[:-3])
fout.close()
print(counter)

