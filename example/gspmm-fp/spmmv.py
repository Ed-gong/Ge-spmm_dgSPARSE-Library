import torch
import scipy
import os
import numpy as np
from scipy.io import mmread
#from torch.utils.cpp_extension import load
import time
import argparse
import util



parser = argparse.ArgumentParser(description='pick argu')
parser.add_argument('--feat', type=int, default= 16, help='feature size')
parser.add_argument('--gdir', type=str, default= './saved_graph/', help='saved graph directory')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

k = args.feat 
ofile = args.gdir

#sparsePath = "../data/p2p-Gnutella31.mtx"
#sparsecsr = mmread(sparsePath).tocsc().astype('float32')
#weight = torch.from_numpy(sparsecsr.data).to(device).float()
#rowptr = torch.from_numpy(sparsecsr.indptr).to(device).int()
#colind = torch.from_numpy(sparsecsr.indices).to(device).int()

offset_file = ofile + '_csr_noeid.offset'
nebrs_file = ofile + "_csr_noeid.nebrs"
offset = np.fromfile(offset_file, dtype='int32')
nebrs  = np.fromfile(nebrs_file, dtype='int32')
rowptr = torch.from_numpy(offset).to(device)
colind = torch.from_numpy(nebrs).to(device)

"""
#idxs = torch.load('/home/ygong07/exp/GraphPy_workflow/new_script/tensor_nebr.pt')
#ptrs = torch.load('/home/ygong07/exp/GraphPy_workflow/new_script/tensor_off.pt')
idxs = torch.load('/mnt/huge_26TB/data/test2/reddit/zhai_graph/tensor_nebr.pt') 
ptrs = torch.load('/mnt/huge_26TB/data/test2/reddit/zhai_graph/tensor_off.pt')
rowptr = ptrs.to(device).int()
colind = idxs.to(device).int()
#node_feature = torch.from_numpy(np.ones((n, k))).to(device).float()
#edge_feature = torch.from_numpy(np.ones((nnz, 1))).to(device).float()
"""

n = rowptr.shape[0] - 1

print("dim = ", n, k)
node_feature = torch.ones(n, k).to(device)

a = time.time()
#ue = util.u_sub_e_sum(rowptr, colind, edge_feature, node_feature)
ue = util.copy_u_sum(rowptr, colind, node_feature)
torch.cuda.synchronize()
b = time.time()
time_our_ue = b-a
print(f"running copy_u_sum our time is: {time_our_ue:.4f}")
print("result", ue)
