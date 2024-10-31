import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
import iFeatureOmegaCLI  # 确保已经安装了这个库
import re
import math

import iFeatureOmegaCLI
import pandas as pd
def generate_features(input_txt_path):
	Kmer = iFeatureOmegaCLI.iRNA(input_txt_path)
	Kmer.get_descriptor("Kmer type 1")
	Kmer.display_feature_types()


	DPCP = iFeatureOmegaCLI.iRNA(input_txt_path)
	DPCP.get_descriptor("DPCP")

	NAC = iFeatureOmegaCLI.iRNA(input_txt_path)
	NAC.get_descriptor("NAC")

	LPDF = iFeatureOmegaCLI.iRNA(input_txt_path)
	LPDF.get_descriptor("LPDF")

	PCPseDNC = iFeatureOmegaCLI.iRNA(input_txt_path)
	PCPseDNC.get_descriptor("PCPseDNC")

	# ASDC = iFeatureOmegaCLI.iRNA(input_txt_path)
	# ASDC.get_descriptor("ASDC")

	DPCP2 = iFeatureOmegaCLI.iRNA(input_txt_path)
	DPCP2.get_descriptor("DPCP")

	PseKNC = iFeatureOmegaCLI.iRNA(input_txt_path)
	PseKNC.get_descriptor("PseKNC")

	PseDNC = iFeatureOmegaCLI.iRNA(input_txt_path)
	PseDNC.get_descriptor("PseDNC")

	CKSNAP = iFeatureOmegaCLI.iRNA(input_txt_path)
	CKSNAP.get_descriptor("CKSNAP type 1")

	# dde = feature_DDE(input_txt_path)  # 确保你有这个函数的定义

	# 重置索引
	Kmer.encodings = Kmer.encodings.reset_index(drop=True)
	CKSNAP.encodings = CKSNAP.encodings.reset_index(drop=True)
	DPCP.encodings = DPCP.encodings.reset_index(drop=True)
	PseDNC.encodings = PseDNC.encodings.reset_index(drop=True)
	PseKNC.encodings = PseKNC.encodings.reset_index(drop=True)
	PCPseDNC.encodings = PCPseDNC.encodings.reset_index(drop=True)
	NAC.encodings = NAC.encodings.reset_index(drop=True)
	# print(NAC.encodings.shape, Kmer.encodings.shape, DPCP.encodings.shape, PseDNC.encodings.shape,
		  # PCPseDNC.encodings.shape, CKSNAP.encodings.shape)
	result = pd.concat([NAC.encodings, Kmer.encodings, DPCP.encodings, PseDNC.encodings, PCPseDNC.encodings, CKSNAP.encodings], axis=1)
	result.index = DPCP2.encodings.index


	# 将Label列移动到第一列
	cols = result.columns.tolist()
	result = result[cols]

	return result



inputfile = 'RNA.txt'
test_df = generate_features(inputfile)
print(test_df)
x2 = torch.tensor(test_df.iloc[:, 0:].values, dtype=torch.float32)
print(x2.shape)
x2_np = x2.numpy()
np.savetxt('RNA_feature.txt', x2_np)
dataset2 = TensorDataset(x2)