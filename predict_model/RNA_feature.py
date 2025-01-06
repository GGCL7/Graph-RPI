
import torch


import iFeatureOmegaCLI
import pandas as pd
def generate_features_rna(input_txt_path):
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


	DPCP2 = iFeatureOmegaCLI.iRNA(input_txt_path)
	DPCP2.get_descriptor("DPCP")

	PseKNC = iFeatureOmegaCLI.iRNA(input_txt_path)
	PseKNC.get_descriptor("PseKNC")

	PseDNC = iFeatureOmegaCLI.iRNA(input_txt_path)
	PseDNC.get_descriptor("PseDNC")

	CKSNAP = iFeatureOmegaCLI.iRNA(input_txt_path)
	CKSNAP.get_descriptor("CKSNAP type 1")




	Kmer.encodings = Kmer.encodings.reset_index(drop=True)
	CKSNAP.encodings = CKSNAP.encodings.reset_index(drop=True)
	DPCP.encodings = DPCP.encodings.reset_index(drop=True)
	PseDNC.encodings = PseDNC.encodings.reset_index(drop=True)
	PseKNC.encodings = PseKNC.encodings.reset_index(drop=True)
	PCPseDNC.encodings = PCPseDNC.encodings.reset_index(drop=True)
	NAC.encodings = NAC.encodings.reset_index(drop=True)

	print(NAC.encodings.shape, Kmer.encodings.shape, DPCP.encodings.shape, PseDNC.encodings.shape,
		  PCPseDNC.encodings.shape, CKSNAP.encodings.shape)

	result = pd.concat([NAC.encodings, Kmer.encodings, DPCP.encodings, PseDNC.encodings, PCPseDNC.encodings, CKSNAP.encodings], axis=1)

	result.index = DPCP2.encodings.index



	cols = result.columns.tolist()
	result = result[cols]
	result = torch.tensor(result.iloc[:, 0:].values, dtype=torch.float32)
	result = result.numpy()

	return result

