import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from mask import Mask
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import Linear, GINConv
from torch_geometric.utils import add_self_loops, negative_sampling, degree, to_undirected
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor
from utils import calculate_metrics
from utils import set_seed
from model import *
SEED = 2023
set_seed(SEED)
def parse_ids(file_path):
    ids = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('>'):
                ids.append(line[1:].strip())
    return ids


def parse_pairs(file_path):
    positive_pairs = []
    negative_pairs = []
    with open(file_path, 'r') as file:
        for line in file:
            protein_id, rna_id, label = line.strip().split('\t')
            if int(label) == 1:
                positive_pairs.append((protein_id, rna_id))
            elif int(label) == 0:
                negative_pairs.append((protein_id, rna_id))
    return positive_pairs, negative_pairs

protein_ids = parse_ids('Protein.txt')
rna_ids = parse_ids('RNA.txt')
positive_pairs, negative_pairs = parse_pairs('Pairs.txt')


protein_bert_features = np.loadtxt('protein_bert.txt')
rna_features = np.loadtxt('RNA_feature.txt')
protein_new_features = np.loadtxt('protein_feature.txt')

RNA = rna_features
protein = np.hstack((protein_bert_features, protein_new_features))

print(RNA.shape)
print(protein.shape)

output_dim = 909


rna_emb = []
for rna in range(len(RNA)):
    rna_emb.append(RNA[rna].tolist())
rna_emb = [lst + [0] * (output_dim - len(rna_emb[0])) for lst in rna_emb]
rna_emb = torch.Tensor(rna_emb)


protein_emb = []
for prot in range(len(protein)):
    protein_emb.append(protein[prot].tolist())
protein_emb = [lst + [0] * (output_dim - len(protein_emb[0])) for lst in protein_emb]
protein_emb = torch.Tensor(protein_emb)


feature = torch.cat([rna_emb, protein_emb])


relation_matrix = np.zeros((len(rna_ids), len(protein_ids)), dtype=int)
protein_id_to_index = {protein_id: index for index, protein_id in enumerate(protein_ids)}
rna_id_to_index = {rna_id: index for index, rna_id in enumerate(rna_ids)}


for protein_id, rna_id in positive_pairs:
    if protein_id in protein_id_to_index and rna_id in rna_id_to_index:
        protein_index = protein_id_to_index[protein_id]
        rna_index = rna_id_to_index[rna_id]
        relation_matrix[rna_index, protein_index] = 1


pos_edge_index = []
for rna_index in range(len(rna_ids)):
    for protein_index in range(len(protein_ids)):
        if relation_matrix[rna_index, protein_index] == 1:
            pos_edge_index.append([rna_index, protein_index + len(rna_ids)])
pos_edge_index = torch.LongTensor(pos_edge_index).t()

data = Data(x=feature, edge_index=pos_edge_index)


train_data, _, test_data = T.RandomLinkSplit(num_val=0, num_test=0.2,
                                             is_undirected=True, split_labels=True,
                                             add_negative_train_samples=True)(data)




splits = dict(train=train_data, test=test_data)


encoder = GNNEncoder(in_channels=output_dim, hidden_channels=64, out_channels=128, heads=8)
edge_decoder = EdgeDecoder(in_channels=128, hidden_channels=64, out_channels=1)
degree_decoder = DegreeDecoder(in_channels=128, hidden_channels=64, out_channels=1)
mask = Mask(p=0.4)
model = RPI(encoder, edge_decoder, degree_decoder, mask)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)


best_acc = 0
best_model_state = None

for epoch in range(5000):
    model.train()
    model.train_epoch(splits['train'], optimizer, alpha=0.4)

    model.eval()
    test_data = splits['test']
    z = model.encoder(test_data.x, test_data.edge_index)
    test_auc, test_ap, acc, sen, pre, spe, F1, mcc = model.test(
        z, test_data.pos_edge_label_index, test_data.neg_edge_label_index
    )

    if acc > best_acc:
        best_acc = acc
        best_model_state = model.state_dict()
        torch.save(best_model_state, 'best_model_RPI488.pth')

    print(f'Epoch: {epoch + 1:03d}, AUC: {test_auc:.6f}, AP: {test_ap:.6f}, ACC: {acc:.6f}, SEN: {sen:.6f}, PRE: {pre:.6f}, SPE: {spe:.6f}, F1: {F1:.6f}, MCC: {mcc:.6f}')


print(test_data.pos_edge_label_index)
print(test_data.pos_edge_label_index.shape)


print(test_data.neg_edge_label_index)
print(test_data.neg_edge_label_index.shape)

#
# model.load_state_dict(torch.load('best_model_RPI488.pth'))
# model.eval()
# # z = model.encoder(test_data.x, test_data.edge_index)
# z = model.encoder(train_data.x, train_data.edge_index)
# test_auc, test_ap, acc, sen, pre, spe, F1, mcc = model.test(z, test_data.pos_edge_label_index,
#                                                             test_data.neg_edge_label_index)
# results = {'AUC': "{:.6f}".format(test_auc),
#            'AP': "{:.6f}".format(test_ap),
#            "ACC": "{:.6f}".format(acc),
#            "SEN": "{:.6f}".format(sen),
#            "PRE": "{:.6f}".format(pre),
#            "SPE": "{:.6f}".format(spe),
#            "F1": "{:.6f}".format(F1),
#            "MCC": "{:.6f}".format(mcc)}
# print('Best model results:', results)
