import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

def read_protein_sequences_from_fasta(file_path):
    sequences = []
    sequence = ''
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if sequence:  # 如果已经读取了序列，先保存它
                    sequences.append(sequence)
                    sequence = ''  # 重置序列为新的序列
            else:
                sequence += line
        if sequence:  # 确保最后一个序列也被添加
            sequences.append(sequence)
    return sequences


test_seq = read_protein_sequences_from_fasta('Protein.txt')


tokenizer = AutoTokenizer.from_pretrained("ESM_Pre_model", trust_remote_code=True)
model = AutoModel.from_pretrained("ESM_Pre_model", trust_remote_code=True)

def extract_features_from_protein_sequences(sequences):
    features = []
    for seq in sequences:
        inputs = tokenizer(seq, return_tensors='pt')["input_ids"]
        with torch.no_grad():  
            hidden_states = model(inputs)[0]  # [1, sequence_length, 768]
        embedding_mean = torch.mean(hidden_states[0], dim=0)
        features.append(embedding_mean)

    features_tensor = torch.stack(features)
    return features_tensor


test_features = extract_features_from_protein_sequences(test_seq)


test_features_np = test_features.numpy()
print(test_features_np.shape)
np.savetxt('protein_bert.txt', test_features_np)
