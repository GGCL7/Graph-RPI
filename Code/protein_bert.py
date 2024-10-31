import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

def read_protein_sequences_from_fasta(file_path):
    """读取FASTA格式的文件，并返回蛋白质序列的列表"""
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

# 示例：读取蛋白质序列
test_seq = read_protein_sequences_from_fasta('Protein.txt')

# 初始化ESM模型和分词器
tokenizer = AutoTokenizer.from_pretrained("ESM_Pre_model", trust_remote_code=True)
model = AutoModel.from_pretrained("ESM_Pre_model", trust_remote_code=True)

def extract_features_from_protein_sequences(sequences):
    features = []

    # 遍历所有序列，提取特征
    for seq in sequences:
        inputs = tokenizer(seq, return_tensors='pt')["input_ids"]
        with torch.no_grad():  # 不计算梯度，减少内存消耗
            hidden_states = model(inputs)[0]  # [1, sequence_length, 768]
        embedding_mean = torch.mean(hidden_states[0], dim=0)
        features.append(embedding_mean)

    features_tensor = torch.stack(features)
    return features_tensor

# 提取特征
test_features = extract_features_from_protein_sequences(test_seq)

# 将特征转换为NumPy数组
test_features_np = test_features.numpy()
print(test_features_np.shape)
# 将NumPy数组保存为文本文件
np.savetxt('protein_bert.txt', test_features_np)
