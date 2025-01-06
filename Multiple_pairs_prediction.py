import argparse
import json
from transformers import AutoTokenizer, AutoModel
from predict_model.model import *
from predict_model.RNA_feature import *
from predict_model.mask import Mask
from predict_model.protein_feature import *


def parse_ids(file_path):
    ids = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('>'):
                ids.append(line[1:].strip())
    return ids

def parse_pairs(file_path):
    pairs = []
    with open(file_path, 'r') as file:
        for line in file:
            protein_id, rna_id, label = line.strip().split('\t')
            pairs.append((protein_id, rna_id))
    return pairs

def read_protein_sequences_from_fasta(file_path):

    sequences = []
    sequence = ''
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if sequence:
                    sequences.append(sequence)
                    sequence = ''
            else:
                sequence += line
        if sequence:
            sequences.append(sequence)
    return sequences


def generate_features_protein_bert(sequences, tokenizer, model):
    features = []
    for seq in sequences:
        inputs = tokenizer(seq, return_tensors='pt')["input_ids"]
        with torch.no_grad():
            hidden_states = model(inputs)[0]
        embedding_mean = torch.mean(hidden_states[0], dim=0)
        features.append(embedding_mean)
    features_np = np.vstack(features)
    print(f"Generated protein BERT features with shape: {features_np.shape}")
    return features_np


def convert_numpy_int_to_int(input_list):
    """Convert all numpy.int64 values in the input list to int."""
    for item in input_list:
        for key, value in item.items():
            if isinstance(value, np.int64):
                item[key] = int(value)
    return input_list


def parse_pairs(file_path):
    """解析对文件，获取所有蛋白质-RNA对"""
    pairs = []
    with open(file_path, 'r') as file:
        for line in file:
            protein_id, rna_id, label = line.strip().split('\t')
            pairs.append((protein_id, rna_id))
    return pairs


def main(pairs_file, protein_file, rna_file, output_file):

    protein_sequences = read_protein_sequences_from_fasta(protein_file)


    rna_features = generate_features_rna(rna_file)
    print(f"Generated RNA features with shape: {rna_features.shape}")


    tokenizer = AutoTokenizer.from_pretrained("ESM_Pre_model", trust_remote_code=True)
    model = AutoModel.from_pretrained("ESM_Pre_model", trust_remote_code=True)


    protein_bert_features = generate_features_protein_bert(protein_sequences, tokenizer, model)
    protein_new_features = generate_features_protein(protein_file)
    print(f"Generated protein new features with shape: {protein_new_features.shape}")


    output_dim = 909


    protein_ids = parse_ids(protein_file)
    rna_ids = parse_ids(rna_file)
    pairs = parse_pairs(pairs_file)
    print(f"Number of pairs to predict: {len(pairs)}")


    encoder = GNNEncoder(in_channels=output_dim, hidden_channels=64, out_channels=128, heads=8)
    edge_decoder = EdgeDecoder(in_channels=128, hidden_channels=64, out_channels=1)
    degree_decoder = DegreeDecoder(in_channels=128, hidden_channels=64, out_channels=1)
    mask = Mask(p=0.5)
    model = RPI(encoder, edge_decoder, degree_decoder, mask)
    model.load_state_dict(torch.load(
        'Model_saved/Model_RPI1807.pth'))
    model.eval()
    print("Model loaded and set to evaluation mode.")

    # 预测样本对的结果
    results = []
    positive_count = 0

    for protein_id, rna_id in pairs:
        try:
            rna_feature = rna_features[rna_ids.index(rna_id)]
            protein_feature = np.hstack((protein_bert_features[protein_ids.index(protein_id)],
                                         protein_new_features[protein_ids.index(protein_id)]))
        except ValueError as e:
            print(f"Error: {e}")
            continue

        rna_feature = np.concatenate((rna_feature, np.zeros(output_dim - len(rna_feature))))
        protein_feature = np.concatenate((protein_feature, np.zeros(output_dim - len(protein_feature))))

        rna_feature_tensor = torch.Tensor(rna_feature).unsqueeze(0)
        protein_feature_tensor = torch.Tensor(protein_feature).unsqueeze(0)

        z_rna = model.encoder(rna_feature_tensor, torch.LongTensor([[0], [0]]))
        z_protein = model.encoder(protein_feature_tensor, torch.LongTensor([[0], [0]]))

        edge_index = torch.tensor([[0], [1]])
        z = torch.cat([z_rna, z_protein], dim=0)


        logit = model.edge_decoder(z, edge_index)


        prob = torch.sigmoid(logit)


        prediction = (prob >= 0.5).float()

        if prediction.item() == 1.0:
            positive_count += 1

        results.append({
            'protein_id': protein_id,
            'rna_id': rna_id,
            'logit': logit.item(),
            'probability': prob.item(),
            'prediction': int(prediction.item())
        })

    # 输出预测结果到文件
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4, separators=(',', ': '))
        print(f"Results successfully saved to {output_file}")
    except Exception as e:
        print(f"Failed to save results to {output_file}: {e}")

    print(f"Total positive interactions predicted: {positive_count}")



parser = argparse.ArgumentParser(description='Predict Protein-RNA Interactions')
parser.add_argument('-p', '--pairs', type=str, required=True, help='File containing protein-RNA pairs')
parser.add_argument('-prot', '--protein', type=str, required=True, help='Protein sequence file')
parser.add_argument('-rna', '--rna', type=str, required=True, help='RNA sequence file')
parser.add_argument('-o', '--output', type=str, required=True, help='Output JSON file')
args = parser.parse_args()


main(args.pairs, args.protein, args.rna, args.output)
