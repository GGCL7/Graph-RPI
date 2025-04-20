# Welcome to Graph-RPI: Predicting RNA-protein interactions via graph autoencoder and self-supervised learning strategies
RNA-protein interactions (RPIs) are essential for many biological functions and are associated with various diseases. Traditional methods for detecting RPIs are labor-intensive and costly, necessitating efficient computational methods. We proposed a novel RPI prediction framework based on graph neural networks (GNNs) that addressed key limitations of existing methods, such as inadequate feature integration and negative samples construction. Compared with other state-of-the-art RPI prediction methods, our approach demonstrates greater robustness and stability in RPI prediction.

![The workflow of this study](https://github.com/GGCL7/Graph-RPI/blob/main/workflow.png)


# Dataset for this study
We provided our dataset and you can find them [Datasets](https://github.com/GGCL7/Graph-RPI/tree/main/Data)


## 🔧 Installation instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourname/Graph-RPI.git
cd Graph-RPI
```
2. **Set up the Python environment**
```bash
conda create -n graphrpi python=3.10
conda activate graphrpi
pip install -r requirements.txt
```
3. **ESM-2 Language model embeddings**
```bash
https://huggingface.co/facebook/esm2_t6_8M_UR50D
```

# Predicting RNA-protein interactions

## Single pair prediction

To predict whether a given RNA-protein pair interacts, use the following command:

```bash
python Single_pair_prediction.py -i ./Single/Protein.fasta ./Single/RNA.fasta -o record.json
```

## Multiple pairs prediction
To predict interactions for multiple RNA-protein pairs, use the following command:

```bash
python Multiple_pairs_prediction.py \                                                       
    -p "./Multiple/interaction.txt" \
    -prot "./Multiple/Protein.txt" \
    -rna "./Multiple/RNA.txt" \
    -o "record.json"
```
