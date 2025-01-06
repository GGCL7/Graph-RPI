# Welcome to: Predicting RNA-protein interactions via graph autoencoder and self-supervised learning strategies
RNA-protein interactions (RPIs) are essential for many biological functions and are associated with various diseases. Traditional methods for detecting RPIs are labor-intensive and costly, necessitating efficient computational methods. We proposed a novel RPI prediction framework based on graph neural networks (GNNs) that addressed key limitations of existing methods, such as inadequate feature integration and negative samples construction. Our method represented RNAs and proteins as nodes in a unified interaction graph, enhancing the representation of RPI pairs through multi-feature fusion and employing self-supervised learning strategies for model training. The model's performance was validated through 5-fold cross-validation, achieving accuracies of 0.936, 0.881, 0.955, 0.974, and 0.951 on the RPI488, RPI369, RPI2241, RPI1807, and RPI1446 datasets, respectively. Additionally, in cross-species generalization tests, our method outperformed existing methods, achieving an overall accuracy of 0.997 across 10,093 RPI pairs. Compared with other state-of-the-art RPI prediction methods, our approach demonstrates greater robustness and stability in RPI prediction.

This RPI prediction tool developed by a team from the Chinese University of Hong Kong (Shenzhen)

![The workflow of this study](https://github.com/GGCL7/Graph-RPI/blob/main/workflow.png)


# Dataset for this study
We provided our dataset and you can find them [Datasets](https://github.com/GGCL7/Graph-RPI/tree/main/Data)


# Model source code
The source code for training our models can be found here [Model source code](https://github.com/GGCL7/Graph-RPI/tree/main/Code).

