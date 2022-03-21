# Homophily-Enhanced Self-supervision (HES-GSL)


This is a PyTorch implementation of the Homophily-Enhanced Self-supervision (HES-GSL), and the code includes the following modules:

* Dataset Loader (Cora, Citeseer, Pubmed, ogbn-arxiv, Texas, Cornell, Wisconsin, Actor, MNIST, and Fashion-MNIST)

* Various supervision signals ($\mathcal{L}_{cla}$, $\mathcal{L}_{DAE}$, and $\mathcal{L}_{HES}$)

* Training paradigm pre-training and fine-tuning on ten datasets

* Visualization and evaluation metrics 

  

## Main Requirements

* networkx==2.5
* numpy==1.19.2
* scipy==1.3.1
* torch==1.6.0
* dgl==0.6.1



## Description

* main.py  
  * get_loss_classification() -- Calculate downstream supervision (loss $\mathcal{L}_{cla}$)
  * get_loss_reconstruction() -- Calculate reconstruction-based self-supervision (loss $\mathcal{L}_{DAE}$)
  * main() -- Pre-train and fine-tune model for node classification task on ten real-world datasets
* model.py  
  
  * GCN_CLA() -- GCN classifier
  * GSL() -- Learn a task-specific underlying graph structure
  * GCN_DAE() -- Denoising Autoencoder
  * get_loss_homophily() -- Calculate the homophily-enhanced self-supervision (loss $\mathcal{L}_{HES}$)
* dataset.py  

  * load_data() -- Load Cora, Citeseer, Pubmed, Texas, Cornell, Wisconsin, and Actor datasets
  * load_ogb_data() -- Load ogbn-arxiv dataset
  * load_mnist_data() -- Load MNIST dataset
  * load_fashionmnist_data() -- Load Fashion-MNIST dataset
* utils.py  
  * get_random_mask() -- Add noise to input node features
  * top_k() -- Construct a kNN graph
  * knn_fast() -- Construct a kNN graph in a faster manner
  * get_homophily() -- Calculate the homophily ratio of the generated graph




## Running the code

1. Install the required dependency packages

3. To get the results on a specific *dataset*, please run with proper hyperparameters:

  ```
python main.py --dataset data_name
  ```

where the *data_name* is one of the ten datasets (Cora, Citeseer, Pubmed, ogbn-arxiv, Texas, Cornell, Wisconsin, Actor, MNIST, and Fashion-MNIST). Use the *Cora* dataset as an example: 

```
python main.py --dataset cora
```



## License

Homophily-Enhanced Self-supervision (HES-GSL) is released under the MIT license.