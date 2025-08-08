# Awesome Activation Sparsification

A curated list of awesome neural network **activation sparsification** methods. Inspired by [Awesome Model Quantization](https://github.com/Efficient-ML/Awesome-Model-Quantization) and [Awesome Pruning](https://github.com/ghimiredhikura/Awasome-Pruning).

Please feel free to contribute to add more papers.

## Type of Sparsification

| Type | `U` | `S` | `R` | `T` | `D` | `Pre` | `Post` |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Explanation** | Unstructured | Structured | Regularizer | Threshold | Dropout | Pre-training | Post-training |

## Survey Papers

### 2021

- [[ICML](https://arxiv.org/abs/2102.00554)] Sparsity in Deep Learning: Pruning and growth for efficient inference and training in neural networks

## Papers

### 2025

- [[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34227)] From PEFT to DEFT: Parameter Efficient Finetuning for Reducing Activation Density in Transformers [**`U`**] [**`R`**] [**`Post`**]
- [[Coring](https://aclanthology.org/2025.coling-main.180/)] Prosparse: Introducing and enhancing intrinsic activation sparsity within large language models [**`U`**] [**`R`**] [**`T`**] [**`Post`**]
- [[ICLR](https://openreview.net/forum?id=dGVZwyq5tV)] Training-Free Activation Sparsity in Large Language Models [**`S`**] [**`T`**]
- [[ICLR](https://openreview.net/forum?id=9VMW4iXfKt)] R-Sparse: Rank-Aware Activation Sparsity for Efficient LLM Inference [**`S`**] [**`T`**]
- [[ICML](https://openreview.net/forum?id=1b6NNpFYI4)] La RoSA: Enhancing LLM Efficiency via Layerwise Rotated Sparse Activation [**`U`**]

### 2024

- [[COLT](https://proceedings.mlr.press/v247/awasthi24a.html)] Learning Neural Networks with Sparse Activations [**`U`**] [**`Pre`**]
- [[ICLR](https://openreview.net/forum?id=uvXK8Xk9Jk)] Deep Neural Network Initialization with Sparsity Inducing Activations [**`U`**] [**`T`**] [**`Pre`**]
- [[ICLR](https://openreview.net/forum?id=vZfi5to2Xl)] SAS: Structured Activation Sparsification [**`S`**] [**`Pre`**]
- [[NeurIPS](https://openreview.net/forum?id=Ppj5KvzU8Q)] 
Improving Sparse Decomposition of Language Model Activations with Gated Sparse Autoencoders [**`U`**] [**`R`**] [**`Post`**]
- [[NeurIPS](https://neurips.cc/virtual/2024/poster/96769)] Exploiting Activation Sparsity with Dense to Dynamic-k Mixture-of-Experts Conversion [**`U`**] [**`R`**] [**`Post`**]
- [[NeurIPSW](https://proceedings.mlr.press/v262/seng-chua24a.html)] Post-Training Statistical Calibration for Higher Activation Sparsity [**`U`**] [**`T`**] [**`Post`**]
- [[EMNLP](https://aclanthology.org/2024.emnlp-main.1038/)] CHESS: Optimizing LLM Inference via Channel-Wise Thresholding and Selective Sparsification [**`S`**] [**`T`**]
- [[WACV](https://openaccess.thecvf.com/content/WACV2024/html/Zhu_CATS_Combined_Activation_and_Temporal_Suppression_for_Efficient_Network_Inference_WACV_2024_paper.html)] CATS: Combined Activation and Temporal Suppression for Efficient Network Inference [**`U`**] [**`R`**] [**`T`**] [**`Post`**]

### 2023

- [[arXiv](https://arxiv.org/abs/2310.04564)] ReLU Strikes Back: Exploiting Activation Sparsity in Large Language Models [**`U`**] [**`Post`**]
- [[CVPR](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_SparseViT_Revisiting_Activation_Sparsity_for_Efficient_High-Resolution_Vision_Transformer_CVPR_2023_paper.html)] SparseViT: Revisiting Activation Sparsity for Efficient High-Resolution Vision Transformer [**`U`**] [**`Post`**]
- [[CVPRW](https://openaccess.thecvf.com/content/CVPR2023W/ECV/html/Zhu_STAR_Sparse_Thresholded_Activation_Under_Partial-Regularization_for_Activation_Sparsity_Exploration_CVPRW_2023_paper.html)] STAR: Sparse Thresholded Activation under partial-Regularization for Activation Sparsity Exploration [**`U`**] [**`R`**] [**`T`**] [**`Post`**]
- [[ICCVW](https://openaccess.thecvf.com/content/ICCV2023W/RCV/html/Grimaldi_Accelerating_Deep_Neural_Networks_via_Semi-Structured_Activation_Sparsity_ICCVW_2023_paper.html)] Accelerating Deep Neural Networks via Semi-Structured Activation Sparsity [**`S`**] [**`Post`**]
- [[SIGIR](https://dl.acm.org/doi/abs/10.1145/3539618.3592051)] Representation Sparsification with Hybrid Thresholding for Fast SPLADE-based Document Retrieval [**`U`**] [**`R`**] [**`T`**] [**`Post`**]

### 2022

- [[DSD](https://ieeexplore.ieee.org/abstract/document/9996686)] ARTS: An adaptive regularization training schedule for activation sparsity exploration [**`U`**] [**`R`**] [**`Post`**]

### 2020

- [[ICML](https://proceedings.mlr.press/v119/kurtz20a.html)] Inducing and Exploiting Activation Sparsity for Fast Inference on Deep Neural Networks [**`U`**] [**`R`**] [**`T`**] [**`Post`**]

### 2019

- [[arXiv](https://arxiv.org/abs/1903.11257)] How Can We Be So Dense? The Benefits of Using Highly Sparse Representations [**`U`**] [**`Pre`**]
- [[CVPR](https://openaccess.thecvf.com/content_CVPR_2019/html/Georgiadis_Accelerating_Convolutional_Neural_Networks_via_Activation_Map_Compression_CVPR_2019_paper.html)] Accelerating Convolutional Neural Networks via Activation Map Compression [**`U`**] [**`R`**] [**`Post`**]
- [[ICTAI](https://ieeexplore.ieee.org/abstract/document/8995451)] DASNet: Dynamic Activation Sparsity for Neural Network Efficiency Improvement [**`U`**] [**`D`**] [**`Post`**]

### 2018

- [[HPCA](https://ieeexplore.ieee.org/document/8327000)] Compressing DMA Engine: Leveraging Activation Sparsity for Training Deep Neural Networks [**`U`**] [**`D`**] [**`Pre`**]

### 2015

- [[NeurIPS](https://papers.nips.cc/paper_files/paper/2015/hash/5129a5ddcd0dcd755232baa04c231698-Abstract.html)] Winner-Take-All Autoencoders [**`U`**] [**`Pre`**]

## Relevant Awesome Lists

- [Awesome Activation Engineering](https://github.com/ZFancy/awesome-activation-engineering)

- [Awesome Pruning](https://github.com/ghimiredhikura/Awasome-Pruning)

- [Awesome-LLM-Compression](https://github.com/HuangOwen/Awesome-LLM-Compression)

- [Awesome-Mixture-of-Experts](https://github.com/SuperBruceJia/Awesome-Mixture-of-Experts)