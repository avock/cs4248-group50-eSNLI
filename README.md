## BERT for Natural Language Inference on e-SNLI Dataset
#### CS4248 Group 50 Project Code Repository

This project explores the application of the BERT (Bidirectional Encoder Representations from Transformers) model to the [e-SNLI dataset](https://github.com/OanaMariaCamburu/e-SNLI), aiming to understand its state-of-the-art performance in natural language inference tasks. The focus is on dissecting BERT's architecture and its interplay with semantic complexity to demystify the "black-box" nature of such models. By exploring BERT’s deep bidi-rectional architecture and its interplay with the semantic complexity of NLI, this project aims to elucidate the mechanisms enabling BERT to model contextual relationships in text effectively. 

## Repository Overview
This repository contais code files and notebooks used for various tasks including data analysis, data pre-processing, model training and model evaluation. Here is a brief overview of the repository's structure: 
- model:  BERT model `BERT-model.py`, baseline model `base-model.py`  
- analysis: Contains utility code involved in corpus analysis
- pre-processing: Contains functions used for data pre-processing and transformation of input before training of model
- experiments: Contains main experiments that we conduct in ablation study, keyword analysis and sequence analysis

## Getting Started 
For people who are interested in replicating our experiments, you may take a look at each notebook we used stored under `experiments` folder.

Make sure you have the necessary dependencies installed. We primarily use PyTorch for implementing the BERT model and conducting experiments. Here's a quick guide to getting started:

Prerequisites
- Python 3.9
- PyTorch
- Scikit-learn
- Transformers

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgments
- Dataset provided by [e-SNLI Dataset](https://github.com/OanaMariaCamburu/e-SNLI).
- We'd like to thank Ms. Esther Gan as well as Prof. Kan Min-Yen and Prof. Christian Von Der Weth for their guidance throughout this project as well as the course throughout the semester.
