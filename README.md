# BERT for Natural Language Inference on e-SNLI Dataset
## CS4248 Group 50 Project Code Repository

This project explores the application of the BERT (Bidirectional Encoder Representations from Transformers) model to the [e-SNLI dataset](https://github.com/OanaMariaCamburu/e-SNLI), aiming to understand its state-of-the-art performance in natural language inference tasks. The focus is on dissecting BERT's architecture and its interplay with semantic complexity to demystify the "black-box" nature of such models. By exploring BERT’s deep bidi-rectional architecture and its interplay with the semantic complexity of NLI, this project aims to elucidate the mechanisms enabling BERT to model contextual relationships in text effectively. Please find our test results and saved models [here](https://drive.google.com/drive/folders/16zaMKJEi7cRjNexH_t97SAsJzUjC0KoA?usp=drive_link)

## Repository Overview
This repository contains code files and notebooks used for various tasks including data analysis, data pre-processing, model training and model evaluation. Here is a brief overview of the repository's structure: 
```
|-analysis/
|-experiments/
|-models/
|-pre-processing/
|-requirements.txt
```
Contents:
- `/model`:  Contains the implementation of the BERT model `BERT-model.py` and a baseline model `base-model.py`  
- `/analysis`: Contains utility code involved in corpus analysis
- `/pre-processing`: Contains functions used for data pre-processing and transformation of input before training of model
- `/experiments`: Contains core experiments that we conducted throughout this project

This repository is organized to include both original Jupyter Notebook `.ipynb` files, located in the `\notebook` directory within each folder, where the majority of development took place, as well as cleaned-up Python `.py` files. These Python files contain only the essential components needed to implement our code easily, while the Jupyter Notebooks could help guide users through our implementation.

## Getting Started 
For people who are interested in replicating our experiments, you may take a look at each notebook we used stored under `experiments` folder.

Make sure you have the necessary dependencies installed. We primarily use PyTorch for implementing the BERT model and conducting experiments. Dependencies used throughout this project are recorded in `requirements.txt`, to install all necessary dependencies, run:

```
pip install -r requirements. txt 
```

## Acknowledgments
- Dataset provided by [e-SNLI Dataset](https://github.com/OanaMariaCamburu/e-SNLI).
- We'd like to thank Ms. Esther Gan as well as Prof. Kan Min-Yen and Prof. Christian Von Der Weth for their guidance throughout this project as well as the course throughout the semester.

## Contributing
If you encounter any difficulties, please raise an [Issue](https://github.com/avock/cs4248-group50-eSNLI/issues). If you have any suggestions or improvements you'd like to implement, raise a [Pull Request](https://github.com/avock/cs4248-group50-eSNLI/pulls)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
