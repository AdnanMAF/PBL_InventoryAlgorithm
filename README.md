# PBL Inventory Forecasting Algorithm

This repository contains experiments and utilities for forecasting product sales and analyzing news sentiment. The project is organized into two main components:

1. **DataMining** – time series forecasting using SARIMAX.
2. **NaturalLanguageProcessing** – sentiment classification of news articles using DistilBERT.

Git LFS is used for the datasets stored under `Data/`.

## Repository structure

```
Data/                        Raw datasets used by the notebooks and scripts
  DM/                        Sales data (`train.csv`, `test.csv`)
  NLP/                       News datasets and labeled sentiment data
DataMining/                  Forecasting notebooks and scripts
  Early Models/              Initial SARIMAX notebooks
  Forecasts/                 Sample forecast CSV output
  Performance/               Hold‑out performance results
  SARIMAX_Train.py           Script to train/forecast for a single product family
NaturalLanguageProcessing/   DistilBERT training and evaluation notebooks
README.md                    Project overview (this file)
```

## Forecasting with SARIMAX

`DataMining/SARIMAX_Train.py` trains a SARIMAX model for one product family at a time. Adjust the constants near the top of the file:

- `FAMILY` – product family name as it appears in `train.csv`.
- `TRAIN_CSV`/`TEST_CSV` – paths to the training and test data.
- `N_VALID` – number of days held out for validation.

Running the script produces two CSV files in the working directory:

- `performance_<FAMILY>.csv` – RMSE, MAE and R² on the validation set.
- `forecast_<FAMILY>.csv` – predictions for every row in `test.csv` for that family.

Example performance output:

```
family,RMSE,MAE,R2,MAPE
AUTOMOTIVE,58.10884029887869,44.713777144597366,0.49796940054631833,10.810187299889614
```

## News sentiment classification

The `NaturalLanguageProcessing` directory contains Jupyter notebooks for training DistilBERT models using either TensorFlow or PyTorch. The datasets in `Data/NLP/` include raw news articles and versions labeled with sentiment scores. The typical workflow is:

1. **Data_Labelling.ipynb** – manually label or review news data.
2. **Data_balancing.ipynb** – balance the dataset across sentiment classes.
3. **TrainDistilBERT-*.ipynb** – train a sentiment classifier.
4. **Evaluation.ipynb** – assess model accuracy.

These notebooks require GPU-enabled TensorFlow or PyTorch along with Hugging Face Transformers.

## Requirements

The forecasting script depends on:

- pandas
- numpy
- statsmodels
- scikit-learn
- pmdarima

The NLP notebooks additionally rely on TensorFlow or PyTorch and the Transformers library.

Install the needed packages with `pip` or your preferred environment manager before running the notebooks or the script.

## License

This project was created as part of a PBL (Project‑Based Learning) exercise and has no specific license. Use the contents as a reference or starting point for your own experiments.
