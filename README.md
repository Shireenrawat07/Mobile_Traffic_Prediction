
# Federated 5G Traffic Prediction

A privacy-preserving federated learning framework for predicting 5G mobile network traffic using deep learning. The system enables multiple clients to collaboratively train a global model without sharing raw data, ensuring both data privacy and accurate traffic forecasting.

---

## Overview

This project addresses the privacy challenges of centralized machine learning by implementing a Federated Learning (FL) architecture. Multiple clients train models locally while a central server aggregates model parameters to build a global model. The project also introduces **Reliability-Aware Federated Averaging (RA-FedAvg)** to improve aggregation in heterogeneous (Non-IID) environments.

---

## Features

- Privacy-preserving federated learning
- LSTM-based traffic prediction
- Flower (FLWR) client-server implementation
- Comparison of LSTM, GRU, CNN, RNN, and MLP
- Comparison of FedAvg, FedProx, FedNova, and RA-FedAvg
- Non-IID client simulation using Dirichlet distribution
- Interactive Streamlit dashboard
- Performance evaluation using MAE, RMSE, and NRMSE

---

## Tech Stack

- Python
- PyTorch
- Flower (FLWR)
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib

---

## Dataset

The project uses a real-world LTE traffic dataset collected from multiple base stations. The data is partitioned among multiple clients to simulate a distributed federated learning environment with varying levels of data heterogeneity.

---

## Workflow

1. Data preprocessing
2. Client-wise data partitioning
3. Local model training
4. Server-side aggregation
5. Global model update
6. Traffic prediction
7. Performance evaluation
8. Dashboard visualization

---

## Results

### Best Deep Learning Model

| Model | MAE | NRMSE |
|------|------:|------:|
| **LSTM** | **0.01363** | **0.02852** |

### Aggregation Comparison

- FedAvg outperformed Simple Average and Median Average.
- RA-FedAvg demonstrated better robustness under heterogeneous client data.
- LSTM combined with RA-FedAvg achieved the best overall prediction performance.

---

## Installation

```bash
git clone https://github.com/Shireenrawat07/Mobile_Traffic_Prediction.git
cd Federated-5G-Traffic-Prediction
pip install -r requirements.txt
```

Launch the dashboard:

```bash
streamlit run app.py
```


## License

This project is developed for academic and research purposes.