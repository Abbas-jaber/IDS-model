#  Cortex Guard: AI-Powered Network Intrusion Detection System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.0-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An advanced Anomaly-Based Network Intrusion Detection System (ANIDS) that leverages deep learning to detect evolving cyber threats that traditional signature-based systems miss.

##  Overview

Cortex Guard addresses the critical limitations of traditional Signature-Based Intrusion Detection Systems (SNIDS) like Snort and Suricata by using deep learning to identify zero-day attacks, Advanced Persistent Threats (APTs), and novel attack patterns in network traffic.

### Key Features

- **99% Detection Accuracy** on real-world network traffic datasets
- **Multiple Deep Learning Architectures**: DNN, LSTM, CNN, and ANN implementations
- **Dual Classification Support**: Binary (benign/malicious) and multi-class (DoS, DDoS, Botnet)
- **Automated Data Pipeline**: Comprehensive preprocessing with cleaning and feature extraction
- **GPU-Optimized Training**: Efficient model training with CUDA support
- **AI-Powered Insights**: Integrated Ollama chatbot for system transparency
- **Command-Line Framework**: User-friendly interface for model training and evaluation

##  Quick Start

### Prerequisites

```bash
# Python 3.9 or higher
python --version

# CUDA-capable GPU (optional but recommended)
nvidia-smi
```

### Installation

```bash
# Clone the repository
git clone https://github.com/Abbas-jaber/IDS-model.git
cd IDS-model

# Create virtual environment
conda create -n cortex-guard python=3.9
conda activate cortex-guard

# Install dependencies
pip install -r requirements.txt

# For GPU support
conda install cudatoolkit=11.2 cudnn=8.1.0
pip install tensorflow-gpu==2.10.0
```

### Dataset Setup

Download the CSE-CIC-IDS2018 dataset:

```bash
# Install AWS CLI
# Follow instructions at: https://aws.amazon.com/cli/

# Pull dataset from S3
aws s3 sync s3://cse-cic-ids2018 ./datasets --no-sign-request --region ca-central-1
```

## 📊 Usage

### Launch Cortex Guard

```bash
python cortex_guard.py
```

### 1. Data Preprocessing

```bash
# Option 1: Clean all datasets
[1] Clean Data → Clean all files

# Option 2: Combine datasets
[2] Combine Data

# Option 3: Final cleaning
[3] Clean Data (Version 2 - Epoch Check)
```

### 2. Model Training

```bash
# Option 4: Build and train model
[4] Build IDS Model

# Select architecture:
# - DNN (Deep Neural Network)
# - LSTM (Long Short-Term Memory)
# - CNN (Convolutional Neural Network)
# - ANN (Artificial Neural Network)

# Provide dataset path
# Example: ./datasets/binary-class.csv
```

### 3. Model Evaluation

```bash
# Option 5: Evaluate trained model
[5] Evaluate Model

# Provide:
# - Model architecture (DNN/LSTM/CNN/ANN)
# - Test dataset path
# - Trained model file (.h5)

# Option 6: Generate confusion matrix
[6] Generate Confusion Matrix
```

##  Architecture
### Deep Learning Models

#### DNN (Deep Neural Network)
- **Architecture**: 128 → 64 → output layers
- **Accuracy**: 99.71% (best performing)
- **Best for**: General-purpose classification

#### LSTM (Long Short-Term Memory)
- **Architecture**: 100 LSTM units + dense layers
- **Accuracy**: 99.97%
- **Best for**: Sequential pattern detection

#### CNN (Convolutional Neural Network)
- **Architecture**: 2 conv blocks (32, 64 filters) + dense layers
- **Accuracy**: Varies by dataset
- **Best for**: Spatial feature extraction

#### ANN (Artificial Neural Network)
- **Architecture**: 128 → 64 → output with dropout
- **Accuracy**: 89.67%
- **Best for**: Baseline comparisons

### System Pipeline

```
Raw Dataset → Data Cleaning → Feature Extraction → 
Model Training → Evaluation → Confusion Matrix
```

##  Performance Results

### DNN Model Performance

| Dataset | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| 02-16-2018 | 99.71% | 99.71% | 99.71% | 99.71% |
| 02-22-2018 | 99.97% | 99.96% | 99.97% | 99.96% |
| Binary-class | 96.24% | 96.20% | 96.24% | 96.19% |
| Multi-class | 96.15% | 94.99% | 96.15% | 95.35% |

### LSTM Model Performance

| Dataset | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| 02-22-2018 | 99.97% | 99.96% | 99.97% | 99.96% |
| Binary-class | 90.12% | 89.97% | 90.12% | 89.72% |
| Multi-class | 91.79% | 91.11% | 91.79% | 90.84% |

##  Technical Stack

- **Deep Learning**: TensorFlow 2.10.0, Keras
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **AI Integration**: Ollama (deepseek-r1)
- **Hardware Acceleration**: CUDA, cuDNN
- **Dataset**: CSE-CIC-IDS2018 from UNB

##  Project Structure

```
IDS-model/
├── cortex_guard.py          # Main CLI framework
├── models/
│   ├── dnn_model.py         # DNN architecture
│   ├── lstm_model.py        # LSTM architecture
│   ├── cnn_model.py         # CNN architecture
│   └── ann_model.py         # ANN architecture
├── preprocessing/
│   ├── data_cleaning.py     # Initial cleaning
│   ├── data_combining.py    # Dataset combination
│   └── data_cleaning_v2.py  # Final cleaning
├── evaluation/
│   ├── model_evaluator.py   # Model testing
│   └── confusion_matrix.py  # Visualization
├── datasets/                # Training data
├── trained_models/          # Saved .h5 models
└── results/                 # Evaluation outputs
```

##  Key Innovations

1. **Hybrid Architecture Approach**: Multiple deep learning models for comprehensive threat detection
2. **Advanced Data Pipeline**: Automated cleaning with AI-powered explanations
3. **Real-World Dataset**: Trained on contemporary attack patterns (CSE-CIC-IDS2018)
4. **Multi-Classification**: Identifies specific attack types, not just malicious/benign
5. **GPU Optimization**: Efficient training for large-scale deployments

##  Academic Context

This project was developed as a Bachelor's thesis at Bahrain Polytechnic:

- **Title**: Developing an Anomaly-Based Network Intrusion Detection System Using Deep Learning for Detecting Evolving Cyber Threats
- **Author**: Abbas Al-mutawa
- **Supervisor**: Dr. Saeed Al-Samhi
- **Date**: May 2025
- **Grade**: Distinction

##  Citation

If you use this work in your research, please cite:

```bibtex
@thesis{almutawa2025cortexguard,
  title={Developing an Anomaly-Based Network Intrusion Detection System Using Deep Learning for Detecting Evolving Cyber Threats},
  author={Al-mutawa, Abbas},
  year={2025},
  school={Bahrain Polytechnic},
  type={Bachelor's Thesis}
}
```


##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Future Enhancements

- [ ] Real-time network traffic analysis integration
- [ ] Explainable AI visualization components
- [ ] Federated learning implementation
- [ ] Adversarial defense mechanisms
- [ ] REST API for model deployment
- [ ] Docker containerization
- [ ] Web-based dashboard

##  Contact

Abbas Al-mutawa - [GitHub](https://github.com/Abbas-jaber)

Project Link: [https://github.com/Abbas-jaber/IDS-model](https://github.com/Abbas-jaber/IDS-model)

##  Acknowledgments

- Dr. Saeed Al-Samhi (Project Supervisor)
- Mr. Cyril Anthony (Project Manager)
- Dr. Christos Gatzoulis (ICT Faculty)
- Bahrain Polytechnic ICT Department
- University of New Brunswick (CSE-CIC-IDS2018 dataset)

---

⭐ **Star this repository if you find it helpful!**
