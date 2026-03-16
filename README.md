
```markdown
# MLOps
# MLOps Assignment – MNIST Digit Classifier

## Overview
This repository demonstrates a PyTorch-based digit classifier trained on the MNIST dataset with MLflow integration for experiment tracking and model logging.

The model is a simple feedforward neural network with one hidden layer. Multiple training runs are performed with different learning rates and batch sizes, and metrics such as loss and accuracy are tracked in MLflow.

---

## Repository Structure
```

MLOps/
├─ train.py              # Training script with MLflow logging
├─ requirements.txt      # Required Python packages
├─ data/                 # MNIST dataset (downloaded automatically)
├─ mlruns/               # MLflow experiment tracking logs
└─ README.md             # Project documentation

````

---

## Setup Instructions

1. **Clone the repository**
```bash
git clone -b test-branch https://github.com/AhmedElrashidy11/MLOps.git
cd MLOps
````

2. **Create a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Running the Training

The training script runs multiple experiments with different hyperparameters:

```bash
python train.py
```

The following runs will be executed:

* lr=0.1, batch=64, epochs=5
* lr=0.01, batch=64, epochs=5
* lr=0.001, batch=64, epochs=5
* lr=0.001, batch=32, epochs=5
* lr=0.0005, batch=128, epochs=5

Metrics (loss and accuracy) and models are logged automatically to MLflow.

---

## MLflow Tracking

To view experiment results:

```bash
mlflow ui
```

Open the UI at `http://127.0.0.1:5000` to visualize metrics, parameters, and registered models.

---

## Dependencies

* Python 3.8+
* torch
* torchvision
* mlflow

See `requirements.txt` for exact versions.

---

## Author

**Ahmed Mohammed** – Student ID: 202202168

```

You can save this text directly as `README.md` in your repository root.  

If you want, I can also **add placeholders for screenshots** to make it visually appealing for a report submission. Do you want me to do that?
```
