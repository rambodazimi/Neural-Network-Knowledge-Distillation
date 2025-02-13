# Knowledge Distillation on Simple Convolutional Neural Networks (CNNs)

This repository contains a Python implementation of **Knowledge Distillation** (KD) applied to simple Convolutional Neural Networks (CNNs) using the **CIFAR-10** dataset. The goal is to transfer knowledge from a **larger, more accurate Teacher model** to a **smaller, more efficient Student model**, improving the student's performance through distillation.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
- [Training Procedure](#training-procedure)
- [Results](#results)
- [Usage](#usage)
- [References](#references)

## Introduction

Knowledge Distillation is a technique where a smaller **student** model learns from a larger **teacher** model by mimicking its output probabilities. This allows the student model to achieve higher accuracy than if it were trained independently, while maintaining a lower computational cost.

In this project:

- The **teacher model** is a CNN with **4 convolutional layers**.
- The **student model** is a CNN with **2 convolutional layers**.
- We train both models separately and then train the student model using the teacher’s knowledge.

## Installation

To run this notebook, you need to install the required dependencies. You can do so using the following commands:

```bash
pip install torch torchvision matplotlib
```

Alternatively, you can create a virtual environment:

```bash
python -m venv kd_env
source kd_env/bin/activate  # On Windows use: kd_env\Scripts\activate
pip install torch torchvision matplotlib
```

## Dataset

We use the **CIFAR-10** dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

- **Training set:** 50,000 images
- **Test set:** 10,000 images

The dataset is loaded and preprocessed using `torchvision.datasets` and `torchvision.transforms`.

## Model Architectures

### Teacher Model:

- 4 **Conv2D** layers
- 2 **MaxPool2D** layers
- 1 **Flatten** layer
- 2 **Fully Connected (Linear)** layers
- **Parameters:** 1,186,986

### Student Model:

- 2 **Conv2D** layers
- 2 **MaxPool2D** layers
- 1 **Flatten** layer
- 2 **Fully Connected (Linear)** layers
- **Parameters:** 267,738

## Training Procedure

### Step 1: Individual Training

- Train both the **teacher** and **student** models separately.
- Evaluate performance without knowledge distillation.

### Step 2: Knowledge Distillation Training

- Use the **teacher model’s logits** (pre-softmax outputs) as targets for the **student model**.
- Apply **KL-Divergence loss** alongside standard cross-entropy loss.
- Experiment with different temperature values in softmax for improved performance.

## Results

### Without Knowledge Distillation

| Model   | Parameters | Training Time | Test Accuracy |
| ------- | ---------- | ------------- | ------------- |
| Teacher | 1.19M      | \~3 min       | **74.43%**    |
| Student | 267K       | \~2 min       | **69.51%**    |

### With Knowledge Distillation

| Model   | Parameters | Training Time | Test Accuracy |
| ------- | ---------- | ------------- | ------------- |
| Teacher | 1.19M      | \~3 min       | **74.43%**    |
| Student | 267K       | \~2 min       | **71.89%**    |

The student model achieves a **higher accuracy when trained using KD** compared to training independently.

## Usage

Clone the repository and run the Jupyter Notebook:

```bash
git clone https://github.com/yourusername/knowledge-distillation.git
cd knowledge-distillation
jupyter notebook knowledge_distillation.ipynb
```

To train the models individually:

```python
python train_teacher.py
python train_student.py
```

To train the student using knowledge distillation:

```python
python train_kd.py
```

## References

- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. [arXiv preprint arXiv:1503.02531](https://arxiv.org/abs/1503.02531).
- CIFAR-10 Dataset: [Krizhevsky, 2009](https://www.cs.toronto.edu/~kriz/cifar.html)
