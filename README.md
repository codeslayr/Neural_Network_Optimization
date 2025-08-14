# HyperOptX: A Unified Framework for Comparative Evaluation of Optimizers in CNN Training

**Authors:** Abdurrahman Ansari, Nafis Bhamjee, Sahil Ketkar  
**Affiliation:** Ontario Tech University  

---

## 📖 Abstract
The performance of Convolutional Neural Networks (CNNs) is critically dependent on the effective tuning of their hyperparameters.  
This project introduces **HyperOptX**, a standardized **PyTorch** framework for the comparative evaluation of first-order, second-order, and metaheuristic optimizers for this task.  

Using the **Fashion-MNIST** dataset, we conduct a rigorous comparison of:
- **Baseline Adam**
- **Random Search-tuned SGD** and **L-BFGS**
- **Two metaheuristic algorithms:** Genetic Algorithm (GA) and Particle Swarm Optimization (PSO)  

The metaheuristic optimizers were uniquely tasked with **simultaneously optimizing both the training parameters and key architectural properties of the CNN**.  

**Key finding:** PSO achieved the highest test accuracy of **91.23%**.  
This study provides strong evidence that for complex deep learning tasks, **metaheuristic algorithms that co-optimize architecture and hyperparameters** offer a more effective and robust pathway to achieving state-of-the-art performance.

---

## 🚀 The HyperOptX Framework
The **core** of this research is the **HyperOptX** framework, designed to provide a fair and reproducible _"apples-to-apples"_ comparison of different optimizer families.

---

## 🔬 Optimizers Compared
We evaluated **five** distinct optimizers, categorized into three families:

### First-Order Optimization
- **Adam** (baseline)
- **Stochastic Gradient Descent (SGD)** — tuned via Random Search

### Quasi-Second-Order Optimization
- **L-BFGS** — tuned via Random Search

### Metaheuristic Optimization
- **Genetic Algorithm (GA)**
- **Particle Swarm Optimization (PSO)**

> For GA and PSO, the search space included not only training parameters (e.g., learning rate) but also **architectural parameters** (number of convolutional filters and dense layer neurons).

---

## 📊 Key Results

### Final Performance on the Test Set

| Optimizer                  | Type            | Final Test Accuracy | Final Test Loss |
|----------------------------|-----------------|---------------------|-----------------|
| Adam                       | Baseline        | 88.57%              | 0.3513          |
| SGD                        | Random Search   | 85.40%              | 0.4169          |
| L-BFGS                      | Random Search   | 85.12%              | 1.1023          |
| Genetic Algorithm (GA)     | Metaheuristic   | 90.39%              | 0.3554          |
| Particle Swarm Opt. (PSO)  | Metaheuristic   | **91.23%**          | **0.3055**      |

---

### Comparative Learning Dynamics
The plot below (generated in the experiments) shows the validation accuracy for each optimizer over the **last 10 training epochs**.  
It clearly illustrates the **superior stability and convergence** of the metaheuristic approaches (GA and PSO).

---

## ⚙️ How to Run the Code

### 1. Prerequisites
This project is built using Python and PyTorch.  
Install the required dependencies:
```bash
pip install torch torchvision pandas matplotlib seaborn scikit-learn jupyter
```
### 2. Running the Experiment

Clone the repository and navigate to the directory:

```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

Open and run the cnn_hyperparameter_optimization.ipynb notebook in Jupyter:
```bash
jupyter notebook cnn_hyperparameter_optimization.ipynb
```

### 📂 Notebook Organization
1. Setup & Imports
2. Data Loading
3. Model Definitions
4. Core Utilities
5. Optimizer Experiment Functions
6. Plotting Functions
7. Main Execution

### 📄 Citation

If you use this work, please cite:
```bash
@misc{ansari2025hyperoptx,
  title     = {HyperOptX: A Unified Framework for Comparative Evaluation of First-Order, Second-Order, and Metaheuristic Optimizers in CNN Training},
  author    = {Ansari, Abdurrahman and Bhamjee, Nafis and Ketkar, Sahil},
  year      = {2025},
  howpublished = {ENGR 5010G - Advanced Optimization, Ontario Tech University}
}
```
