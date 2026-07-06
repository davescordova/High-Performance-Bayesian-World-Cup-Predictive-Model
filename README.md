# 🏆 2026 FIFA World Cup — High-Performance Bayesian Predictive Model

This repository contains an advanced, GPU-accelerated statistical framework that models international football team strengths and simulates the 2026 FIFA World Cup. Utilizing **PyMC**, **JAX**, and the **BlackJAX** NUTS sampler, the framework performs rapid MCMC sampling, real-time match result injections, and precise tournament path simulations.

---

## 📂 Project Architecture & Notebooks

The model is distributed across three discrete workflows designed for high-performance data pipeline execution:

### 1️⃣ `2026 World Cup.ipynb`
- **🎯 Purpose:** Core baseline configuration and parameters inference.
- **🔍 Details:** Downloads and filters historical international match data from the post-2022 World Cup era. Compiles the baseline Bayesian Poisson model on the GPU, extracts latent team metrics (`attack` and `defense`), and exports the foundational 32×32 win probability matrix.

### 2️⃣ `2026 World Cup - finals.ipynb`
- **🎯 Purpose:** Predictive tournament simulation engine.
- **🔍 Details:** Runs the dual-engine tournament propagation. It performs exact Markov Chain path multiplication across all prospective tournament combinations alongside a deterministic maximum-likelihood bracket tree visualization using custom, dynamically colored nodes.

### 3️⃣ `2026 World Cup - finals updated after closed games.ipynb`
- **🎯 Purpose:** Live real-time tournament tracking and pruning.
- **🔍 Details:** Features an interactive shell prompt to inject real-time match results as they conclude during the tournament. It dynamically masks out completed matches, clips the probability matrix bounds, and instantly recalculates updated compounding title probabilities for the remaining active teams.

---

## 🧮 Mathematical Formulation

The model evaluates the goal-scoring dynamic between competing teams ($i$ and $j$) at neutral or non-neutral venues as independent Poisson variables:

$$\text{Goals}_{i} \sim \text{Poisson}(\theta_{i})$$
$$\text{Goals}_{j} \sim \text{Poisson}(\theta_{j})$$

The log-linear scoring intensities ($\theta_i, \theta_j$) incorporate overall baseline efficiency, home-field advantage (where applicable), host population proxies, and structural team parameters:

$$\log(\theta_{i}) = \text{baseline} + \text{home-adv} \cdot \text{home-adv-mask} + \text{attack}_{i} - \text{defense}_{j}$$
$$\log(\theta_{j}) = \text{baseline} + \text{attack}_{j} - \text{defense}_{i}$$

⚖️ To ensure model identifiability, zero-sum constraints are enforced on the attack and defense deterministic vectors via mean-centering within the PyMC graph.

---

## ⚡ Optimization & Execution Environment

The scripts are engineered exclusively for **VS Code Jupyter Notebooks running inside an Ubuntu/Linux WSL environment on Windows**.

🛡️ To prevent CPU bottlenecks, thread thrashing, and VRAM memory crashes under heavily parallelized JAX operations, the following strict optimization block is programmatically enforced at the start of every notebook:

```python
import os, jax, logging, pymc.sampling.jax

os.environ.update({
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "XLA_PYTHON_CLIENT_ALLOCATOR": "platform"
})

logging.getLogger("pymc").setLevel(logging.ERROR)
jax.config.update("jax_platform_name", "cuda")
```

🚀 Sampling is executed using `pymc.sampling.jax.sample_blackjax_nuts` with vectorized chains and progress bars disabled to keep the runtime console clean.

---

## 🧰 Technical Stack

- 🐍 **Languages:** Python
- 🎲 **Probabilistic Programming:** PyMC (v5+), JAX, BlackJAX
- 📊 **Data Science:** Pandas, NumPy, SciPy
- 📈 **Visualization:** Matplotlib (Custom vector SVG graphs)

---

## 🖥️ Requirements

To replicate the execution environment, ensure your local workspace matches the following specifications:

### 🔧 System & Hardware Specifications
- 🐧 **Operating System:** Ubuntu / Linux WSL2 environment running on Windows 10/11.
- 🎮 **Hardware Acceleration:** CUDA-compatible NVIDIA GPU with a matching standalone installation of the CUDA Toolkit.

### 📦 Python Environment Dependencies
- `python` >= 3.12
- `pymc` >= 5.0.0
- `jax` & `jaxlib` (Must be compiled/installed explicitly with matching `cuda` platform wheels)
- `blackjax`
- `pandas`
- `numpy`
- `matplotlib`
