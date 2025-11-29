# Risk-Aware Stochastic Resource Allocation (Airport Gate Assignment)

**Course:** Machine Learning (Fall 2025)  
**Student:** CHANG Yung-Hsuan (張永璿)  
**ID:** 111652004  

## 📖 Project Overview
This project implements a **Risk-Aware Stochastic Bin Packing Model** to optimize airport gate assignment under uncertainty. Unlike traditional deterministic heuristics (e.g., Best-Fit Decreasing), this project utilizes **Reinforcement Learning (REINFORCE / Policy Gradient)** to dynamically assign safety buffers to flights based on their scheduled duration and volatility.

The model learns to balance two conflicting objectives:
1.  **Efficiency**: Minimizing the number of gates used.
2.  **Robustness**: Minimizing the risk of overflow (delays).

## 📂 File Structure
The project is organized as follows:

- **`111652004_ML_Final_Project.pdf`** The comprehensive final report containing the problem formulation, methodology, experimental setup, and detailed analysis of the results.

- **`rawbp.ipynb`** The main Jupyter Notebook used for:
  - Defining the Neural Network Policy (`PaddingPolicy`).
  - Implementing the Multi-Seed Training Loop.
  - Evaluating strategies and visualizing results (Boxplots).

- **`core.py`** A helper module designed for high-performance simulation. It contains:
  - `bfd`: Numba-accelerated Best-Fit Decreasing algorithm.
  - `worker_simulate_episode`: Worker function for multiprocessing simulations.
  - `AirportDataGen`: Synthetic data generator for stochastic flight schedules.

- **`figures/`** Contains the generated plots used in the report:
  - `Focused_Efficiency_Gates_Used.png`: Efficiency comparison (AI vs. Fixed Strategies).
  - `Focused_Robustness_Total_Overflow.png`: Robustness comparison.
  - `*_Hybrid_Comparison.png`: Additional visualization including Naive baselines.

## 🚀 How to Run
### Prerequisites
This project requires Python 3.8+ and the following libraries:
```bash
pip install tensorflow numpy matplotlib seaborn tqdm numba pandas
