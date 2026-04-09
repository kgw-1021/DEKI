
# 🌌 Distributed Ensemble Consensus Optimization (DECO)

**DECO (Distributed Ensemble Consensus Optimization)** is a derivative-free, and fully Bayesian framework for solving complex nonlinear distributed optimization problems using Factor Graphs. 

By combining the structural elegance of **ADMM (Alternating Direction Method of Multipliers)** with the derivative-free power of **EKI (Ensemble Kalman Inversion)**, this framework treats optimization not as finding a single point, but as evolving a fluid distribution of beliefs.

---

## ✨ Key Innovations

### 1. Ensemble Consensus (Bayesian ADMM)
Unlike traditional ADMM or standard Kalman filters that compress information into Gaussian approximations (mean and covariance), our Variable Nodes (`VNode`) perform **Particle-to-Particle (1:1) matching**.
- Preserves Non-Gaussian and Multi-modal distributions natively.
- Full Uncertainty Quantification: The final $Z$ ensemble represents the true posterior distribution, not just a MAP estimate.
- Dual variables ($\lambda$) act as Bayesian message-passing agents, minimizing KL divergence between local and global beliefs.

### 2. Jacobian-Free Local Optimization
Factor Nodes (`FNode`) utilize **Ensemble Kalman Inversion (EKI)** to solve complex nonlinear constraints.
- **No Derivatives Required:** Gradients are implicitly inferred from the cross-covariance of the ensemble.
- Highly parallelizable and robust to highly non-convex error landscapes.

### 3. Adaptive NIS-based Annealing (Innovation Consistency)
The framework features a self-aware noise scheduling mechanism using the **Normalized Innovation Squared (NIS, $\eta$)**.
- Evaluates the filter's confidence (`Expected Variance`) against reality (`Actual Error Squared`).
- **$\eta \gg 1$:** Injects explosive noise to escape local minima.
- **$\eta \ll 1$:** Rapidly quenches noise for exact and stable convergence.

---

## 📐 Mathematical Formulation

### 1. Local Optimization via EKI ($X$-update)
Each `FNode` updates its local ensemble $X$ to minimize physical constraint errors $e(X)$ and ADMM penalties without computing Jacobians:

$$ K = C_{XY} (C_{YY} + \Gamma_{total})^{-1} $$
$$ X_{new}^{(j)} = X^{(j)} + K \left( -E_{total}(X^{(j)}) + \nu^{(j)} \right) $$

Where $\nu^{(j)}$ is adaptively scaled by the $\eta$-annealing mechanism.

### 2. Pure Ensemble Consensus ($Z$-update)
The central `VNode` orchestrates global consensus purely using the ADMM penalty $\rho$ as a scalar weight, strictly preserving the $j$-th particle's identity:

$$Z^{(j)} = \frac{\sum_k \rho_k \left( X_k^{(j)} + \lambda_k^{(j)}/\rho_k \right)}{\sum_k \rho_k}$$

### 3. Dual Variable Update ($\lambda$-update)
Independent dual updates for every particle $j$:

$$\lambda_k^{(j)} \leftarrow \lambda_k^{(j)} + \rho_k \left( X_k^{(j)} - Z^{(j)} \right)$$

---

## 🏗️ Architecture

- **`Graph.py`**: Contains the core topological structures (`Node`, `Edge`, `Graph`). Manages the connections and memory (local ensembles and dual variables) for each edge.
- **`Node.py`**:
  - `VNode`: Handles the Pure Ensemble $Z$-update and dynamic $\rho$ scheduling.
  - `FNode`: Handles the EKI-based local $X$-update, error evaluations, and $\eta$-based adaptive noise annealing.
- **`toy.py`**: A 2D simulation playground demonstrating distance and prior constraints resolving a non-convex configuration.

---
