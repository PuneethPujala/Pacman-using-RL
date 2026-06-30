# Pacman Reinforcement Learning (Approximate Q-Learning)

This repository contains a full web-integrated implementation of **Pacman Reinforcement Learning** based on **Approximate Q-Learning** with a local FastAPI backend and dynamic frontend testing dashboard. 

The reinforcement learning agent was modified and optimized to resolve fundamental algorithmic inefficiencies (including state bootstrapping bugs, feature scaling conflicts, and dead-end trapping). These optimizations raised the agent's win rate on the complex `smallClassic` layout from a baseline of **2%** to an impressive **50%**.

---

## Architecture Overview

1.  **AI Engine (`bustersAgents.py`, `backend/engine.py`):**
    *   Implements the `SavedApproximateQAgent` which uses linear function approximation:
        $$Q(s, a) = \sum_{i} w_i f_i(s, a)$$
    *   State space is summarized via hand-engineered features mapped to weight vectors, allowing the agent to generalize to unseen layouts.
2.  **API Server (`backend/server.py`):**
    *   Built using FastAPI, exposing REST endpoints to manage layout retrieval, start/step/reset game loop instances, toggle background training threads, and save weights.
3.  **Frontend Dashboard (`frontend/`):**
    *   A web-based interface built with vanilla HTML/CSS and JavaScript. Features a real-time visualization of the Pacman board, layout selector, model weight save/download tools, and a training controller.

---

## Reinforcement Learning Enhancements

### 1. Polymorphic Bootstrap Target Fix
*   **The Problem:** Tabular Q-learning agents retrieve maximum successor values using table lookups (`self.q_table`). The approximate Q-agent inherits from the tabular agent but uses feature-weight linear combinations. The inherited `computeValueFromQValues` was hardcoded to perform tabular lookups, resulting in successor state values returning `0.0` for every step. The agent was only updating weights based on immediate rewards, completely unaware of future steps.
*   **The Fix:** Updated `computeValueFromQValues` to polymorphically query `self.getQValue(state, action)` for each action. This allows `ApproximateQAgent` to assess future consequences:
    $$\delta = r + \gamma \max_{a'} Q(s', a') - Q(s, a)$$

### 2. Epsilon & Alpha Decay Schedules
*   **The Problem:** Linear approximate Q-learning can easily diverge or oscillate wildly under static hyperparameter rates. During training, weights would jump from large positive to negative values from one episode to the next, causing the final model to settle on chaotic policies.
*   **The Fix:** Implemented exponential decay for the exploration rate ($\epsilon$) and learning rate ($\alpha$) inside the episode-end hook:
    $$\alpha_t = \alpha_0 \cdot 0.985^t, \quad \epsilon_t = \epsilon_0 \cdot 0.985^t$$
    This transitions the agent smoothly from broad exploration to fine exploitation, stabilizing weights for clean convergence.

### 3. Hybrid Linear/Non-Linear Features
*   **The Problem ("Disappearing Feature Weight Decay"):** Utilizing inverse closest-food distance (`1.0 / minDist`) causes a massive, positive feature value to drop abruptly (e.g., from `1.0` to `0.2`) when Pacman eats a pellet and the next closest pellet is far away. This drop creates a large negative TD-error that decays the food-seeking weight, punishing Pacman for eating food.
*   **The Fix:** Refactored the state extractor to use a **hybrid** design:
    *   **Linear Manhattan Distance** for global objectives (closest food/capsule distance divided by board area). When food is eaten, the linear increase in distance creates a negligible drop in Q-value, which is easily offset by the positive food reward.
    *   **Non-Linear Inverse Distance** for local events (active/scared ghost proximity), where threat/chasing gradients must be highly localized.

### 4. Narrow Corridor Danger Feature
*   **The Problem:** Approximate Q-agents only plan one step ahead, making them highly susceptible to ghost traps in dead-ends or tight corners.
*   **The Fix:** Implemented a `narrow-corridor-danger` feature that activates if the chosen action places Pacman in a corridor or corner (where `len(legal_actions) <= 2`) and an active ghost is within 3 steps. The agent successfully learns a massive negative weight for this feature, teaching Pacman to steer clear of narrow dead-ends when active ghosts are nearby.

---

## Performance Results (`smallClassic` Layout)

Evaluated over 50 test games on the `smallClassic` layout (featuring 4 active ghosts and 2 power capsules):

| Agent Configuration | Win Rate | Average Score | Average Steps | Key Converged Weight Settings |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline** (Stale Bootstrapping) | **2.00%** | `48.26` | `84.94` | `closest-food`: `+84.43`, `inv-active-ghost`: `+404.07` |
| **Stage 1 Fix** (Bootstrap Restored) | **38.00%** | `652.08` | `106.12` | `closest-food`: `+95.76`, `inv-active-ghost`: `-105.54` |
| **Stage 2 Optimized** (Decay + Corridor Danger) | **50.00%** | **`997.04`** | **`94.36`** | `closest-food`: `+48.99`, `narrow-corridor-danger`: `-971.53` |

---

## How to Run & Test

### Prerequisites
*   Python 3.7+
*   Pip dependencies: `fastapi`, `uvicorn`

### Run Backend Server
Start the local FastAPI application from the project root:
```bash
python backend/server.py
```
By default, the server runs on `http://127.0.0.1:8000`.

### Run Frontend Client
Open the web dashboard in your browser by opening `frontend/index.html`. You can select layouts, adjust hyperparameters, train the model in the background, and load saved weights directly inside the visual Testing Arena.

### Running Automated Evaluation
To evaluate weights locally in python without running the server:
```bash
# To train and test
python scratch/train.py
```
