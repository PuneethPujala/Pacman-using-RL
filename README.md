# Pacman Reinforcement Learning (Approximate Q-Learning with Human-Like Vision)

This repository contains an advanced implementation of **Pacman Reinforcement Learning** using **Approximate Q-Learning** with a **Human-Like Vision & Topology Extractor (`HumanVisionExtractor`)**, a standalone interactive HTML5 web visualizer, and test suite.

By integrating **Cardinal Raycasting**, **Turn-Aware Dead-End Analysis**, and **Multi-Step Maze Distance Gradients**, the agent achieved an **86% Win Rate** on `mediumClassic` and **76% Win Rate** on `smallClassic`.

---

## Performance Scorecard

| Layout | Episodes | Win Rate | Average Score | Peak Score | Key Converged Weight Settings |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`smallClassic`** (20×7) | 2,000 | **76.00%** | `+767.33` | `+1,545.00` | `ghost-1step`: `-573.31`, `dead-end`: `-291.45`, `closest-scared`: `+754.09` |
| **`mediumClassic`** (20×11) | 2,000 | **86.00%** | `+1,360.27` | `+1,913.00` | `ghost-1step`: `-624.94`, `ghost-2step`: `-337.79`, `eats-capsule`: `+102.30` |

---

## Key AI & Vision Features (`HumanVisionExtractor`)

```
                          ┌──────────────────────────┐
                          │   HumanVisionExtractor   │
                          └─────────────┬────────────┘
                                        │
      ┌───────────────────────┬─────────┴───────────────┬────────────────────────┐
      ▼                       ▼                         ▼                        ▼
 Cardinal Raycast      Escape Topology           Maze Danger Gradient    Objectives & Hunting
 • Line-of-sight sight • Differentiates L-turns  • BFS 1-step, 2-step,   • Safe dot eating
 • Approaching vs.       from true cul-de-sacs     3-step warnings         • Power pellet focus
   retreating vectors  • Ghost-sealed traps      • Corner threat sensing • Scared ghost hunt
```

### 1. Cardinal Raycasting & Trajectory Matching (`castRay`)
* Casts visual sight rays in 4 cardinal directions ($N, S, E, W$) until hitting a wall.
* Checks ghost velocity vectors:
  * **Approaching Ghosts** (`ghost-visible-approaching = 1.0 / dist`): Triggers immediate retreat when an active ghost enters the same corridor heading toward Pacman.
  * **Retreating Ghosts** (`ghost-visible-retreating = 1.0 / dist`): Near-neutral penalty for ghosts moving away.
  * **Scared Ghosts** (`scared-ghost-visible = 1.0 / dist`): High-reward attraction to hunt down edible ghosts during power mode.

### 2. Turn-Aware Dead-End & Escape Analysis (`isTrappedInDeadEnd`)
* Performs a bounded BFS on the reachable maze subgraph starting at successor $(x', y')$, avoiding ghost threat zones.
* **Distinguishes L-turns from true cul-de-sacs**: Open corridors that bend/turn but eventually connect to multi-way junctions are classified as **safe escape paths** (`trapped-in-dead-end = 0.0`).
* **Sealed Cul-de-Sacs**: Closed pockets with 0 alternative exits blocked by an approaching ghost are penalized heavily (`trapped-in-dead-end = 1.0`, weight $\approx -291$).

### 3. Multi-Step Danger Gradient (BFS Maze Distances)
* Calculates true shortest path distances around blind corners:
  * `ghost-1-step-away`: Massive penalty ($-573$ to $-624$) preventing immediate collisions.
  * `ghost-2-steps-away`: Active 2-step avoidance ($-214$ to $-337$).
  * `ghost-3-steps-away`: Early warning gradient ($-180$ to $-202$).

---

## Converged Weights Overview

```text
Final Converged Weights (mediumClassic):
  ghost-1-step-away:           -624.94   (Collision avoidance)
  ghost-2-steps-away:          -337.79   (High-priority 2-step avoidance)
  trapped-in-dead-end:         -285.62   (Dead-end pocket penalty)
  ghost-3-steps-away:          -202.84   (3-step early warning)
  closest-scared-ghost:        +315.66   (Aggressive hunting in power mode)
  eats-capsule:                +102.30   (High priority for power pellets)
  scared-ghost-visible:         +97.29   (Hallway chase reward)
  ghost-visible-approaching:    -76.51   (Line-of-sight hallway penalty)
  eats-food:                    +74.83   (Dot clearing reward)
  closest-food:                 -55.51   (Food proximity attraction)
  bias:                         +40.86   (Baseline bias)
  ghost-visible-retreating:      +1.04   (Neutral for ghosts moving away)
```

---

## Project Structure

```text
├── featureExtractors.py      # HumanVisionExtractor, RaycastExtractor, SimpleExtractor
├── bustersAgents.py          # ApproximateQAgent, QLearningAgent with transition updates
├── test_vision_extractor.py  # Unit test suite verifying raycasting, L-turns, & Q-values
├── index.html                # Standalone interactive HTML5 visualizer (GitHub Pages ready)
├── pacman.py                 # Core Berkeley Pacman engine
├── game.py                   # Game state, grid rules, and collision mechanics
├── ghostAgents.py            # RandomGhost and DirectionalGhost AI
├── layouts/                  # smallClassic, mediumClassic, trickyClassic, etc.
└── backend/ & frontend/      # Optional FastAPI server & dashboard
```

---

## How to Run & Test

### 1. Run the Automated Unit Test Suite
Verify raycasting, L-turns vs cul-de-sacs, and $Q(\text{Backward}) > Q(\text{Forward})$ kiting:
```bash
python test_vision_extractor.py
```

### 2. Train & Evaluate with ApproximateQAgent
```bash
# Train on smallClassic for 2000 episodes + 100 evaluation games:
python pacman.py -p ApproximateQAgent -a extractor=HumanVisionExtractor -x 2000 -n 2100 -l smallClassic -q

# Train on mediumClassic for 2000 episodes + 100 evaluation games:
python pacman.py -p ApproximateQAgent -a extractor=HumanVisionExtractor -x 2000 -n 2100 -l mediumClassic -q
```

### 3. Watch Live Desktop GUI Game Window
```bash
python pacman.py -p ApproximateQAgent -a extractor=HumanVisionExtractor -x 2000 -n 2005 -l smallClassic --frameTime 0.1
```

### 4. Interactive Web Visualizer (`index.html`)
Open [`index.html`](index.html) in any web browser to see the live HTML5 simulation with raycasting lasers, dynamic Q-value meters, and test scenario controls. Deployable directly via **GitHub Pages**.

