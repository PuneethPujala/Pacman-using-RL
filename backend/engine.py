"""
Engine wrapper for Pacman RL — bridges existing game logic with the web API.
Does NOT modify any original Pacman files.

Key facts about the base classes (from bustersAgents.py):
  - QLearningAgent.__init__ opens/creates qtable.txt relative to CWD — we
    must os.chdir(PACMAN_DIR) before constructing any agent.
  - QLearningAgent.final() increments episodesSoFar and sets epsilon=0 once
    episodesSoFar >= numTraining. There is NO gradual decay — it's a hard
    flip to 0 at the end of training.
  - ApproximateQAgent.final() does NOT call super().final() — so
    episodesSoFar is NEVER incremented and epsilon NEVER becomes 0 unless we
    fix it. We fix this in SavedApproximateQAgent.final().
  - getAction() does NOT call update() internally — the training loop must
    call agent.update(state, action, nextState, reward) explicitly each step.
  - registerInitialState() belongs to BustersAgent which requires
    __main__._display to be set — it is NOT safe to call from the web server.
    We skip it and manage weight loading directly.
"""
import sys
import os
import threading
import copy

PACMAN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PACMAN_DIR not in sys.path:
    sys.path.insert(0, PACMAN_DIR)

import layout as layout_module
from game import Directions, Agent
from pacman import GameState, ClassicGameRules
from ghostAgents import RandomGhost, DirectionalGhost
import json
import util
from bustersAgents import ApproximateQAgent
from featureExtractors import SimpleExtractor

# Absolute path for weights — consistent regardless of CWD.
WEIGHTS_FILE = os.path.join(PACMAN_DIR, "weights.json")


# ---------------------------------------------------------------------------
# PacmanExtractor — 6 hand-crafted features
# ---------------------------------------------------------------------------
# Features:
#   1. eats-food            : 1 if Pacman lands on a food tile this step
#   2. closest-food         : 1 / dist_to_nearest_food  (0 if no food left)
#   3. active-ghost-1-step  : 1 if a non-scared ghost is exactly 1 step away
#   4. active-ghost-2-step  : 1 if a non-scared ghost is exactly 2 steps away
#   5. scared-ghost-1-step  : 1 if a scared ghost is exactly 1 step away
#   6. scared-ghost-2-step  : 1 if a scared ghost is exactly 2 steps away
#
# Note on eats-food: SimpleExtractor checks food on the *successor* state so
# the tile is already consumed and the flag is always 0. We check the
# *pre-move* (current) state food grid instead.

# ---------------------------------------------------------------------------
# SmartExtractor — Advanced features for mediumClassic
# ---------------------------------------------------------------------------
# Features:
#   1. eats-food            : 1 if eating food (essential reward signal)
#   2. eats-capsule         : 1 if eating capsule
#   3. closest-food         : 1 / dist (normalized)
#   4. inv-active-ghost     : 1 / dist to closest active ghost (avoidance gradient)
#   5. inv-scared-ghost     : 1 / dist to closest scared ghost (chase gradient)
#   6. bias                 : 1.0

class SmartExtractor(SimpleExtractor):
    def getFeatures(self, state, action):
        from util import manhattanDistance
        feats = util.Counter()
        feats["bias"] = 1.0

        successor = state.generateSuccessor(0, action)
        pacPos    = successor.getPacmanPosition()

        # ---- 1. Eats Food (pre-move check) ----
        # Check if food existed at new position in OLD state
        if state.getFood()[int(pacPos[0])][int(pacPos[1])]:
            feats["eats-food"] = 1.0

        # ---- 2. Closest Food (inverse Manhattan distance) ----
        foodList = successor.getFood().asList()
        if foodList:
            minDist = min([manhattanDistance(pacPos, f) for f in foodList])
            if minDist > 0:
                feats["closest-food"] = float(1.0 / minDist)
            else:
                feats["closest-food"] = 1.0 
        
        # ---- 3. Capsules ----
        # Eats capsule?
        if pacPos in state.getCapsules():
            feats["eats-capsule"] = 1.0
            
        # ---- 4. Ghosts (inverse Manhattan distance) ----
        ghosts = successor.getGhostStates()
        for ghost in ghosts:
            dist = manhattanDistance(pacPos, ghost.getPosition())
            if ghost.scaredTimer > 0:
                # Scared ghost: chase!
                if dist == 0: 
                    feats["eats-ghost"] = 1.0
                else: 
                    feats["inv-scared-ghost"] += 1.0 / dist
            else:
                # Active ghost: avoid!
                if dist <= 1:
                    feats["collision-ghost"] = 1.0
                else:
                    feats["inv-active-ghost"] += 1.0 / dist

        # ---- 5. Narrow corridor danger ----
        # If Pacman enters a corridor/dead-end (at most 1 movement option besides STOP)
        # and an active ghost is nearby (Manhattan distance <= 3), it is a dangerous trap.
        legal_actions = successor.getLegalActions(0)
        if len(legal_actions) <= 2:
            active_ghosts = [g for g in successor.getGhostStates() if g.scaredTimer == 0]
            if active_ghosts:
                min_ghost_dist = min([manhattanDistance(pacPos, g.getPosition()) for g in active_ghosts])
                if min_ghost_dist <= 3:
                    feats["narrow-corridor-danger"] = 1.0
                    
        return feats


# ---------------------------------------------------------------------------
# SavedApproximateQAgent
# ---------------------------------------------------------------------------

class SavedApproximateQAgent(ApproximateQAgent):
    """
    ApproximateQAgent with:
      - PacmanExtractor (6 features, fixes eats-food bug)
      - JSON weight persistence at an absolute path
      - Correct episodesSoFar / epsilon bookkeeping via fixed final()

    TRAINING mode : epsilon > 0, alpha > 0, numTraining == total episodes.
    TESTING  mode : epsilon = 0, alpha = 0, numTraining = 0.
    """

    def __init__(self, **args):
        # Must be in PACMAN_DIR when calling super().__init__() because
        # QLearningAgent.__init__ opens qtable.txt relative to CWD.
        super().__init__(extractor='SimpleExtractor', **args)
        # Override with our advanced SmartExtractor
        self.featExtractor = SmartExtractor()
        self.weights_file  = WEIGHTS_FILE
        self.initial_alpha = self.alpha
        self.initial_epsilon = self.epsilon

    # ------------------------------------------------------------------
    # Weight I/O
    # ------------------------------------------------------------------

    def loadWeights(self):
        """Load weights from JSON; returns True on success."""
        if os.path.exists(self.weights_file):
            try:
                with open(self.weights_file, 'r') as f:
                    data = json.load(f)
                self.weights = util.Counter()
                for k, v in data.items():
                    self.weights[str(k)] = float(v)
                print(f"[Agent] Loaded weights: {dict(self.weights)}")
                return True
            except Exception as e:
                print(f"[Agent] Error loading weights: {e}")
        else:
            print(f"[Agent] No weights file at {self.weights_file} — starting fresh.")
        return False

    def saveWeights(self):
        """Persist weights to JSON."""
        try:
            with open(self.weights_file, 'w') as f:
                json.dump(dict(self.weights), f, indent=2)
            print(f"[Agent] Saved weights: {dict(self.weights)}")
        except Exception as e:
            print(f"[Agent] Error saving weights: {e}")

    # ------------------------------------------------------------------
    # final() — THE critical fix
    # ------------------------------------------------------------------

    def final(self, state):
        """
        End-of-episode hook.

        THE BUG: ApproximateQAgent.final() does NOT call super().final(), so
        QLearningAgent.final() (which does episodesSoFar += 1 and the epsilon
        flip) never runs. Epsilon stays at its initial value forever.

        THE FIX: call ApproximateQAgent.final() for its weight-printing side
        effect, then manually run the QLearningAgent.final() logic ourselves.
        """
        # Step 1: ApproximateQAgent.final() — prints weights when done.
        ApproximateQAgent.final(self, state)

        # Step 2: QLearningAgent.final() logic (skipped by the base class).
        self.episodesSoFar += 1
        
        # Exponentially decay epsilon and alpha to stabilize training
        if self.numTraining > 0:
            decay_factor = 0.985 ** self.episodesSoFar
            self.epsilon = self.initial_epsilon * decay_factor
            self.alpha = self.initial_alpha * decay_factor

        if self.episodesSoFar >= self.numTraining:
            self.epsilon = 0.0   # pure exploitation after training ends
            self.alpha = 0.0

        # Step 3: Persist weights every episode so crashes don't lose progress.
        self.saveWeights()


# ---------------------------------------------------------------------------
# state_to_snapshot
# ---------------------------------------------------------------------------

def state_to_snapshot(game_state, step_num=0, q_values=None, visit_counts=None):
    """Convert a GameState into a JSON-serialisable dict."""
    walls = game_state.getWalls()
    food  = game_state.getFood()
    width, height = walls.width, walls.height

    walls_grid = [[bool(walls[x][y]) for y in range(height)] for x in range(width)]
    food_grid  = [[bool(food[x][y])  for y in range(height)] for x in range(width)]

    pac_pos = game_state.getPacmanPosition()
    pac_dir = game_state.data.agentStates[0].getDirection()

    ghosts = []
    for gs in game_state.getGhostStates():
        gpos = gs.getPosition()
        ghosts.append({
            "x": gpos[0], "y": gpos[1],
            "direction": str(gs.getDirection()),
            "scared": gs.scaredTimer > 0,
            "scaredTimer": gs.scaredTimer,
        })

    capsules = [[int(c[0]), int(c[1])] for c in game_state.getCapsules()]

    snapshot = {
        "width": width, "height": height,
        "walls": walls_grid, "food": food_grid, "capsules": capsules,
        "pacman": {"x": pac_pos[0], "y": pac_pos[1], "direction": str(pac_dir)},
        "ghosts": ghosts,
        "score":   game_state.getScore(),
        "isWin":   game_state.isWin(),
        "isLose":  game_state.isLose(),
        "step":    step_num,
        "numFood": game_state.getNumFood(),
    }
    if q_values     is not None: snapshot["qValues"]     = q_values
    if visit_counts is not None: snapshot["visitCounts"] = visit_counts
    return snapshot


# ---------------------------------------------------------------------------
# GameEngine — test / visualisation (no learning)
# ---------------------------------------------------------------------------

class GameEngine:
    """
    Runs a Pacman game step-by-step for web visualisation.
    Agent uses epsilon=0, alpha=0, numTraining=0: pure exploitation of saved weights.
    """

    def __init__(self):
        self.game_state   = None
        self.ghost_agents = []
        self.pacman_agent = None
        self.step_count   = 0
        self.game_over    = False
        self.move_history = []
        self.visit_counts = None
        self._layout_name = None
        self._layout_cfg  = {}

    def start_game(self, layout_name="mediumClassic", ghost_type="random",
                   num_ghosts=4, model_path=None):
        """Initialise a new test game and load trained weights."""
        original_dir = os.getcwd()
        try:
            os.chdir(PACMAN_DIR)

            lay = layout_module.getLayout(layout_name)
            if lay is None:
                raise ValueError(f"Layout '{layout_name}' not found")

            self._layout_name = layout_name
            self._layout_cfg  = {"ghost_type": ghost_type,
                                  "num_ghosts": num_ghosts,
                                  "model_path": model_path}

            num_g = min(num_ghosts, lay.getNumGhosts())
            if ghost_type == "directional":
                self.ghost_agents = [DirectionalGhost(i + 1) for i in range(num_g)]
            else:
                self.ghost_agents = [RandomGhost(i + 1) for i in range(num_g)]

            # Pure exploitation — no learning during testing.
            self.pacman_agent = SavedApproximateQAgent(
                epsilon=0.0, alpha=0.0, discount=0.8, numTraining=0
            )

            if model_path:
                self.pacman_agent.weights_file = model_path

            # Load weights directly (no registerInitialState — unsafe without
            # a full Pacman game context and __main__._display).
            self.pacman_agent.loadWeights()

            self.game_state = GameState()
            self.game_state.initialize(lay, num_g)

            self.step_count   = 0
            self.game_over    = False
            self.move_history = []

            self.visit_counts = [[0] * lay.height for _ in range(lay.width)]
            px, py = self.game_state.getPacmanPosition()
            self.visit_counts[int(px)][int(py)] += 1

        finally:
            os.chdir(original_dir)

        return self.get_snapshot()

    def step(self):
        """Execute one full game step (Pacman + all ghosts)."""
        if self.game_over or self.game_state is None:
            return self.get_snapshot()

        # Pacman move
        try:
            pac_action = self.pacman_agent.getAction(self.game_state.deepCopy())
        except Exception:
            pac_action = Directions.STOP
        if pac_action is None:
            pac_action = Directions.STOP

        try:
            self.game_state = self.game_state.generateSuccessor(0, pac_action)
        except Exception:
            self.game_over = True
            return self.get_snapshot()

        self.move_history.append({"agent": 0, "action": str(pac_action)})
        px, py = self.game_state.getPacmanPosition()
        self.visit_counts[int(px)][int(py)] += 1

        if self.game_state.isWin() or self.game_state.isLose():
            self.game_over = True
            self.step_count += 1
            return self.get_snapshot()

        # Ghost moves
        for i, ghost in enumerate(self.ghost_agents):
            gi = i + 1
            if gi >= self.game_state.getNumAgents():
                break
            try:
                ga = ghost.getAction(self.game_state)
                self.game_state = self.game_state.generateSuccessor(gi, ga)
                self.move_history.append({"agent": gi, "action": str(ga)})
            except Exception:
                pass
            if self.game_state.isWin() or self.game_state.isLose():
                self.game_over = True
                break

        self.step_count += 1
        return self.get_snapshot()

    def get_snapshot(self):
        if self.game_state is None:
            return {"error": "No game initialised"}

        q_vals = None
        try:
            legal = self.game_state.getLegalActions(0)
            if legal:
                q_vals = {
                    str(a): float(self.pacman_agent.getQValue(self.game_state, a))
                    for a in legal
                }
        except Exception:
            pass

        return state_to_snapshot(self.game_state, self.step_count,
                                  q_vals, self.visit_counts)

    def reset(self):
        if self._layout_name:
            cfg = self._layout_cfg
            return self.start_game(self._layout_name,
                                   ghost_type=cfg.get("ghost_type", "random"),
                                   num_ghosts=cfg.get("num_ghosts", 4),
                                   model_path=cfg.get("model_path"))
        return {"error": "No layout previously loaded"}


# ---------------------------------------------------------------------------
# TrainingRunner
# ---------------------------------------------------------------------------

class TrainingRunner:
    """
    Runs RL training in a background thread.

    Weight-learning contract
    ------------------------
    1. One SavedApproximateQAgent for the entire run — weights accumulate.
    2. Weights always start at zero — no checkpoint loading.
    3. agent.update(state, action, nextState, reward) called explicitly after
       every Pacman step — getAction() does NOT update weights internally.
    4. Shaped reward per step:
         +10  eating food, +50 capsule, +500 win, -500 death, -1 per step.
    5. Step limit = width * height * 4 so agent has time to actually win.
    6. agent.final() called each episode: episodesSoFar++, saves weights.
       Then linear epsilon decay is applied: initial → 0.05 over all episodes.
    Recommended hyperparameters: alpha=0.2, gamma=0.9, epsilon=0.5
    """

    def __init__(self):
        self.is_training = False
        self.metrics     = self._blank_metrics(0)
        self._thread     = None
        self._lock       = threading.Lock()

    @staticmethod
    def _blank_metrics(total):
        return {
            "episode_rewards":   [],
            "episode_scores":    [],
            "win_rate_history":  [],
            "epsilon_history":   [],
            "steps_per_episode": [],
            "current_episode":   0,
            "total_episodes":    total,
            "wins":   0,
            "losses": 0,
            "training_complete": False,
        }

    def start(self, episodes=100, alpha=0.5, gamma=0.8, epsilon=0.3,
              layout_name="mediumClassic", num_ghosts=4, ghost_type="random"):
        if self.is_training:
            return {"error": "Training already in progress"}

        # Delete any existing weights so training always starts from zero.
        if os.path.exists(WEIGHTS_FILE):
            os.remove(WEIGHTS_FILE)
            print("[TrainingRunner] Cleared existing weights — starting fresh.")

        with self._lock:
            self.metrics = self._blank_metrics(episodes)

        self._thread = threading.Thread(
            target=self._train_loop,
            args=(episodes, alpha, gamma, epsilon, layout_name, num_ghosts, ghost_type),
            daemon=True,
        )
        self.is_training = True
        self._thread.start()
        return {"status": "training_started", "episodes": episodes}

    def stop(self):
        self.is_training = False

    def _train_loop(self, episodes, alpha, gamma, epsilon,
                    layout_name, num_ghosts, ghost_type):
        original_dir = os.getcwd()
        try:
            os.chdir(PACMAN_DIR)

            lay = layout_module.getLayout(layout_name)
            if lay is None:
                print(f"[TrainingRunner] Layout '{layout_name}' not found.")
                with self._lock:
                    self.metrics["training_complete"] = True
                self.is_training = False
                return

            num_g = min(num_ghosts, lay.getNumGhosts())

            # Single agent — weights accumulate across all episodes.
            initial_epsilon = float(epsilon)
            epsilon_min     = 0.05   # never drop below 5% exploration
            agent = SavedApproximateQAgent(
                epsilon=initial_epsilon,
                alpha=float(alpha),
                discount=float(gamma),
                numTraining=int(episodes),
            )
            agent.weights_file = WEIGHTS_FILE
            # Weights start at zero — no checkpoint load here.

            for ep in range(episodes):
                if not self.is_training:
                    break

                ghosts = (
                    [DirectionalGhost(i + 1) for i in range(num_g)]
                    if ghost_type == "directional"
                    else [RandomGhost(i + 1) for i in range(num_g)]
                )

                game_state = GameState()
                game_state.initialize(lay, num_g)

                step_count   = 0
                total_reward = 0.0
                # Step limit scales with map size so the agent always has
                # enough time to actually win (smallClassic needs ~200 steps,
                # mediumClassic can need 500+).
                max_steps = lay.width * lay.height * 4

                while True:
                    # ---- Pacman turn ----
                    try:
                        pac_action = agent.getAction(game_state.deepCopy())
                        if pac_action is None:
                            pac_action = Directions.STOP
                    except Exception:
                        pac_action = Directions.STOP

                    try:
                        next_state = game_state.generateSuccessor(0, pac_action)
                    except Exception:
                        break

                    # ---- Shaped reward ----
                    # Raw score delta covers food (+10), ghost-eat (+200),
                    # death (-500), win (+500) automatically.
                    reward = next_state.getScore() - game_state.getScore()

                    # Bonus for eating a food pellet (encourages collection)
                    if next_state.getNumFood() < game_state.getNumFood():
                        reward += 10.0

                    # Bonus for eating a power capsule
                    if len(next_state.getCapsules()) < len(game_state.getCapsules()):
                        reward += 50.0

                    # Large terminal bonuses so the agent strongly
                    # prefers winning / strongly avoids dying
                    if next_state.isWin():
                        reward += 500.0
                    if next_state.isLose():
                        reward -= 500.0

                    # Small step penalty to discourage wandering
                    reward -= 1.0

                    total_reward += reward

                    # Explicit update — correct pattern for this codebase.
                    try:
                        agent.update(game_state, pac_action, next_state, reward)
                    except Exception as e:
                        print(f"[TrainingRunner] update() error: {e}")

                    game_state = next_state

                    if game_state.isWin() or game_state.isLose():
                        break

                    # ---- Ghost turns ----
                    for i, ghost in enumerate(ghosts):
                        gi = i + 1
                        if gi >= game_state.getNumAgents():
                            break
                        try:
                            ga = ghost.getAction(game_state)
                            game_state = game_state.generateSuccessor(gi, ga)
                        except Exception:
                            pass
                        if game_state.isWin() or game_state.isLose():
                            break

                    if game_state.isWin() or game_state.isLose():
                        break

                    step_count += 1
                    if step_count > max_steps:
                        break

                # ---- End of episode ----
                # final() handles episodesSoFar++ and weight saving.
                # We then apply linear epsilon decay ourselves — the base
                # class only does a hard flip to 0 at the very end.
                try:
                    agent.final(game_state)
                except Exception as e:
                    print(f"[TrainingRunner] final() error ep {ep}: {e}")

                # Linear decay: initial_epsilon → epsilon_min over all episodes.
                progress        = (ep + 1) / episodes
                current_epsilon = max(epsilon_min, initial_epsilon * (1.0 - progress))
                agent.epsilon   = current_epsilon   # takes effect next episode
                is_win = game_state.isWin()

                with self._lock:
                    self.metrics["episode_rewards"].append(round(total_reward, 2))
                    self.metrics["episode_scores"].append(round(game_state.getScore(), 2))
                    self.metrics["steps_per_episode"].append(step_count)
                    self.metrics["epsilon_history"].append(round(current_epsilon, 4))
                    self.metrics["current_episode"] = ep + 1
                    if is_win:
                        self.metrics["wins"] += 1
                    else:
                        self.metrics["losses"] += 1
                    total_games = self.metrics["wins"] + self.metrics["losses"]
                    self.metrics["win_rate_history"].append(
                        round(self.metrics["wins"] / total_games, 4)
                        if total_games else 0.0
                    )

        except Exception as e:
            import traceback
            print(f"[TrainingRunner] Fatal: {e}")
            traceback.print_exc()
        finally:
            os.chdir(original_dir)
            with self._lock:
                self.metrics["training_complete"] = True
            self.is_training = False

    def get_metrics(self):
        with self._lock:
            return copy.deepcopy(self.metrics)


# ---------------------------------------------------------------------------
# Layout utilities
# ---------------------------------------------------------------------------

def get_available_layouts():
    layouts_dir = os.path.join(PACMAN_DIR, "layouts")
    if not os.path.isdir(layouts_dir):
        return []
    return [f.replace(".lay", "")
            for f in sorted(os.listdir(layouts_dir))
            if f.endswith(".lay")]


def get_layout_preview(layout_name):
    path = os.path.join(PACMAN_DIR, "layouts", layout_name + ".lay")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return f.read()