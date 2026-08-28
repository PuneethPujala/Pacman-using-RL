from __future__ import print_function

import os.path
import random
import pickle
from builtins import object, range

try:
    import busters
    import inference
    from distanceCalculator import Distancer
except ImportError:
    busters = None
    inference = type('InferenceModuleMock', (), {'InferenceModule': object})
    Distancer = None

import util
from game import Agent, Directions
from keyboardAgents import KeyboardAgent
from featureExtractors import *
from game import Actions

# ... [original comments and licensing info remain unchanged] ...

# ================ Keep all original classes unchanged ================
class NullGraphics(object):
    def initialize(self, state, isBlue=False): pass
    def update(self, state): pass
    def pause(self): pass
    def draw(self, state): pass
    def updateDistributions(self, dist): pass
    def finish(self): pass

class KeyboardInference(inference.InferenceModule):
    def initializeUniformly(self, gameState):
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState): pass
    def getBeliefDistribution(self): return self.beliefs

class BustersAgent(object):
    def __init__(self, index=0, inference="ExactInference", ghostAgents=None, observeEnable=True, elapseTimeEnable=True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable

    def registerInitialState(self, gameState):
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True

    def observationFunction(self, gameState):
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
        return gameState

    def getAction(self, gameState):
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        return Directions.STOP

class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    def __init__(self, index=0, inference="KeyboardInference", ghostAgents=None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        return KeyboardAgent.getAction(self, gameState)

class RandomPAgent(BustersAgent):
    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def countFood(self, gameState):
        return sum(sum(row) for row in gameState.data.food)

    def printGrid(self, gameState):
        table = ""
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table += gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        return table[:-1]

    def chooseAction(self, gameState):
        legal = gameState.getLegalActions(0)
        if not legal: return Directions.STOP
        return random.choice(legal)

class GreedyBustersAgent(BustersAgent):
    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        pacmanPosition = gameState.getPacmanPosition()
        legal = gameState.getLegalPacmanActions()
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = [
            beliefs for i, beliefs in enumerate(self.ghostBeliefs) if livingGhosts[i+1]
        ]

        # Find most likely position of each living ghost
        ghostPositions = []
        for beliefs in livingGhostPositionDistributions:
            if beliefs.totalCount() == 0:
                continue
            ghostPositions.append(beliefs.argMax())

        if not ghostPositions:
            return random.choice(legal) if legal else Directions.STOP

        # Find closest ghost (by maze distance)
        closestGhost = min(ghostPositions, key=lambda g: self.distancer.getDistance(pacmanPosition, g))

        # Choose action that minimizes distance to closest ghost
        bestAction = legal[0]
        bestDist = float('inf')
        for action in legal:
            successor = Actions.getSuccessor(pacmanPosition, action)
            dist = self.distancer.getDistance(successor, closestGhost)
            if dist < bestDist:
                bestDist = dist
                bestAction = action
        return bestAction

class BasicAgentAA(BustersAgent):
    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0

    def countFood(self, gameState):
        return sum(sum(row) for row in gameState.data.food)

    def printGrid(self, gameState):
        table = ""
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table += gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        return table[:-1]

    def printInfo(self, gameState):
        print("---------------- TICK ", self.countActions, " --------------------------")
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print("Width: ", width, " Height: ", height)
        print("Pacman position: ", gameState.getPacmanPosition())
        print("Legal actions: ", gameState.getLegalPacmanActions())
        print("Pacman direction: ", gameState.data.agentStates[0].getDirection())
        print("Number of ghosts: ", gameState.getNumAgents() - 1)
        print("Living ghosts: ", gameState.getLivingGhosts())
        print("Ghosts positions: ", gameState.getGhostPositions())
        print("Ghosts directions: ", [gameState.getGhostDirections().get(i) for i in range(gameState.getNumAgents() - 1)])
        print("Ghosts distances: ", gameState.data.ghostDistances)
        print("Pac dots: ", gameState.getNumFood())
        print("Distance nearest pac dots: ", gameState.getDistanceNearestFood())
        print("Map:\n", gameState.getWalls())
        print("Score: ", gameState.getScore())

    def chooseAction(self, gameState):
        self.countActions += 1
        self.printInfo(gameState)
        legal = gameState.getLegalActions(0)
        if not legal:
            return Directions.STOP

        # Fallback to random if something fails
        try:
            distancer = self.distancer
            pacmanPos = gameState.getPacmanPosition()
            ghostPositions = [g for i, g in enumerate(gameState.getGhostPositions()) if gameState.getLivingGhosts()[i+1]]
            if not ghostPositions:
                return random.choice(legal)

            # Find closest ghost
            distances = [distancer.getDistance(pacmanPos, g) for g in ghostPositions]
            closestGhost = ghostPositions[distances.index(min(distances))]

            # Try to move toward it
            bestAction = legal[0]
            bestDist = float('inf')
            for action in legal:
                successor = Actions.getSuccessor(pacmanPos, action)
                dist = distancer.getDistance(successor, closestGhost)
                if dist < bestDist:
                    bestDist = dist
                    bestAction = action
            return bestAction
        except:
            return random.choice(legal)

    def printLineData(self, gameState):
        return "XXXXXXXXXX"


from collections import defaultdict
import pickle
import os
import random

class QLearningAgent(Agent):
    """
    Q-Learning Agent for classic Pacman (not Busters).
    Uses dictionary-based Q-values and robust state representation.
    """

    def __init__(self, epsilon=0.3, alpha=0.5, discount=0.8, numTraining=0, **args):
        # Parse parameters (from command line strings)
        self.epsilon = float(epsilon)
        self.initial_alpha = float(alpha)
        self.discount = float(discount)
        self.numTraining = int(numTraining)
        
        # Initialize state
        self.episodesSoFar = 0
        self.alpha = self.initial_alpha  # will be decayed in update()
        
        # Q-table (for tabular version)
        self.q_values = defaultdict(float)
        self.q_file = "qvalues.pkl"
        self.readQvalues()

    def getStateKey(self, state):
        pacman_pos = state.getPacmanPosition()
        ghost_states = state.getGhostStates()
        ghost_positions = tuple(g.getPosition() for g in ghost_states)
        scared_timers = tuple(g.scaredTimer for g in ghost_states)
        num_food = state.getNumFood()
        capsules = tuple(state.getCapsules())
        return (pacman_pos, ghost_positions, scared_timers, num_food, capsules)

    def getQValue(self, state, action):
        return self.q_values[(self.getStateKey(state), action)]

    def computeValueFromQValues(self, state):
        legalActions = [a for a in state.getLegalPacmanActions() if a != 'Stop']
        if not legalActions:
            return 0.0
        return max(self.getQValue(state, a) for a in legalActions)

    def computeActionFromQValues(self, state):
        legalActions = [a for a in state.getLegalPacmanActions() if a != 'Stop']
        if not legalActions:
            return 'Stop'
        best_action = None
        best_value = float('-inf')
        for action in legalActions:
            value = self.getQValue(state, action)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action

    def registerInitialState(self, state):
        self.last_state = None
        self.last_action = None

    def observationFunction(self, state):
        if self.last_state is not None and self.last_action is not None:
            reward = self.getReward(self.last_state, self.last_action, state)
            self.update(self.last_state, self.last_action, state, reward)
        return state

    def getAction(self, state):
        legalActions = [a for a in state.getLegalPacmanActions() if a != 'Stop']
        if not legalActions:
            return 'Stop'

        # Decay epsilon during training
        if self.episodesSoFar < self.numTraining:
            fraction = self.episodesSoFar / float(self.numTraining)
            current_epsilon = self.epsilon * (1.0 - fraction)
            current_epsilon = max(current_epsilon, 0.01)
        else:
            current_epsilon = 0.0

        if util.flipCoin(current_epsilon):
            action = random.choice(legalActions)
        else:
            action = self.getPolicy(state)

        self.last_state = state
        self.last_action = action
        return action

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def getReward(self, state, action, nextState):
        reward = nextState.getScore() - state.getScore()

        if nextState.getNumFood() < state.getNumFood():
            reward += 10
        if len(nextState.getCapsules()) < len(state.getCapsules()):
            reward += 50
        if nextState.isWin():
            reward += 500
        if nextState.isLose():
            reward -= 500

        ghost_states = nextState.getGhostStates()
        pacman_pos = nextState.getPacmanPosition()
        non_scared_ghosts = [g for g in ghost_states if g.scaredTimer == 0]
        if non_scared_ghosts:
            min_dist = min(util.manhattanDistance(pacman_pos, g.getPosition()) for g in non_scared_ghosts)
            if min_dist <= 1:
                reward -= 100
            elif min_dist == 2:
                reward -= 50
            elif min_dist == 3:
                reward -= 20
            elif min_dist == 4:
                reward -= 5

        scared_ghosts = [g for g in ghost_states if g.scaredTimer > 0]
        if scared_ghosts:
            old_dists = [util.manhattanDistance(state.getPacmanPosition(), g.getPosition()) for g in scared_ghosts]
            new_dists = [util.manhattanDistance(pacman_pos, g.getPosition()) for g in scared_ghosts]
            if min(new_dists) < min(old_dists):
                reward += 10

        return reward

    def update(self, state, action, nextState, reward):
        # Decay alpha during training
        if self.episodesSoFar < self.numTraining:
            fraction = self.episodesSoFar / float(self.numTraining)
            self.alpha = self.initial_alpha * (1.0 - fraction)
        else:
            self.alpha = 0.0

        target = reward + self.discount * self.getValue(nextState)
        self.q_values[(self.getStateKey(state), action)] += self.alpha * (target - self.getQValue(state, action))

    def readQvalues(self):
        if os.path.exists(self.q_file):
            try:
                with open(self.q_file, 'rb') as f:
                    loaded = pickle.load(f)
                    self.q_values = defaultdict(float, loaded)
            except Exception as e:
                print(f"Warning: Could not load Q-values: {e}")
                self.q_values = defaultdict(float)
        else:
            self.q_values = defaultdict(float)

    def writeQvalues(self):
        try:
            with open(self.q_file, 'wb') as f:
                pickle.dump(dict(self.q_values), f)
        except Exception as e:
            print(f"Warning: Could not save Q-values: {e}")

    def final(self, state):
        if self.last_state is not None and self.last_action is not None:
            reward = self.getReward(self.last_state, self.last_action, state)
            self.update(self.last_state, self.last_action, state, reward)
        self.last_state = None
        self.last_action = None

        self.episodesSoFar += 1
        if self.episodesSoFar >= self.numTraining:
            self.epsilon = 0.0
        self.writeQvalues()


class ApproximateQAgent(QLearningAgent):
    def __init__(self, extractor='SimpleExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        QLearningAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getQValue(self, state, action):
        features = self.featExtractor.getFeatures(state, action)
        return sum(self.weights[f] * features[f] for f in features)

    def update(self, state, action, nextState, reward):
        # 🔥 DECAY ALPHA HERE (overrides QLearningAgent.update)
        if self.episodesSoFar < self.numTraining:
            fraction = self.episodesSoFar / float(self.numTraining)
            self.alpha = self.initial_alpha * (1.0 - fraction)
        else:
            self.alpha = 0.01

        features = self.featExtractor.getFeatures(state, action)
        difference = (reward + self.discount * self.getValue(nextState)) - self.getQValue(state, action)
        for f in features:
            self.weights[f] += self.alpha * difference * features[f]

    def final(self, state):
        QLearningAgent.final(self, state)
        if self.episodesSoFar == self.numTraining:
            print("\nFinal trained weights:")
            for f, w in self.weights.items():
                print(f"  {f}: {w:.4f}")
        
    def registerInitialState(self, state):
        QLearningAgent.registerInitialState(self, state)
        import __main__
        if '__main__' in dir(__main__) and hasattr(__main__, '_distancer'):
            self.distancer = __main__._distancer
        else:
            try:
                from distanceCalculator import Distancer
                self.distancer = Distancer(state.data.layout, True)
            except ImportError:
                self.distancer = None