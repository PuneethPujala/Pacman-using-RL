from __future__ import print_function

import os.path
from builtins import object, range

import busters
import inference
import numpy as np
import util
from game import Agent, Directions
from keyboardAgents import KeyboardAgent
from game import Agent
from featureExtractors import *

# bustersAgents.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).



class NullGraphics(object):
    "Placeholder for graphics"
    def initialize(self, state, isBlue = False):
        pass
    def update(self, state):
        pass
    def pause(self):
        pass
    def draw(self, state):
        pass
    def updateDistributions(self, dist):
        pass
    def finish(self):
        pass

class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """
    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
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

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent(object):
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__( self, index = 0, inference = "ExactInference", ghostAgents = None, observeEnable = True, elapseTimeEnable = True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True

    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
        return gameState

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        #for index, inf in enumerate(self.inferenceModules):
        #    if not self.firstMove and self.elapseTimeEnable:
        #        inf.elapseTime(gameState)
        #    self.firstMove = False
        #    if self.observeEnable:
        #        inf.observeState(gameState)
        #    self.ghostBeliefs[index] = inf.getBeliefDistribution()
        #self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP

class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(self, index = 0, inference = "KeyboardInference", ghostAgents = None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        return KeyboardAgent.getAction(self, gameState)

import random
import sys

from distanceCalculator import Distancer
from game import Actions, Directions

'''Random PacMan Agent'''
class RandomPAgent(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food

    ''' Print the layout'''
    def printGrid(self, gameState):
        table = ""
        ##print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def chooseAction(self, gameState):
        move = Directions.STOP
        legal = gameState.getLegalActions(0) ##Legal position from the pacman
        move_random = random.randint(0, 3)
        if   ( move_random == 0 ) and Directions.WEST in legal:  move = Directions.WEST
        if   ( move_random == 1 ) and Directions.EAST in legal: move = Directions.EAST
        if   ( move_random == 2 ) and Directions.NORTH in legal:   move = Directions.NORTH
        if   ( move_random == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move

class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (according to mazeDistance!).

        To find the mazeDistance between any two positions, use:
          self.distancer.getDistance(pos1, pos2)

        To find the successor position of a position after an action:
          successorPosition = Actions.getSuccessor(position, action)

        livingGhostPositionDistributions, defined below, is a list of
        util.Counter objects equal to the position belief
        distributions for each of the ghosts that are still alive.  It
        is defined based on (these are implementation details about
        which you need not be concerned):

          1) gameState.getLivingGhosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             gameState.getLivingGhosts() list.
        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i+1]]
        return Directions.EAST

class BasicAgentAA(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0

    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food

    ''' Print the layout'''
    def printGrid(self, gameState):
        table = ""
        #print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def printInfo(self, gameState):
        print("---------------- TICK ", self.countActions, " --------------------------")
        # Map size
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print("Width: ", width, " Height: ", height)
        # Pacman position
        print("Pacman position: ", gameState.getPacmanPosition())
        # Legal actions for Pacman in current position
        print("Legal actions: ", gameState.getLegalPacmanActions())
        # Pacman direction
        print("Pacman direction: ", gameState.data.agentStates[0].getDirection())
        # Number of ghosts
        print("Number of ghosts: ", gameState.getNumAgents() - 1)
        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        print("Living ghosts: ", gameState.getLivingGhosts())
        # Ghosts positions
        print("Ghosts positions: ", gameState.getGhostPositions())
        # Ghosts directions
        print("Ghosts directions: ", [gameState.getGhostDirections().get(i) for i in range(0, gameState.getNumAgents() - 1)])
        # Manhattan distance to ghosts
        print("Ghosts distances: ", gameState.data.ghostDistances)
        # Pending pac dots
        print("Pac dots: ", gameState.getNumFood())
        # Manhattan distance to the closest pac dot
        print("Distance nearest pac dots: ", gameState.getDistanceNearestFood())
        # Map walls
        print("Map:")
        print( gameState.getWalls())
        # Score
        print("Score: ", gameState.getScore())


    def chooseAction(self, gameState):
        self.countActions = self.countActions + 1
        self.printInfo(gameState)
        move = Directions.STOP
        legal = gameState.getLegalActions(0) ##Legal position from the pacman

        if f"{gameState.data.agentStates[0].getDirection()}" == "STOP":
            start = gameState.getPacmanPosition()

        realdistance=[]
        distancer = Distancer(gameState.data.layout)
        for ghost in gameState.getGhostPositions():
            realdistance.append(distancer.getDistance(gameState.getPacmanPosition(), ghost))
        index = realdistance.index(min(z for z in realdistance if z is not None))
        print("Ghosts real distances: ", realdistance)

        PacX = gameState.getPacmanPosition()[0]
        PacY = gameState.getPacmanPosition()[1]

        if len(gameState.getLegalPacmanActions())==1 :
            move = Directions.gameState.getLegalPacmanActions()[0]
        else:
            try :
                if distancer.getDistance((PacX-1, PacY), gameState.getGhostPositions()[index]) < realdistance[index] and (Directions.WEST in legal):
                    move = Directions.WEST
            except:
                None
            try :
                if distancer.getDistance((PacX+1, PacY), gameState.getGhostPositions()[index]) < realdistance[index] and (Directions.EAST in legal):
                    move = Directions.EAST
            except:
                None
            try :
                if distancer.getDistance((PacX, PacY+1), gameState.getGhostPositions()[index]) < realdistance[index] and (Directions.NORTH in legal):
                    move = Directions.NORTH
            except:
                None
            try :
                if distancer.getDistance((PacX, PacY-1), gameState.getGhostPositions()[index]) < realdistance[index] and (Directions.SOUTH in legal):
                    move = Directions.SOUTH
            except:
                None
        return move

    def printLineData(self, gameState):
        return "XXXXXXXXXX"


class QLearningAgent(Agent):
    def __init__(self, epsilon=0.3, alpha=0.5, discount=0.8, numTraining=0):
        self.epsilon = epsilon
        self.alpha = alpha
        self.discount = discount
        self.numTraining = numTraining
        self.episodesSoFar = 0
        self.actions = {"North":0, "East":1, "South":2, "West":3, "Stop":4}

        if os.path.exists("qtable.txt"):
            self.table_file = open("qtable.txt", "r+")
            self.q_table = self.readQtable()
        else:
            self.table_file = open("qtable.txt", "w+")
            self.initializeQtable(500)

            
    def initializeQtable(self, nrows):
        "Initialize qtable"
        self.q_table = np.zeros((nrows,len(self.actions)))

    def readQtable(self):
        "Read qtable from disc"
        table = self.table_file.readlines()
        q_table = []

        for i, line in enumerate(table):
            row = line.split()
            row = [float(x) for x in row]
            q_table.append(row)

        return q_table


    def writeQtable(self):
        self.table_file.seek(0)
        self.table_file.truncate()

        for line in self.q_table:
            for item in line:
                self.table_file.write(str(item)+" ")
            self.table_file.write("\n")

    def printQtable(self):
        "Print qtable"
        for line in self.q_table:
            print(line)
        print("\n")


    def __del__(self):
        "Destructor. Invokation at the end of each episode"
        self.writeQtable()
        self.table_file.close()


    def computePosition(self, state):
        """
        Compute the row of the qtable for a given state.
        Always return an integer index.
        """
        realdistance = []
        distancer = Distancer(state.data.layout)

        # Distance to each ghost
        for ghost in state.getGhostPositions():
            realdistance.append(distancer.getDistance(state.getPacmanPosition(), ghost))
        index = realdistance.index(min(z for z in realdistance if z is not None))

        PacX, PacY = state.getPacmanPosition()
        width, height = state.data.layout.width, state.data.layout.height
        longestdistance = width + height - 6

        pacmanstate = int(realdistance[index])  # ✅ default fallback

        try:
            if distancer.getDistance((PacX-1, PacY), state.getGhostPositions()[index]) < realdistance[index]:
                pacmanstate = int(realdistance[index])
        except:
            pass
        try:
            if distancer.getDistance((PacX+1, PacY), state.getGhostPositions()[index]) < realdistance[index]:
                pacmanstate = int(realdistance[index] + longestdistance)
        except:
            pass
        try:
            if distancer.getDistance((PacX, PacY+1), state.getGhostPositions()[index]) < realdistance[index]:
                pacmanstate = int(realdistance[index] + (2 * longestdistance))
        except:
            pass
        try:
            if distancer.getDistance((PacX, PacY-1), state.getGhostPositions()[index]) < realdistance[index]:
                pacmanstate = int(realdistance[index] + (3 * longestdistance))
        except:
            pass

        return pacmanstate



    def getQValue(self, state, action):
        position = int(self.computePosition(state))  # 👈 cast to int
        action_column = self.actions[action]
        return self.q_table[position][action_column]

    def computeValueFromQValues(self, state):
        legalActions = state.getLegalPacmanActions()
        if 'Stop' in legalActions: legalActions.remove("Stop")
        if len(legalActions) == 0:
            return 0.0
        return max([self.getQValue(state, action) for action in legalActions])

    def computeActionFromQValues(self, state):
        """
            Compute the best action to take in a state.  Note that if there
            are no legal actions, which is the case at the terminal state,
            you should return None.
        """
        legalActions = state.getLegalPacmanActions()
        if 'Stop' in legalActions: legalActions.remove("Stop")
        if len(legalActions)==0:
            return None

        best_actions = [legalActions[0]]
        best_value = self.getQValue(state, legalActions[0])
        for action in legalActions:
            value = self.getQValue(state, action)
            if value == best_value:
                best_actions.append(action)
            if value > best_value:
                best_actions = [action]
                best_value = value

        return random.choice(best_actions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
        """

        # Pick Action
        legalActions = state.getLegalPacmanActions()
        if 'Stop' in legalActions: legalActions.remove("Stop")
        action = None

        if len(legalActions) == 0:
                return action

        flip = util.flipCoin(self.epsilon)

        if flip:
            return random.choice(legalActions)
        return self.getPolicy(state)


    def update(self, state, action, nextState, reward):
        position = int(self.computePosition(state))  # 👈 cast to int
        action_column = self.actions[action]
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        self.q_table[position][action_column] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * sample
        return self.q_table[position][action_column]

    def getPolicy(self, state):
        "Return the best action in the qtable for a given state"
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        "Return the highest q value for a given state"
        return self.computeValueFromQValues(state)

    def getReward(self, state, action, nextState):
        reward = 0

    # Base score difference
        reward += nextState.getScore() - state.getScore()

    # Encourage eating food
        if nextState.getNumFood() < state.getNumFood():
            reward += 10

    # Encourage eating power capsules
        if len(nextState.getCapsules()) < len(state.getCapsules()):
            reward += 50

    # Encourage moving closer to ghosts (only when scared)
        scaredTimes = [ghost.scaredTimer for ghost in nextState.getGhostStates()]
        if any(scaredTimes):
            oldDist = min(Distancer(state.data.layout).getDistance(state.getPacmanPosition(), g)
                      for g in state.getGhostPositions())
            newDist = min(Distancer(nextState.data.layout).getDistance(nextState.getPacmanPosition(), g)
                      for g in nextState.getGhostPositions())
            if newDist < oldDist:
                reward += 5

    # Big penalty for dying
        if nextState.isLose():
            reward -= 500

    # Big bonus for winning
        if nextState.isWin():
            reward += 500

        return reward

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def final(self, state):
        """Called at the end of each game."""
        self.episodesSoFar += 1
        if self.episodesSoFar >= self.numTraining:
            self.epsilon = 0   # stop random moves after training


class ApproximateQAgent(QLearningAgent):
    """
    Approximate Q-Learning Agent

    Uses a feature extractor to approximate Q-values:
        Q(s,a) = w · f(s,a)
    where w are learned weights and f are feature values.
    """

    def __init__(self, extractor='SimpleExtractor', **args):
        # feature extractor (e.g., SimpleExtractor)
        self.featExtractor = util.lookup(extractor, globals())()
        QLearningAgent.__init__(self, **args)
        # weights are stored in a util.Counter (dict with default 0)
        self.weights = util.Counter()

    def getQValue(self, state, action):
        """
        Return Q(s,a) = w · f(s,a)
        """
        features = self.featExtractor.getFeatures(state, action)
        q_value = 0.0
        for f in features:
            q_value += self.weights[f] * features[f]
        return q_value

    def update(self, state, action, nextState, reward):
        """
        Update weights based on transition.
        """
        features = self.featExtractor.getFeatures(state, action)
        difference = (reward + self.discount * self.getValue(nextState)) - self.getQValue(state, action)

        for f in features:
            self.weights[f] += self.alpha * difference * features[f]

    def final(self, state):
        """
        Called at the end of each game.
        Prints weights after training is finished.
        """
        if self.episodesSoFar == self.numTraining:
            print("Final weights:")
            for f, w in self.weights.items():
                print(f"{f}: {w:.4f}")
