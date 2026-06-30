import util
from game import Directions
import math

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
        Returns a dict (util.Counter) of feature values for a (state, action).
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state, action)] = 1.0
        return feats

class SimpleExtractor(FeatureExtractor):
    """
    Simple features:
      - bias
      - eats-food
      - closest-food
      - ghosts-1-step
    """

    def getFeatures(self, state, action):
        from util import manhattanDistance
        feats = util.Counter()
        feats["bias"] = 1.0

        # successor state
        successor = state.generateSuccessor(0, action)
        pacPos = successor.getPacmanPosition()
        food = successor.getFood()
        ghosts = successor.getGhostPositions()

        # if eating food
        if food[pacPos[0]][pacPos[1]]:
            feats["eats-food"] = 1.0

        # ghost danger
        for ghost in ghosts:
            if manhattanDistance(pacPos, ghost) <= 1:
                feats["ghosts-1-step"] += 1.0

        # closest food distance
        foodList = food.asList()
        if len(foodList) > 0:
            minDist = min(manhattanDistance(pacPos, f) for f in foodList)
            feats["closest-food"] = float(minDist) / (food.width * food.height)
        else:
            feats["closest-food"] = 0.0

        return feats
