
import util
from game import Actions, Directions
from collections import deque

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
        Returns a dict/Counter from features to counts.
        """
        util.raiseNotDefined()


class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        features = util.Counter()
        features[(state, action)] = 1.0
        return features


class SimpleExtractor(FeatureExtractor):
    """
    Extracts simple features for Pacman:
    - whether Pacman eats food
    - distance to closest food
    - whether a ghost is one step away
    """
    def getFeatures(self, state, action):
        features = util.Counter()
        successor = state.generateSuccessor(0, action)
        pacmanPos = successor.getPacmanPosition()
        food = successor.getFood()
        ghostStates = successor.getGhostPositions()

        # Closest food
        foodList = food.asList()
        if foodList:
            minDist = min([util.manhattanDistance(pacmanPos, f) for f in foodList])
            features["closestFood"] = float(minDist) / (food.width * food.height)
        else:
            features["closestFood"] = 0.0

        # Ghost proximity
        for ghost in ghostStates:
            dist = util.manhattanDistance(pacmanPos, ghost)
            if dist < 2:
                features["ghostDanger"] = 1.0
            else:
                features["ghostDanger"] = 0.0

        # If eating food
        if state.getNumFood() > successor.getNumFood():
            features["eatFood"] = 1.0

        # Bias term
        features["bias"] = 1.0
        return features


class HumanVisionExtractor(FeatureExtractor):
    """
    Human-Like Vision and Topology-Aware Feature Extractor for Pacman RL.

    Features extracted:
    1. Line-of-Sight Raycasting (Cardinal Directions):
       - ghost-visible-approaching: 1.0 / dist for active ghosts in direct corridor sight facing Pacman.
       - ghost-visible-retreating: 1.0 / dist for active ghosts facing away or perpendicular.
       - scared-ghost-visible: 1.0 / dist for edible ghosts in line of sight.
    2. Multi-Step Danger Gradient (BFS Maze Distance for blind corners):
       - ghost-1-step-away: Active ghost within 1 maze step.
       - ghost-2-steps-away: Active ghost at 2 maze steps.
       - ghost-3-steps-away: Active ghost at 3 maze steps.
    3. Topology & Escape Analysis:
       - trapped-in-dead-end: 1.0 if moving to successor puts Pacman in a sealed dead-end pocket
         with an active ghost approaching/blocking the entrance.
    4. Goal & Reward Objectives:
       - eats-food: 1.0 if successor position eats a dot safely.
       - closest-food: Normalized BFS maze distance to the nearest food dot.
       - eats-capsule: 1.0 if successor consumes a power pellet.
       - closest-scared-ghost: Normalized BFS maze distance to nearest edible scared ghost.
       - bias: 1.0 constant bias term.
    """

    @staticmethod
    def castRay(pos, direction, walls, ghostStates):
        """
        Casts a ray from `pos` in `direction` until a wall is hit.
        Returns list of (ghostState, distance) for all ghosts along the ray.
        """
        x, y = pos
        dx, dy = Actions.directionToVector(direction)
        if (dx, dy) == (0, 0):
            return []

        ghostsFound = []
        dist = 0
        while True:
            dist += 1
            x, y = int(x + dx), int(y + dy)
            if x < 0 or x >= walls.width or y < 0 or y >= walls.height or walls[x][y]:
                break
            for g in ghostStates:
                gx, gy = g.getPosition()
                if (int(gx), int(gy)) == (x, y):
                    ghostsFound.append((g, dist))
        return ghostsFound

    @staticmethod
    def computeMazeDistances(startPos, walls, maxDepth=25):
        """
        Runs BFS from startPos to compute shortest maze distances to all reachable tiles.
        Returns a dict mapping (x, y) -> shortest_distance.
        """
        distances = {startPos: 0}
        queue = deque([startPos])

        while queue:
            curr = queue.popleft()
            currDist = distances[curr]
            if currDist >= maxDepth:
                continue

            x, y = curr
            for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                dx, dy = Actions.directionToVector(action)
                nx, ny = int(x + dx), int(y + dy)
                if 0 <= nx < walls.width and 0 <= ny < walls.height and not walls[nx][ny]:
                    neighbor = (nx, ny)
                    if neighbor not in distances:
                        distances[neighbor] = currDist + 1
                        queue.append(neighbor)
        return distances

    @staticmethod
    def isTrappedInDeadEnd(pacmanPos, activeGhosts, walls, maxPocketSize=8):
        """
        Determines if pacmanPos is located within a sealed dead-end pocket
        with an approaching active ghost blocking the exit.

        Distinguishes:
        - Corridors with turns leading to open junctions (NOT trapped).
        - Genuine cul-de-sacs / dead ends with an active ghost approaching the entrance (TRAPPED).
        - Branching junctions inside a pocket whose exits are sealed by ghosts (TRAPPED).
        """
        if not activeGhosts:
            return False

        # Compute ghost threat positions (ghost position + 1 step forward in ghost direction)
        ghostPositions = set()
        ghostThreatPositions = set()
        for g in activeGhosts:
            gx, gy = int(g.getPosition()[0]), int(g.getPosition()[1])
            ghostPositions.add((gx, gy))
            gdx, gdy = Actions.directionToVector(g.getDirection())
            ghostThreatPositions.add((int(gx + gdx), int(gy + gdy)))

        # If Pacman is already at ghost position, handled by collision features
        if pacmanPos in ghostPositions:
            return True

        # BFS to find reachable tiles without crossing ghost positions
        visited = {pacmanPos}
        queue = deque([pacmanPos])
        openJunctionsFound = 0

        while queue:
            curr = queue.popleft()
            x, y = curr

            # Find all wall-free neighbors
            openNeighbors = []
            for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                dx, dy = Actions.directionToVector(action)
                nx, ny = int(x + dx), int(y + dy)
                if 0 <= nx < walls.width and 0 <= ny < walls.height and not walls[nx][ny]:
                    openNeighbors.append((nx, ny))

            # A junction has degree >= 3
            if len(openNeighbors) >= 3:
                # Check if this junction has unblocked escape paths
                unblockedNeighbors = [n for n in openNeighbors if n not in ghostPositions and n not in ghostThreatPositions]
                if len(unblockedNeighbors) >= 2:
                    openJunctionsFound += 1

            for neighbor in openNeighbors:
                if neighbor not in visited:
                    # Do not expand past active ghost positions
                    if neighbor not in ghostPositions:
                        visited.add(neighbor)
                        queue.append(neighbor)

            # If reachable area is large or multiple escape junctions exist, it's not a dead-end pocket
            if len(visited) > maxPocketSize or openJunctionsFound >= 1:
                return False

        # If reachable safe tiles are bounded and no unblocked escape junction exists:
        # Check if an active ghost is within close proximity (<= 4 steps)
        minGhostDist = min(util.manhattanDistance(pacmanPos, g.getPosition()) for g in activeGhosts)
        return minGhostDist <= 4.0

    def getFeatures(self, state, action):
        features = util.Counter()
        successor = state.generateSuccessor(0, action)
        pacmanPos = successor.getPacmanPosition()
        walls = successor.getWalls()
        food = successor.getFood()
        capsules = successor.getCapsules()
        ghostStates = successor.getGhostStates()

        gridArea = float(walls.width * walls.height)

        activeGhosts = [g for g in ghostStates if g.scaredTimer <= 1]
        scaredGhosts = [g for g in ghostStates if g.scaredTimer > 1]

        # -------------------------------------------------------------
        # 1. Line-of-Sight Raycasting (Cardinal Directions)
        # -------------------------------------------------------------
        for rayDir in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            ghostsInSight = self.castRay(pacmanPos, rayDir, walls, ghostStates)
            for g, dist in ghostsInSight:
                invDist = 1.0 / float(dist)
                if g.scaredTimer > 1:
                    features["scared-ghost-visible"] = max(features["scared-ghost-visible"], invDist)
                else:
                    gDir = g.getDirection()
                    isApproaching = (gDir == Actions.reverseDirection(rayDir))
                    if isApproaching:
                        features["ghost-visible-approaching"] = max(features["ghost-visible-approaching"], invDist)
                    else:
                        features["ghost-visible-retreating"] = max(features["ghost-visible-retreating"], invDist)

        # -------------------------------------------------------------
        # 2. Multi-Step Danger Gradient (BFS Maze Distances)
        # -------------------------------------------------------------
        mazeDists = self.computeMazeDistances(pacmanPos, walls, maxDepth=10)

        for g in activeGhosts:
            gx, gy = int(g.getPosition()[0]), int(g.getPosition()[1])
            gDist = mazeDists.get((gx, gy), None)
            if gDist is not None:
                if gDist <= 1:
                    features["ghost-1-step-away"] += 1.0
                elif gDist == 2:
                    features["ghost-2-steps-away"] += 1.0
                elif gDist == 3:
                    features["ghost-3-steps-away"] += 1.0

        # -------------------------------------------------------------
        # 3. Topology & Escape Analysis (Turn-Aware Dead-End Traps)
        # -------------------------------------------------------------
        if self.isTrappedInDeadEnd(pacmanPos, activeGhosts, walls):
            features["trapped-in-dead-end"] = 1.0

        # -------------------------------------------------------------
        # 4. Food, Capsules & Scared Ghost Objectives
        # -------------------------------------------------------------
        # Safe food eating
        if state.getNumFood() > successor.getNumFood():
            if features["ghost-1-step-away"] == 0 and features["ghost-visible-approaching"] == 0:
                features["eats-food"] = 1.0

        # Closest food dot (BFS maze distance)
        foodList = food.asList()
        if foodList:
            foodDists = [mazeDists[f] for f in foodList if f in mazeDists]
            if foodDists:
                features["closest-food"] = float(min(foodDists)) / gridArea
            else:
                features["closest-food"] = 1.0
        else:
            features["closest-food"] = 0.0

        # Capsules (Power Pellets)
        if pacmanPos in state.getCapsules():
            features["eats-capsule"] = 1.0

        # Scared ghosts (hunting)
        if scaredGhosts:
            scaredDists = [mazeDists[(int(g.getPosition()[0]), int(g.getPosition()[1]))]
                           for g in scaredGhosts if (int(g.getPosition()[0]), int(g.getPosition()[1])) in mazeDists]
            if scaredDists:
                features["closest-scared-ghost"] = float(min(scaredDists)) / gridArea

        # Bias term
        features["bias"] = 1.0

        return features


# Alias for convenience
RaycastExtractor = HumanVisionExtractor