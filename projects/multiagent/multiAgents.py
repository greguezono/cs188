# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def distClosestFood(self, currPos, foodGrid):
        output = float('inf')
        i = 0
        for row in foodGrid:
            for j in range(len(foodGrid[0])):
                if foodGrid[i][j]:
                    dist = manhattanDistance(currPos, [i,j])
                    if dist < output:
                        output = dist
            i+= 1
        if output == float('inf') or output == 0:
            return 0
        return 1/output

    def distClosestCap(self, currPos, capGrid):
        output = float('inf')
        i = 0
        for row in capGrid:
            for j in range(len(capGrid[0])):
                if capGrid[i][j]:
                    dist = manhattanDistance(currPos, [i,j])
                    if dist < output:
                        output = dist
            i+= 1
        if output == float('inf'):
            return 1/1000
        elif output == 0:
            return 0
        return 1/output

    def distGhosts(self, newGhostStates, successorGameState, newPos):
        dist = 0
        for i in range(1, len(newGhostStates)+1):
            state = successorGameState.getGhostPosition(i)
            dist += manhattanDistance(newPos, state)
        if (dist == 0):
            return 1
        return dist

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]



    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newCapsules = successorGameState.getCapsules()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # Ghost Heuristic
        food = 0
        dist_pm_g = self.distGhosts(newGhostStates, successorGameState, newPos)

        #if food in this state
        if (currentGameState.hasFood(newPos[0], newPos[1])):
            food = 5
        #if capsule in this state
        elif (newPos in currentGameState.getCapsules()):
            return 1000
        #distance to the closest food
        d = self.distClosestFood(newPos, newFood)
        #distance to the closest capsule
        c = self.distClosestCap(newPos, newCapsules)
        #if the ghosts are scared
        if (sum(newScaredTimes) > 1):
            return food + d
        elif (d < 2 and 5 < dist_pm_g):
            return food + d
        elif (d > 2 and 6 < dist_pm_g):
            return food + d * 3
        else:
            return dist_pm_g + (food + d + c)

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def valueState(self, state, agentIndex, depth):
        if (state.isWin() or state.isLose()):
            return self.evaluationFunction(state)
        elif (depth >= self.depth):
            return self.evaluationFunction(state)
        else:
            if (agentIndex == 0):
                return self.maxValue(state, agentIndex, depth)
            else:
                return self.minValue(state, agentIndex, depth)

    def maxValue(self, state, agentIndex, depth):
        v = -float('inf')
        actions = state.getLegalActions(agentIndex)
        optAction = None
        for action in actions:
            successorState = state.generateSuccessor(agentIndex, action)
            successorVal = self.valueState(successorState, agentIndex + 1, depth)
            if (v < successorVal):
                v = successorVal
                optAction = action
        if (depth == 0 and agentIndex == 0):
            return optAction
        return v

    def minValue(self, state, agentIndex, depth):
        v = float('inf')
        actions = state.getLegalActions(agentIndex)
        for action in actions:
            successorState = state.generateSuccessor(agentIndex, action)
            if (agentIndex + 1 >= state.getNumAgents()):
                agentIndex = -1
                depth += 1
            v = min(v, self.valueState(successorState, agentIndex + 1, depth))
        return v

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        return self.valueState(gameState, 0, 0)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def alphaBetaState(self, state, agentIndex, depth, alpha, beta):
        if (state.isWin() or state.isLose()):
            return self.evaluationFunction(state)
        elif (depth >= self.depth):
            return self.evaluationFunction(state)
        else:
            if (agentIndex == 0):
                return self.maxValue(state, agentIndex, depth, alpha, beta)
            else:
                return self.minValue(state, agentIndex, depth, alpha, beta)

    def maxValue(self, state, agentIndex, depth, alpha, beta):
        v = -float('inf')
        actions = state.getLegalActions(agentIndex)
        optAction = None
        for action in actions:
            successorState = state.generateSuccessor(agentIndex, action)
            successorVal = self.alphaBetaState(successorState, agentIndex + 1, depth, alpha, beta)
            if (v < successorVal):
                v = successorVal
                optAction = action
            if (v > beta):
                return v
            alpha = max(alpha, v)
        if (depth == 0 and agentIndex == 0):
            return optAction
        return v

    def minValue(self, state, agentIndex, depth, alpha, beta):
        v = float('inf')
        actions = state.getLegalActions(agentIndex)
        for action in actions:
            successorState = state.generateSuccessor(agentIndex, action)
            if (agentIndex + 1 >= state.getNumAgents()):
                agentIndex = -1
                depth += 1
            v = min(v, self.alphaBetaState(successorState, agentIndex + 1, depth, alpha, beta))
            if (v < alpha):
                return v
            beta = min(beta, v)
        return v

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.alphaBetaState(gameState, 0, 0, -float('inf'), float('inf'))

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def expectimaxState(self, state, agentIndex, depth):
        if (state.isWin() or state.isLose()):
            return self.evaluationFunction(state)
        elif (depth >= self.depth):
            return self.evaluationFunction(state)
        else:
            if (agentIndex == 0):
                return self.maxValue(state, agentIndex, depth)
            else:
                return self.expectedValue(state, agentIndex, depth)

    def maxValue(self, state, agentIndex, depth):
        v = -float('inf')
        actions = state.getLegalActions(agentIndex)
        optAction = None
        flag = False
        if (self.evaluationFunction == betterEvaluationFunction):
            flag = True
        for action in actions:
            if (flag and action == 'Stop'):
                continue
            successorState = state.generateSuccessor(agentIndex, action)
            successorVal = self.expectimaxState(successorState, agentIndex + 1, depth)
            if (v < successorVal):
                v = successorVal
                optAction = action
        if (depth == 0 and agentIndex == 0):
            return optAction
        return v

    def expectedValue(self, state, agentIndex, depth):
        v = 0
        actions = state.getLegalActions(agentIndex)
        for action in actions:
            successorState = state.generateSuccessor(agentIndex, action)
            if (agentIndex + 1 >= state.getNumAgents()):
                agentIndex = -1
                depth += 1
            v += self.expectimaxState(successorState, agentIndex + 1, depth)
        return v / len(actions)

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        return self.expectimaxState(gameState, 0, 0)

def distClosestFood(currPos, foodGrid, gameState):
    output = float('inf')
    i = 0
    for row in foodGrid:
        for j in range(len(foodGrid[0])):
            if foodGrid[i][j]:
                dist = manhattanDistance(currPos, [i,j])
                if dist < output:
                    output = dist
        i+= 1
    if output == float('inf') or output == 0:
        return 0
    return output

def distClosestCap(currPos, capGrid, gameState):
    output = float('inf')
    i = 0
    for row in capGrid:
        for j in range(len(capGrid[0])):
            if capGrid[i][j]:
                dist = manhattanDistance(currPos, [i,j])
                if dist < output:
                    output = dist
        i+= 1
    if output == float('inf'):
        return 1/1000
    elif output == 0:
        return 0
    return output

def distGhosts(currPos, ghostStates, gameState):
    dist = 0
    for i in range(1, len(ghostStates)+1):
        state = gameState.getGhostPosition(i)
        dist += manhattanDistance(currPos, state)
    if (dist == 0):
        return 1
    return dist

def numFood(foodGrid):
    total = 0
    i = 0
    for row in foodGrid:
        for j in range(len(foodGrid[0])):
            if foodGrid[i][j]:
                total+=1
        i+= 1
    return total

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: I kept track of if the ghosts are scared, the number of foods,
    the distance to the nearest ghost from PM, distance to the nearest food from PM,
    and the distance from the nearest capsule to PM. If the gamestate is a win or lose
    , I set as large positive or negative value to denote those states respectively.
    I then check if the ghosts are scared, in which case PM should focus on
    eating as many pellets (case 1) as possible and trying to eat the ghosts as
    well (case 2).
    Case 3: If the ghosts aren't scared, I check if PM is far from the nearest
    ghost (4 away) and if he is close to food (4 away as well). If that is true,
    PM should prioritize eating food than running away from the ghosts
    Case 4: The last case is if PM is close to a ghost, in which case he should
    prioritze running away more than anything else.
    """
    pmPos = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    capsulesGrid = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]


    #distances
    dist_pm_g = distGhosts(pmPos, ghostStates, currentGameState)
    distFood = distClosestFood(pmPos, foodGrid, currentGameState)
    distCapsule = distClosestCap(pmPos, capsulesGrid, currentGameState)

    numfood = numFood(foodGrid)
    score = currentGameState.getScore()
    if (currentGameState.isWin()):
        return score + 1000
    if (currentGameState.isLose()):
        return score - 1000
    if (sum(scaredTimes) > 0):
        # Case 1
        if (dist_pm_g > 0):
            return (score+50) * 1.1 - distCapsule - distFood + 1/dist_pm_g * 1.5
        # Case 2
        else:
            # print("case scared 2: ", score * 1.1 + 100)
            return (score+50) * 1/dist_pm_g * 1.5 + 1.1 + 500
    # Case 3
    elif (dist_pm_g > 4 and distFood < 4):
        return score*1.2 - distFood*.7 - distCapsule*.5 - numfood
    # Case 4
    else:
        return score - distFood - distCapsule*.5 + dist_pm_g*.7 - numfood

# Abbreviation
better = betterEvaluationFunction
