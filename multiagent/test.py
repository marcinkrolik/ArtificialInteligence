from util import manhattanDistance
from game import Directions
import random, util, math

from game import Agent

#Global variables used for cacheing
globalCacheWalls = None
globalMazeDistCache = {}


def cacheMazeDistances(walls):
    global globalCacheWalls
    if (globalCacheWalls == walls):
      return
    globalCacheWalls = walls
    for i in range(walls.width):
      for j in range(walls.height):
        for x in range(walls.width):
          for y in range(walls.height):
            if(((i, j), (x, y)) not in globalMazeDistCache and ((x, y), (i, j)) not in globalMazeDistCache):
              if(not walls[i][j] and not walls[x][y]):
                globalMazeDistCache[(i, j), (x, y)] = mazeDistance((i, j), (x, y), walls)

def getMazeDistanceFromCache(point1, point2):
    p1 = (math.floor(point1[0]), math.floor(point1[1]))
    p2 = (math.floor(point2[0]), math.floor(point2[1]))
    if((p1, p2) in globalMazeDistCache):
      return globalMazeDistCache[(p1, p2)]
    else:
      return globalMazeDistCache[(p2, p1)]

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """

  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
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
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """

    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    oldPos = currentGameState.getPacmanPosition()
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    oldFoodsList = oldFood.asList()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    walls = currentGameState.getWalls()

    #cache the mazeDistances
    cacheMazeDistances(walls)
    #Find the closest food before moving
    targetFood = None
    distTargetFoodBefore = None
    for food in oldFoodsList:
      dist = getMazeDistanceFromCache(oldPos, food)
      if(distTargetFoodBefore == None or distTargetFoodBefore > dist):
        distTargetFoodBefore = dist
        targetFood = food

    distTargetFoodAfter = getMazeDistanceFromCache(newPos, targetFood)
    capsuleEval = 0

    #Get closest ghost in terms of mazeDistance
    ghostEval = 0
    for ghostState in newGhostStates:
      if(ghostState.scaredTimer <= 0):
        if(getMazeDistanceFromCache(newPos, ghostState.getPosition()) <= 3):
          ghostEval -= (6 - getMazeDistanceFromCache(newPos, ghostState.getPosition()))**4
      else:
        capsuleEval += 10

    "*** YOUR CODE HERE ***"

    return (2 - distTargetFoodAfter) * 2 + ghostEval

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

  def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def minEvalGhosts(self, currentDepth, gameState, ghostIndex):
    if(self.depth == currentDepth or gameState.isLose() or gameState.isWin()):
        return self.evaluationFunction(gameState)
    numGhosts = gameState.getNumAgents() - 1
    minScore = None
    legalActions = gameState.getLegalActions(ghostIndex)
    resultingGameStates = [gameState.generateSuccessor(ghostIndex, action) for action in legalActions]

    if(ghostIndex == numGhosts):
      scores = [self.maxEvalPacman(currentDepth + 1, resultingGameState) for resultingGameState in resultingGameStates]
    elif(ghostIndex < numGhosts):
      scores = [self.minEvalGhosts(currentDepth, resultingGameState, ghostIndex + 1) for resultingGameState in resultingGameStates]

    return min(scores)

  def maxEvalPacman(self, currentDepth, gameState):
    if (self.depth == currentDepth or gameState.isLose() or gameState.isWin()):
      return self.evaluationFunction(gameState)
    legalActions = gameState.getLegalPacmanActions()
    legalActions.remove(Directions.STOP)
    resultingGameStates = [gameState.generatePacmanSuccessor(action) for action in legalActions]
    scores = [self.minEvalGhosts(currentDepth, resultingGameState, 1) for resultingGameState in resultingGameStates]
    return max(scores)

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"

    legalActions = gameState.getLegalPacmanActions()
    legalActions.remove(Directions.STOP)
    resultingGameStates = [gameState.generatePacmanSuccessor(action) for action in legalActions]
    scores = [self.minEvalGhosts(0, resultingGameState, 1) for resultingGameState in resultingGameStates]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    return legalActions[chosenIndex]

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def minEvalGhosts(self, currentDepth, gameState, ghostIndex, alpha, beta):
    if(self.depth == currentDepth or gameState.isLose() or gameState.isWin()):
        return self.evaluationFunction(gameState)
    numGhosts = gameState.getNumAgents() - 1
    minScore = None
    legalActions = gameState.getLegalActions(ghostIndex)
    score = float('infinity')
    for action in legalActions:
      resultingGameState = gameState.generateSuccessor(ghostIndex, action)
      if(ghostIndex < numGhosts):
        score = min(score, self.minEvalGhosts(currentDepth, resultingGameState, ghostIndex + 1, alpha, beta))
      elif(ghostIndex == numGhosts):
        score = min(score, self.maxEvalPacman(currentDepth + 1, resultingGameState, alpha, beta))
      if(score < alpha):
        return score
      beta = min(score, beta)
    return score

  def maxEvalPacman(self, currentDepth, gameState, alpha, beta):
    if (self.depth == currentDepth or gameState.isLose() or gameState.isWin()):
      return self.evaluationFunction(gameState)
    legalActions = gameState.getLegalPacmanActions()
    legalActions.remove(Directions.STOP)
    score = float('-infinity')

    for action in legalActions:
      resultingGameState = gameState.generatePacmanSuccessor(action)
      score = max(score, self.minEvalGhosts(currentDepth, resultingGameState, 1, alpha, beta))
      if(score > beta):
        return score
      alpha = max(alpha, score)

    return score

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"

    alpha = float('-infinity')
    beta = float('infinity')
    legalActions = gameState.getLegalPacmanActions()
    legalActions.remove(Directions.STOP)
    score = float('-infinity')
    bestAction = legalActions[0]

    for action in legalActions:
      resultingGameState = gameState.generatePacmanSuccessor(action)
      resultingScore = self.minEvalGhosts(0, resultingGameState, 1, alpha, beta)
      if(resultingScore > score):
        score = resultingScore
        bestAction = action
      if(resultingScore > beta): #This should never happen at the root
        return bestAction
      alpha = max(alpha, resultingScore)

    return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def expEvalGhosts(self, currentDepth, gameState, ghostIndex):
    if(self.depth == currentDepth or gameState.isLose() or gameState.isWin()):
        return self.evaluationFunction(gameState)
    numGhosts = gameState.getNumAgents() - 1
    minScore = None
    legalActions = gameState.getLegalActions(ghostIndex)
    resultingGameStates = [gameState.generateSuccessor(ghostIndex, action) for action in legalActions]

    if(ghostIndex == numGhosts):
      scores = [self.maxEvalPacman(currentDepth + 1, resultingGameState) for resultingGameState in resultingGameStates]
    elif(ghostIndex < numGhosts):
      scores = [self.expEvalGhosts(currentDepth, resultingGameState, ghostIndex + 1) for resultingGameState in resultingGameStates]

    return sum(scores) / len(scores)

  def maxEvalPacman(self, currentDepth, gameState):
    if (self.depth == currentDepth or gameState.isLose() or gameState.isWin()):
      return self.evaluationFunction(gameState)
    legalActions = gameState.getLegalPacmanActions()
    legalActions.remove(Directions.STOP)
    resultingGameStates = [gameState.generatePacmanSuccessor(action) for action in legalActions]
    scores = [self.expEvalGhosts(currentDepth, resultingGameState, 1) for resultingGameState in resultingGameStates]
    return max(scores)


  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    legalActions = gameState.getLegalPacmanActions()
    legalActions.remove(Directions.STOP)
    resultingGameStates = [gameState.generatePacmanSuccessor(action) for action in legalActions]
    scores = [self.expEvalGhosts(0, resultingGameState, 1) for resultingGameState in resultingGameStates]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    return legalActions[chosenIndex]


def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    Stuff that's good- ghosts not so close that they can kill Pacman, food, power pellets, scared ghosts
    Stuff that's bad- dying
  """
  "*** YOUR CODE HERE ***"

  pacmanPos = currentGameState.getPacmanPosition()
  foodGrid = currentGameState.getFood()
  foodList = foodGrid.asList()
  ghostStates = currentGameState.getGhostStates()
  scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
  wallsGrid = currentGameState.getWalls()

  #Cache the mazeDistances between any pair of positions in the maze, if not cached already.
  #The cache is a globally defined dictionary.
  cacheMazeDistances(wallsGrid)

  #Now use the cache to find the mazeDistance to the closest food pellet.
  distToClosestFood = None
  for food in foodList:
    dist = getMazeDistanceFromCache(pacmanPos, food)
    if(distToClosestFood == None or distToClosestFood > dist):
        distToClosestFood = dist
  if(distToClosestFood == None):
    distToClosestFood = 0

  #Now we must deal with ghosts. Iterate through all the ghosts.
  #If there is a non-scared ghost that can get to Pacman's current position in less than 3 moves,
  #Pacman needs to watch out. The function for 'danger' is a polynomial of degree 3,
  #So danger increases exponentially as ghosts get closer to Pacman

  #We also need to give Pacman an incentive to eat capsules. We give a +10 bonus to the eval
  #for every ghost that is currently scared of Pacman
  danger = 0
  capsuleEval = 0
  for ghostState in ghostStates:
    if(ghostState.scaredTimer <= 0):
      if(getMazeDistanceFromCache(pacmanPos, ghostState.getPosition()) <= 3):
        danger -= (6 - getMazeDistanceFromCache(pacmanPos, ghostState.getPosition()))**3
    else:
      capsuleEval += 10

  #The total eval is the usual score, decreased by the danger of the state,
  #increased by any ghosts that are scared, and decreased very slightly
  #by the mazeDistance to the closest food item (this was to prevent thrashing).
  return scoreEvaluationFunction(currentGameState) + danger + capsuleEval - 0.1 * distToClosestFood

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def __init__(self, evalFn='betterEvaluationFunction', depth='3'):
    MultiAgentSearchAgent.__init__(self, evalFn, depth)

  def minEvalGhosts(self, currentDepth, gameState, ghostIndex, alpha, beta):
    if(self.depth == currentDepth or gameState.isLose() or gameState.isWin()):
        return self.evaluationFunction(gameState)
    numGhosts = gameState.getNumAgents() - 1
    minScore = None
    legalActions = gameState.getLegalActions(ghostIndex)
    score = float('infinity')
    for action in legalActions:
      resultingGameState = gameState.generateSuccessor(ghostIndex, action)
      if(ghostIndex < numGhosts):
        score = min(score, self.minEvalGhosts(currentDepth, resultingGameState, ghostIndex + 1, alpha, beta))
      elif(ghostIndex == numGhosts):
        score = min(score, self.maxEvalPacman(currentDepth + 1, resultingGameState, alpha, beta))
      if(score < alpha):
        return score
      beta = min(score, beta)
    return score

  def maxEvalPacman(self, currentDepth, gameState, alpha, beta):
    if (self.depth == currentDepth or gameState.isLose() or gameState.isWin()):
      return self.evaluationFunction(gameState)
    legalActions = gameState.getLegalPacmanActions()
    legalActions.remove(Directions.STOP)
    score = float('-infinity')

    for action in legalActions:
      resultingGameState = gameState.generatePacmanSuccessor(action)
      score = max(score, self.minEvalGhosts(currentDepth, resultingGameState, 1, alpha, beta))
      if(score > beta):
        return score
      alpha = max(alpha, score)

    return score

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"

    alpha = float('-infinity')
    beta = float('infinity')
    legalActions = gameState.getLegalPacmanActions()
    legalActions.remove(Directions.STOP)
    score = float('-infinity')
    bestAction = legalActions[0]

    for action in legalActions:
      resultingGameState = gameState.generatePacmanSuccessor(action)
      resultingScore = self.minEvalGhosts(0, resultingGameState, 1, alpha, beta)
      if(resultingScore > score):
        score = resultingScore
        bestAction = action
      if(resultingScore > beta): #This should never happen at the root
        return bestAction
      alpha = max(alpha, resultingScore)
    return bestAction


def getSuccessorsForBFS(position, walls):
  successors = []
  x = position[0]
  y = position[1]

  if(x - 1 >= 0):
    if(not walls[x - 1][y]):
      successors.append((x - 1, y))
    if(y - 1 >= 0):
        if(not walls[x - 1][y - 1]):
          successors.append((x - 1, y - 1))
    if(y + 1 < walls.height):
      if(not walls[x - 1][y + 1]):
        successors.append((x - 1, y + 1))

  if(x + 1 < walls.width):
    if(not walls[x + 1][y]):
      successors.append((x + 1, y))
    if(y - 1 >= 0):
        if(not walls[x + 1][y - 1]):
          successors.append((x + 1, y - 1))
    if(y + 1 < walls.height):
      if(not walls[x + 1][y + 1]):
        successors.append((x + 1, y + 1))

    if(y - 1 >= 0):
      if(not walls[x][y - 1]):
        successors.append((x, y - 1))

    if(y + 1 < walls.height):
      if(not walls[x][y + 1]):
        successors.append((x, y + 1))

  return successors

def mazeDistance(start, end, walls):
  from util import Queue
  assert not walls[start[0]][start[1]], 'point1 is a wall: ' + str(start)
  assert not walls[end[0]][end[1]], 'point2 is a wall: ' + str(end)

  bfsQueue = Queue()
  statesExplored = set()
  bfsQueue.push((start, 0))

  while(not bfsQueue.isEmpty()):
    currentNode = bfsQueue.pop()
    if(currentNode[0] == end):
      return currentNode[1]
    successors = getSuccessorsForBFS(currentNode[0], walls)
    for successor in successors:
      if(successor not in statesExplored):
        bfsQueue.push((successor, currentNode[1] + 1))
        statesExplored.add(successor)

  return None

def breadthFirstSearch(problem):
  "Search the shallowest nodes in the search tree first. [p 74]"
  "*** YOUR CODE HERE ***"
  from util import Queue
  successorsExplored = {}
  statesExplored = set()
  bfsQueue = Queue()

  for successor in problem.getSuccessors(problem.getStartState()):
    bfsQueue.push((successor, None))

  statesExplored.add(problem.getStartState())
  if(problem.isGoalState(problem.getStartState())):
    return []

  while(not bfsQueue.isEmpty()):
    currentSuccessorPair = bfsQueue.pop()
    if(currentSuccessorPair[0][0] in statesExplored):
      continue

    successorsExplored[currentSuccessorPair[0]] = currentSuccessorPair[1]
    statesExplored.add(currentSuccessorPair[0][0])
    if(problem.isGoalState(currentSuccessorPair[0][0])):
      return reconstructPath(successorsExplored, currentSuccessorPair[0])

    for successor in problem.getSuccessors(currentSuccessorPair[0][0]):
      if(successor[0] not in statesExplored):
        bfsQueue.push((successor, currentSuccessorPair[0]))

  return None

 

