# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

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
    print legalMoves
    print "scores %r" % scores, bestScore
    print chosenIndex
    print legalMoves[chosenIndex]
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
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    score = successorGameState.getScore()
    currentPos = currentGameState.getPacmanPosition()
     
    ghostPos = [ghost.getPosition() for ghost in newGhostStates] 
    ghostDist = [manhattanDistance(newPos, pos) for pos in ghostPos ]
    foodPos = currentGameState.getFood().asList()
    foodDist = [manhattanDistance(newPos, pos) for pos in foodPos ]
    
    foodScore = 1.0/(1.0+min(foodDist))
    #util.pause()
    
    ghostScore = min(ghostDist)/(1+max(ghostDist))
    peletScore = 1.0/(1.0+len(foodPos))
    
    #if newPos == currentPos:
    #    penalty = 0.1
    #else:
    #    penalty = 0
    print foodScore, ghostScore, peletScore
    score = ghostScore + ghostScore*foodScore
    return score



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
    
    "*** YOUR CODE HERE ***"
    def minAgent(self, depth, gameState, ghostIndex):
        
        if(self.depth == depth or gameState.isLose() or gameState.isWin()):
            return self.evaluationFunction(gameState)
        
        successors = []
        scores = []
        actions = gameState.getLegalActions(ghostIndex)
        
        for action in actions:
            successors.append(gameState.generateSuccessor(ghostIndex, action))
        
        if(ghostIndex == gameState.getNumAgents() - 1):
            for successor in successors:
                scores.append(self.maxAgent(depth + 1, successor))
        
        elif(ghostIndex < gameState.getNumAgents() - 1):
            for successor in successors:
                scores.append(self.minAgent(depth, successor, ghostIndex + 1))
            
        return min(scores)

    def maxAgent(self, depth, gameState):
        
        if (self.depth == depth or gameState.isLose() or gameState.isWin()):
            return self.evaluationFunction(gameState)
        
        successors = []
        scores = []
        actions = gameState.getLegalPacmanActions()
        actions.remove(Directions.STOP)
        
        for action in actions:
            successors.append(gameState.generatePacmanSuccessor(action))
        
        for successor in successors:
            scores.append(self.minAgent(depth, successor, 1))
        
        return max(scores)
    
    def getAction(self, gameState):
        
        actions = gameState.getLegalPacmanActions()
        actions.remove(Directions.STOP)
        successors = []
        scores = []
        
        for action in actions:
            successors.append(gameState.generatePacmanSuccessor(action))
        
        for successor in successors:
            scores.append(self.minAgent(0, successor, 1))
        
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
    
        return actions[chosenIndex]
    

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """
  def minAgent(self, depth, gameState, ghostIndex, alpha, beta):
    
    if(self.depth == depth or gameState.isLose() or gameState.isWin()):
        return self.evaluationFunction(gameState)
    
    actions = gameState.getLegalActions(ghostIndex)
    score = float('infinity')
    
    for action in actions:
      successor = gameState.generateSuccessor(ghostIndex, action)
      
      if ghostIndex < gameState.getNumAgents() - 1:
        score = min(score, self.minAgent(depth, successor, ghostIndex + 1, alpha, beta))
      
      elif ghostIndex == gameState.getNumAgents() - 1:
        score = min(score, self.maxAgent(depth + 1, successor, alpha, beta))
      
      if(score < alpha):
        return score
      
      beta = min(score, beta)
    
    return score

  def maxAgent(self, depth, gameState, alpha, beta):
    
    if (self.depth == depth or gameState.isLose() or gameState.isWin()):
      return self.evaluationFunction(gameState)
    
    actions = gameState.getLegalActions(0)
    if Directions.STOP in actions:
        actions.remove(Directions.STOP)
    score = float('-infinity')

    for action in actions:
      successor = gameState.generateSuccessor(0, action)
      score = max(score, self.minAgent(depth, successor, 1, alpha, beta))
      
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
    score = float('-infinity')
    actions = gameState.getLegalActions(0)
    if Directions.STOP in actions:
        actions.remove(Directions.STOP)
    bestAction = actions[0]

    for action in actions:
      successor = gameState.generateSuccessor(0, action)
      res = self.minAgent(0, successor, 1, alpha, beta)
      
      if(res > score):
        score = res
        bestAction = action
      
      if(res > beta):
        return bestAction
      
      alpha = max(alpha, res)

    return bestAction  

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"

    def minAgent(self, depth, gameState, ghostIndex):
        
        if(self.depth == depth or gameState.isLose() or gameState.isWin()):
            return self.evaluationFunction(gameState)
        
        successors = []
        scores = []
        actions = gameState.getLegalActions(ghostIndex)
        
        for action in actions:
            successors.append(gameState.generateSuccessor(ghostIndex, action))
        
        if(ghostIndex == gameState.getNumAgents() - 1):
            for successor in successors:
                scores.append(self.maxAgent(depth + 1, successor))
        
        elif(ghostIndex < gameState.getNumAgents() - 1):
            for successor in successors:
                scores.append(self.minAgent(depth, successor, ghostIndex + 1))
        
        result = sum(scores) / len(scores)
            
        return result

    def maxAgent(self, depth, gameState):
        
        if (self.depth == depth or gameState.isLose() or gameState.isWin()):
            return self.evaluationFunction(gameState)
        
        successors = []
        scores = []
        actions = gameState.getLegalActions(0)
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        
        for action in actions:
            successors.append(gameState.generateSuccessor(0, action))
        
        for successor in successors:
            scores.append(self.minAgent(depth, successor, 1))
        
        return max(scores)
    
    def getAction(self, gameState):
        
        actions = gameState.getLegalActions(0)
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        successors = []
        scores = []
        
        for action in actions:
            successors.append(gameState.generateSuccessor(0, action))
        
        for successor in successors:
            scores.append(self.minAgent(0, successor, 1))
        
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
    
        return actions[chosenIndex]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
        
    from util import pause
        
    pacmanPos = currentGameState.getPacmanPosition()
    foodPos = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    
    ghostPos = [ghost.getPosition() for ghost in ghostStates] 
    ghostDist = [manhattanDistance(pacmanPos, pos) for pos in ghostPos ]
  
    foodPos = currentGameState.getFood().asList()
    
    if pacmanPos in foodPos: 
        foodScore = 10
    else:
        foodScore = 0 
    
    if min(ghostDist) > 3:
        ghostScore = 10
    elif min(ghostDist) <= 3:
        ghostScore = -10
    
    
        
    #print scoreEvaluationFunction(currentGameState), currentGameState.getScore() 
    if scaredTimes[0] != 0:
        scared = 10
    else:
        scared = 0
    
    return currentGameState.getScore() + ghostScore + foodScore + scared
  
# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

