# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

        #TODO
    def runValueIteration(self):
        # Write value iteration code here
        allStates = self.mdp.getStates()

        for state in allStates:     # Initialize all values to 0
            self.values[state] = 0

        i = 0
        while (i < self.iterations):
            copyofVals = self.values.copy()
            for state in allStates:
                possibleActions = self.mdp.getPossibleActions(state)
                lst = []
                for action in possibleActions:
                    sumOfActionSamples = 0
                    transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
                    for nextState, transitionProb in transitionStatesAndProbs:
                        reward = self.mdp.getReward(state, action, nextState)
                        dValueOfNextState = self.discount * copyofVals[nextState]
                        sumOfActionSamples += transitionProb * (reward + dValueOfNextState)
                    lst.append(sumOfActionSamples)
                if not lst:
                    self.values[state] = 0
                else:
                    self.values[state] = max(lst)
            i+=1


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        qValue = 0
        for nextState, transitionProb in transitionStatesAndProbs:
            reward = self.mdp.getReward(state, action, nextState)
            dValueOfNextState = self.discount * self.values[nextState]
            qValue += transitionProb * (reward + dValueOfNextState)
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        possibleActions = self.mdp.getPossibleActions(state)
        bestAction = None
        maxVal = -float("inf")
        for action in possibleActions:
            sumOfActionSamples = 0
            transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
            for nextState, transitionProb in transitionStatesAndProbs:
                reward = self.mdp.getReward(state, action, nextState)
                dValueOfNextState = self.discount * self.values[nextState]
                sumOfActionSamples += transitionProb * (reward + dValueOfNextState)
            if sumOfActionSamples > maxVal:
                bestAction = action
                maxVal = sumOfActionSamples
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        allStates = self.mdp.getStates()

        for state in allStates:     # Initialize all values to 0
            self.values[state] = 0

        i = 0
        while (i < self.iterations):
            j = i % len(allStates)
            state = allStates[j]
            if self.mdp.isTerminal(state):
                i += 1
                continue
            possibleActions = self.mdp.getPossibleActions(state)
            lst = []
            for action in possibleActions:
                sumOfActionSamples = 0
                transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
                for nextState, transitionProb in transitionStatesAndProbs:
                    reward = self.mdp.getReward(state, action, nextState)
                    dValueOfNextState = self.discount * self.values[nextState]
                    sumOfActionSamples += transitionProb * (reward + dValueOfNextState)
                lst.append(sumOfActionSamples)
            if not lst:
                self.values[state] = 0
            else:
                self.values[state] = max(lst)
            i+=1



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    # Helper function, returns a list of QVals
    def calcQVals(self, state):
        possibleActions = self.mdp.getPossibleActions(state)
        qVals = []
        for action in possibleActions:
            qVal = self.computeQValueFromValues(state, action)
            qVals.append(qVal)
        return qVals

    #Helper function, returns a list of updated Vals
    def calcUpdateVals(self, state):
        lst = []
        for action in self.mdp.getPossibleActions(state):
            qVals = 0
            for nextState, transitionProb in self.mdp.getTransitionStatesAndProbs(state, action):
                reward = self.mdp.getReward(state, action, nextState)
                dValueOfNextState = self.discount * self.values[nextState]
                qVals += transitionProb * (reward + dValueOfNextState)
            lst.append(qVals)
        return lst

    def runValueIteration(self):
        predecessors = {state: set() for state in self.mdp.getStates()} #Set predecessors}
        #Compute Predecesors of all non terminal states
        for state in self.mdp.getStates():
            #Initialize values
            self.values[state] = 0
            for action in self.mdp.getPossibleActions(state):
                for nextState, transitionProb in self.mdp.getTransitionStatesAndProbs(state, action):
                    if transitionProb != 0:
                        predecessors[nextState].add(state)
        fringe = util.PriorityQueue() #Initialize empty pq

        #Push into priority queue if not terminal state
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            currVal = self.getValue(state)
            qVals = self.calcQVals(state)
            maxQVal = max(qVals)
            diff = abs(currVal - maxQVal)
            fringe.push(state, -diff) #push -diff to PQ
        i = 0

        while (i < self.iterations): #for each iteration
            if fringe.isEmpty(): #if pq empty, terminate
                return

            s = fringe.pop() #pop state s off

            #update s value in self.values if not terminal state
            if not self.mdp.isTerminal(s):
                updateVals = self.calcUpdateVals(s)
                self.values[s] = max(updateVals)

            for p in predecessors[s]:
                currPredVal = self.values[p]
                maxQVal = 0
                qVals = self.calcQVals(p)
                if qVals:
                    maxQVal = max(qVals) #get maxQVal
                diff = abs(currPredVal - maxQVal)
                if diff > self.theta:
                    fringe.update(p, -diff)
            i+=1
