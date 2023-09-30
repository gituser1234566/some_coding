## A policy for treating individuals.
## 
##
## features: gender, age, income, genes, comorbidities, symptoms
## action: vaccines choice or treatment
## outcomes: symptoms (including covid-positive)

import numpy as np
import pandas as pd
from auxilliary import symptom_names

class Policy:
    """ A policy for treatment/vaccination. """
    def __init__(self, n_actions, action_set):
        """ Initialise.

        Args:
        n_actions (int): the number of actions
        action_set (list): the set of actions
        """
        self.n_actions = n_actions
        self.action_set = action_set
        print("Initialising policy with ", n_actions, "actions")
        print("A = {", action_set, "}")
    ## Observe the features, treatments and outcomes of one or more individuals
    def observe(self, features, action, outcomes):
        """Observe features, actions and outcomes.

        Args:
        features (t*|X| array)
        actions (t*|A| array)
        outcomes (t*|Y| array)

        The function is used to adapt a model to the observed
        outcomes, given the actions and features. I suggest you create
        a model that estimates P(y | x,a) and fit it as appropriate.

        If the model cannot be updated incrementally, you can save all
        observed x,a,y triplets in a database and retrain whenever you
        obtain new data.

        Pseudocode:
            self.data.append(features, actions, outcomes)
            self.model.fit(data)

        """
        pass
    def get_utility(self, features, action, outcome):
        """ Obtain the empirical utility of the policy on a set of one or more people. 

        If there are t individuals with x features, and the action
        
        Args:
        features (t*|X| array)
        actions (t*|A| array)
        outcomes (t*|Y| array)

        Returns:
        Empirical utility of the policy on this data.

        """

        return 0
    def get_action(self, features):
        """Get actions for one or more people. 

        Args: 
        features (t*|X| array)

        Returns: 
        actions (t*|A| array)

        Here you should take the action maximising expected utility
        according to your model. This model can be arbitrary, but
        should be adapted using the observe() method.

        Pseudocode:
           for action in appropriate_action_set:
                p = self.model.get_probabilities(features, action)
                u[action] = self.get_expected_utility(action, p)
           return argmax(u)

        You are expected to create whatever helper functions you need.

        """
        return 0

class RandomPolicy(Policy):
    """ This is a purely random policy!"""

    def get_utility(self, features, action, outcome):
        """Here the utiliy is defined in terms of the outcomes obtained only, ignoring both the treatment and the previous condition.
        """
        actions = self.get_action(features)
        utility = 0
        utility -= 0.2 * sum(outcome[:,symptom_names['Covid-Positive']])
        utility -= 0.1 * sum(outcome[:,symptom_names['Taste']])
        utility -= 0.1 * sum(outcome[:,symptom_names['Fever']])
        utility -= 0.1 * sum(outcome[:,symptom_names['Headache']])
        utility -= 0.5 * sum(outcome[:,symptom_names['Pneumonia']])
        utility -= 0.2 * sum(outcome[:,symptom_names['Stomach']])
        utility -= 0.5 * sum(outcome[:,symptom_names['Myocarditis']])
        utility -= 1.0 * sum(outcome[:,symptom_names['Blood-Clots']])
        utility -= 100.0 * sum(outcome[:,symptom_names['Death']])
        return utility
    
    def get_action(self, features):
        """Get a completely random set of actions, but only one for each individual.

        If there is more than one individual, feature has dimensions t*x matrix, otherwise it is an x-size array.
        
        It assumes a finite set of actions.

        Returns:
        A t*|A| array of actions
        """

        n_people = features.shape[0]
        ##print("Acting for ", n_people, "people");
        actions = np.zeros([n_people, self.n_actions])
        for t in range(features.shape[0]):
            action = np.random.choice(self.action_set)
            if (action >= 0):
                actions[t,action] = 1
            
        return actions
    
    
class NewPolicy(Policy):
    def __init__(self, n_actions, action_set, model):
        super().__init__(n_actions, action_set)
        self.model = model
        
        self.weights = {'Covid-Recovered': 0, 
                         'Covid-Positive': .2, 
                         'No-Taste/Smell': .1, 
                         'Fever': .1, 
                         'Headache': .1,
                         'Pneumonia': .5,
                         'Stomach': .2, 
                         'Myocarditis': .5, 
                         'Blood-Clots': 1,
                         'Death': 100}
        self.X = None
        self.y = None
    
    def observe(self, features, action, outcome):
        if self.X is None or self.y is None:
            self.X= pd.DataFrame(columns=list(features) + [key for key in action if key not in features])
            self.y = np.zeros((0, outcome.shape[1]))
        
        self.X = pd.concat((self.X, features.assign(**{str(key):action[key] for key in action})))
        self.y = np.vstack((self.y, np.array(outcome)))

        self.model.fit(self.X, self.y)

        
    def get_utility(self, features, action, outcome):
        """ Assume here that the actions get progressively more costly, so associate somewhat lower
        utility to later actions. Outcomes are weighted more highly than the previous condition here, 
        but it is still better to have had a symptom and then remove it, than just never having had
        the symptom. E.g.
        Had a symptom, now not = 0.7 - 0 = 0.7 utility
        had a symptom, still have it = 0.7 - 1 = -0.3 utility
        never had the symptom = 0 - 0 - 0 utility
        
        This also means that doing actions on people with no symptoms will be associated with negative utility.
        
        Inputs are assumed to contain binary features. 
        """
        utility = 0
        for key in action:
            utility -= sum(action[key]) * np.log(self.action_set.index(key) + 10) / 100
            
        #actions = self.get_action(features)
        for key, weight in self.weights.items():
            utility += weight * sum(0.7 * features.loc[:, key] - outcome.loc[:, key])
        
        return utility
    
    def get_expected_utility(self, action, p):
        utility = np.zeros(p.shape[0])
        utility -= np.log(self.action_set.index(action) + 10) / 100
    
        weights = np.array(list(self.weights.values()))
        
        utility -= (p @ weights.T)
        
        return utility
             
    
    def get_action(self, features):
        """ Get actions for one or more people. 
        Args: 
        features (t*|X| array)
        Returns: 
        actions (t*|A| array)
        """
        features = features.assign(**{str(key):np.zeros(features.shape[0])
                                      for key in self.action_set})
        u = []
        for action in (self.action_set):
            p = self.model.get_probabilities(features, action)
            u.append(self.get_expected_utility(action, p))

        args =  np.argmax(u, axis=0)
        res = np.zeros((features.shape[0], len(self.action_set)))
        res[np.arange(res.shape[0]), args] = 1
        return res


