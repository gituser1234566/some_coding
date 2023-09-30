## To-Do

import policy
import simulator
from policy import NewPolicy

class MLPModel(MLPClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def get_probabilities(self, features, action):
        features = features.assign(**{action:np.ones(features.shape[0])})

        return self.predict_proba(features)


## Baseline simulator parameters
n_genes = 128
n_vaccines = 3
n_treatments = 4
n_population = 10000

# symptom names for easy reference
from auxilliary import symptom_names


# Create the underlying population
print("Generating population")
population = simulator.Population(n_genes, n_vaccines, n_treatments)
X = population.generate(n_population)

# Make sure that your policy appropriately filters out the population if necessary. This is just a random sample of 1000 people


# Generate vaccination results

print("Vaccination")
vaccine_policy = NewPolicy(n_vaccines, list(range(-1,n_vaccines)), 
                           MLPModel(activation='logistic', random_state=0)) # make sure to add -1 for 'no vaccine'


print("With a for loop")
# The simplest way to work is to go through every individual in the population
for t in range(n_population):
    a_t = vaccine_policy.get_action(X[t])
    # Then you can obtain results for everybody
    y_t = population.vaccinate([t], a_t)
    # Feed the results back in your policy. This allows you to fit the
    # statistical model you have.
    vaccine_policy.observe(X[t], a_t, y_t)

print("Vaccinate'em all")
# Here you can get an action for everybody in the population
A = vaccine_policy.get_action(X)
# Then you can obtain results for everybody
Y = population.vaccinate(list(range(n_population)), A)
# Feed the results back in your policy. 
vaccine_policy.observe(X, A, Y)



# You can do the same iteratively, where you select some to treat, and the rest remain untreated. The following implements a 3-phase vaccination policy.
print("3-phase policy")
for k in range(3):
    A = vaccine_policy.get_action(X)
    # Then you can obtain results for everybody
    Y = population.vaccinate(list(range(n_population)), A)
    # Feed the results back in your policy
    vaccine_policy.observe(X, A, Y)



    

# We can do the same for treatments
print("Treatment")
treatment_policy = NewPolicy(n_treatments, list(range(n_treatments)),
                                   MLPModel(activation='logistic', random_state=0))

A = treatment_policy.get_action(X)
Y = population.treat(list(range(n_population)), A)
treatment_policy.observe(X, A, Y)


