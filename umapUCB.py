import numpy as np
from typing import List
import pymc3 as pm


def umap_ucb(arms: List[List[float]], num_rounds: int, prior) -> int:
    noise = 0.01
    num_arms = len(arms)
    num_objectives = len(arms[0])
    num_comp = 0

    # previous comparisons
    C = []

    # initialize reward w/ single pull of arms
    rewards_list = [[0.0] * num_objectives for _ in range(num_arms)]
    for a in range(num_arms):
        rewards_list[a] = [arms[a][i] + np.random.normal(0, noise) for i in range(num_objectives)]

    # n_a
    num_pulls = [1] * num_arms

    upper_bounds = [[float('inf')] * num_objectives for _ in range(num_arms)]
    upper_bounds_exb = [[float('inf')] * num_objectives for _ in range(num_arms)]

    for t in range(num_arms, num_rounds + 1):
        if not C:
            weights = prior
        else:
            weights = map_weight_estimate(C)

        for a in range(num_arms):
            upper_bounds[a] = np.dot(np.array(rewards_list[a]), weights)
            exb = exploration_bonus(num_pulls[a], t)
            upper_bounds_exb[a] = np.dot(np.array(rewards_list[a]), weights) + exb

        a_star = np.argmax(upper_bounds)
        a_best = np.argmax(upper_bounds_exb)
        print(upper_bounds)
        print(upper_bounds_exb)
        print("a_star: " + str(a_star))
        print("a_best: " + str(a_best))

        reward = [arms[a_best][i] + np.random.normal(0, noise) for i in range(num_objectives)]  # add noise
        pulls = num_pulls[a_best]
        for d in range(num_objectives):
            rewards_list[a_best][d] = (reward[d] + (rewards_list[a_best][d] * pulls)) / (pulls + 1)
        num_pulls[a_best] += 1

        if a_star != a_best:
            num_comp += 1
            reward_star = [arms[a_star][i] + np.random.normal(0, noise) for i in range(num_objectives)]
            print("Please choose the better result:")
            print("1: " + str(reward_star))
            print("2: " + str(reward))

            # user_input = input("Enter 1 or 2: ")
            user_input = utility(reward_star, reward)
            if user_input == "1":
                C.append([reward_star, reward])
            else:
                C.append([reward, reward_star])

    print("Upper bounds:")
    print(upper_bounds)
    print("Pulls:")
    print(num_pulls)

    return num_comp


def exploration_bonus(n_a, t):
    return np.sqrt((2 * np.log(t)) / n_a)


def exploration_bonus_ch(num_iterations, avg_reward, num_selected):
    sqrt_iter = np.sqrt(num_iterations)
    return np.sqrt((2 * avg_reward * np.log(sqrt_iter)) / num_selected) + (2 * np.log(sqrt_iter) / num_selected)


def project_onto_simplex(w):
    d = len(w)
    mu = np.sort(w)[::-1]
    j = 1

    while j < d and mu[j] + 1.0 / (j + 1) * (1 - np.sum(mu[:j + 1])) > 0:
        j += 1

    rho = 1.0 / j * (1 - np.sum(mu[:j]))
    proj_w = np.maximum(w + rho, 0)
    return proj_w / np.sum(proj_w)


def map_weight_estimate(C):
    X = np.array(C)
    # Flatten the arm comparisons to create X and y for logistic regression
    X = X.reshape(-1, 2)  # Reshape to (n_comparisons, 2)
    print(X)
    y = np.zeros(X.shape[0])  # Create binary labels (0 = second vector preferred)
    y[:X.shape[0] // 2] = 1  # Set labels to 1 for first vector preferred

    with pm.Model() as logistic_model:
        # Priors for weights and intercept
        weights = pm.Normal('weights', mu=0, sd=1, shape=X.shape[1])
        intercept = pm.Normal('intercept', mu=0, sd=1)

        # Logistic regression equation
        logit_p = pm.math.dot(X, weights) + intercept

        # Likelihood
        likelihood = pm.Bernoulli('y', logit_p=logit_p, observed=y)

        # Maximum a posteriori estimation
        map_estimate = pm.find_MAP(model=logistic_model)

    # Access the MAP estimates
    map_weights = map_estimate['weights']
    weights = project_onto_simplex(map_weights)
    print("map projected weights")
    print(weights)

    return weights


def utility(v1, v2):
    u = [0.7, 0.3]
    if np.dot(v1, u) > np.dot(v2, u):
        return 1
    else:
        return 2


# Test
arms = [[0.1, 0.1], [0.8, 0.2], [0.8, 0.1], [0.5, 0.7], [0.4, 0.6], [0.6, 0.4], [0.3, 0.6]]
num_rounds = 2000
umap_ucb(arms, num_rounds, [0.7, 0.3])
print()
