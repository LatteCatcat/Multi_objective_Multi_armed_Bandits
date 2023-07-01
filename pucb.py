import numpy as np
from typing import List


def p_ucb1(arms: List[List[float]], num_rounds: int) -> List[int]:
    num_arms = len(arms)
    num_objectives = len(arms[0])
    num_pulls = [0] * num_arms
    rewards_list = [[0.0] * num_objectives for _ in range(num_arms)]
    upper_bounds = [[float('inf')] * num_objectives for _ in range(num_arms)]
    noise = 0.01

    for t in range(1, num_rounds + 1):
        candidate_set = []
        for a in range(num_arms):
            for d in range(num_objectives):
                if num_pulls[a] != 0:
                    upper_bounds[a][d] = rewards_list[a][d] + exploration_bonus(t, num_objectives, num_arms, num_pulls[a])

            #print("Upper bounds:")
            #print(upper_bounds)

            update_candidate_set(a, num_objectives, candidate_set, upper_bounds)

        # print("Candidates:")
        # print(candidate_set)
        selected_arm = candidate_set[np.random.randint(0, len(candidate_set))]
        # print("Selected arm:")
        # print(selected_arm)
        reward = [arms[selected_arm][i] + np.random.normal(0, noise) for i in range(num_objectives)]  # add noise
        pulls = num_pulls[selected_arm]
        for d in range(num_objectives):
            rewards_list[selected_arm][d] = (reward[d] + (rewards_list[selected_arm][d] * pulls)) / (pulls + 1)
        num_pulls[selected_arm] += 1
        # print(candidate_set)

    print("Upper bounds:")
    print(upper_bounds)
    print("Pulls:")
    print(num_pulls)

    return candidate_set


def update_candidate_set(arm, num_objectives, candidate_set, upper_bounds):
    for c in candidate_set:
        count = 0
        for d in range(num_objectives):
            if upper_bounds[arm][d] < upper_bounds[c][d]:
                count += 1
        if count >= num_objectives:
            return
        elif count == 0:
            candidate_set.remove(c)
    candidate_set.append(arm)


def exploration_bonus(num_iterations, num_objectives, num_arms, num_selected):
    return np.sqrt((2 * np.log(num_iterations) * np.power(num_objectives * num_arms, 1 / 4)) / num_selected)


# Test
arms = [[0, 0], [0.8, 0.2], [0.8, 0.1], [0.5, 0.7], [0.4, 0.6], [0.6, 0.4], [0.3, 0.6]]
num_rounds = 100000
c_set = p_ucb1(arms, num_rounds)
print("Candidate set:", c_set)
