import gymnasium as gym

from deap import base, creator, tools
from landing_tools import forward, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE

import random
import time
from functools import partial
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Network size: 8*16 + 16 + 16*2 + 2 = 178 weights
N_WEIGHTS = (INPUT_SIZE * HIDDEN_SIZE) + HIDDEN_SIZE + (HIDDEN_SIZE * OUTPUT_SIZE) + OUTPUT_SIZE

# ---------------- Genetic Algorithm constants ----------------
POPULATION_SIZE = 200
P_CROSSOVER = 0.7      # probability of crossover for each pair
P_MUTATION = 1.0       # probability of mutating each offspring
MAX_GENERATIONS = 30
T_SIZE = 25            # tournament size (high pressure; try 3-7 to compare)
ELITE_SIZE = 2         # best individuals copied untouched to next generation
HOF_SIZE = 10          # candidates kept for final validation

# ---------------- Seeds ----------------
GA_SEED = 158                         # reproducibility of the GA itself
SEED_POOL = list(range(1, 159))       # training terrains (1..158)
SEEDS_PER_GEN = 5                     # terrains sampled each generation
TEST_SEEDS = list(range(1000, 1100))  # never seen during training

# ---------------- DEAP setup ----------------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -1, 1)
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.attr_float, N_WEIGHTS)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

toolbox.register("select", tools.selTournament, tournsize=T_SIZE)
toolbox.register("mate", tools.cxBlend, alpha=0.5)                          # blend crossover
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.1)   # gaussian mutation

# ---------------- Environment (one per process) ----------------
_env = None

def get_env():
    global _env
    if _env is None:
        _env = gym.make("LunarLander-v3", continuous=True)
    return _env


def run_episode(individual, seed, max_steps=1000):
    env = get_env()
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    for _ in range(max_steps):
        action = forward(obs, individual)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward


def evalLander(individual, seeds):
    return float(np.mean([run_episode(individual, s) for s in seeds])),


def evaluate_population(population, seeds, map_fn):
    fitnesses = map_fn(partial(evalLander, seeds=seeds), population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit


# ---------------- Main loop ----------------
def main():
    random.seed(GA_SEED)
    np.random.seed(GA_SEED)
    seed_rng = random.Random(GA_SEED + 1)   # separate stream for terrain sampling

    population = toolbox.populationCreator(n=POPULATION_SIZE)
    hof = tools.HallOfFame(HOF_SIZE)
    max_hist, mean_hist = [], []

    with Pool() as pool:
        seeds = seed_rng.sample(SEED_POOL, SEEDS_PER_GEN)
        evaluate_population(population, seeds, pool.map)
        hof.update(population)

        for gen in range(1, MAX_GENERATIONS + 1):
            elites = list(map(toolbox.clone, tools.selBest(population, ELITE_SIZE)))
            offspring = list(map(toolbox.clone, toolbox.select(population, POPULATION_SIZE - ELITE_SIZE)))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < P_CROSSOVER:
                    toolbox.mate(child1, child2)

            for mutant in offspring:
                if random.random() < P_MUTATION:
                    toolbox.mutate(mutant)

            population[:] = elites + offspring

            # New terrains every generation -> EVERYONE is re-evaluated (elites too),
            # so all fitness values in a generation are measured on the same seeds.
            seeds = seed_rng.sample(SEED_POOL, SEEDS_PER_GEN)
            evaluate_population(population, seeds, pool.map)
            hof.update(population)

            fits = [ind.fitness.values[0] for ind in population]
            max_hist.append(max(fits))
            mean_hist.append(float(np.mean(fits)))
            print(f"Gen {gen:3d} | best {max(fits):7.1f} | mean {np.mean(fits):7.1f} | seeds {seeds}")

        # Fitness in the HOF comes from different seed sets (noisy), so pick the
        # champion by re-evaluating every candidate on the full training pool.
        val = pool.map(partial(evalLander, seeds=SEED_POOL), hof)
        val_scores = [v[0] for v in val]
        best = hof[int(np.argmax(val_scores))]
        print(f"\nValidation (158 training seeds): {max(val_scores):.1f}")

        # Final honest number: terrains never seen during training
        test_scores = np.array(pool.map(partial(run_episode, best), TEST_SEEDS))

    print(f"Test ({len(TEST_SEEDS)} unseen seeds): {test_scores.mean():.1f} ± {test_scores.std():.1f}")
    print(f"Successful landings (>=200): {np.mean(test_scores >= 200):.0%}")

    np.save("best_weights.npy", np.array(best))

    # ---------------- Plot ----------------
    sns.set_style("whitegrid")
    plt.plot(max_hist, color="red", label="Best")
    plt.plot(mean_hist, color="green", label="Average")
    plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("Fitness (mean reward over sampled seeds)")
    plt.ylim(top=400)
    plt.suptitle("Best and Average Fitness over Generations")
    plt.title(f"Pop: {POPULATION_SIZE}, P_cx: {P_CROSSOVER}, P_mut: {P_MUTATION}, "
              f"Tournament: {T_SIZE}, Elites: {ELITE_SIZE}, Seeds/gen: {SEEDS_PER_GEN}", size=9)
    plt.savefig(f"training_{time.strftime('%Y%m%d_%H%M%S')}.png", dpi=150)


if __name__ == "__main__":
    main()