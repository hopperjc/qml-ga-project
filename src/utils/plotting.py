import os
import matplotlib.pyplot as plt

def plot_fitness(history, out_path: str):
    plt.figure()
    plt.plot(history)
    plt.xlabel("Iteração / Geração")
    plt.ylabel("Objetivo (loss ou fitness)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
