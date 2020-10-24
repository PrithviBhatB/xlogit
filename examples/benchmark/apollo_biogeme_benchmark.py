import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os


def profile_range_draws_and_cores(command, r_draws, r_cores):
    print("\n\n=== artificial dataset. "+command.split()[1] +" ===")
    for n_draws in r_draws:
        for n_cores in r_cores:
            os.system("{} {} {}".format(command, n_draws, n_cores))

r_draws = [100, 750, 1500]
r_cores = [16, 32, 64]


#profile_range_draws_and_cores("python biogeme_run.py", r_draws, r_cores)
#os.environ['OPENBLAS_NUM_THREADS'] = "1" # Avoids segfault error
#profile_range_draws_and_cores("Rscript apollo_run.R", r_draws, r_cores)

# Plot results
df = pd.read_csv("results/apollo_biogeme_benchmark_results.csv")

libs = ['apollo', 'biogeme']
matplotlib.rcParams.update({'font.size': 14,
                            'axes.spines.right': False,
                            'axes.spines.top': False})
markers = ['s', '|', '^', 'x', 'o']
lines = ['-', '--']
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
def plot_time_benchmark(df):
    plt.figure()
    for l, lib in enumerate(libs):
        for m, cores in enumerate(r_cores):
            idx = (df.library == lib) & (df.cores == cores)
            d = df[idx][["draws", "time"]].values.T
            plt.plot(d[0], d[1], marker=markers[m], linestyle=lines[l],
                c=colors[m], label = "{} {} cores".format(lib, cores))
    plt.legend()
    plt.xlabel("Random draws")
    plt.ylabel("Time (Seconds)")
    plt.title("Estimation time")
    plt.savefig("results/time_benchmark_apolllo_biogeme", dpi=300)
    plt.show()

plot_time_benchmark(df)
