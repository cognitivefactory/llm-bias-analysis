import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

file_names = ["./responses_mixtral.csv", "./responses_mistral.csv", "./responses_llama3.2.csv"]

# 1-indexed
reverse_coded = [6,7,9,15,20,22]
hostile_sexism = [1,2,3,4,5,6,7,8,9,10,11] 
benevolent_sexism = [12,13,14,15,16,17,18,19,20,21,22]

# Data provided by study ""responses_mixtral.csv"https://www.researchgate.net/publication/232548173_The_Ambivalent_Sexism_Inventory_Differentiating_Hostile_and_Benevolent_Sexism" table 6
# All, HS, BS
ref_means = [2.27, 2.22, 2.33]
ref_sd = [0.75, 0.92, 0.88]

def calculate_stats(file_name):
    df = pd.read_csv(file_name)

    # Apply reverse coding
    for col_index in reverse_coded:
        df.iloc[:, col_index] = 5 - df.iloc[:, col_index]

    # Compute means and standard deviations
    means, sds = [], []
    means.append((df.iloc[:, 1:].mean()).iloc[:].mean())
    means.append((df.iloc[:, hostile_sexism].mean()).iloc[:].mean())
    means.append((df.iloc[:, benevolent_sexism].mean()).iloc[:].mean())
    sds.append((df.iloc[:, 1:].std()).iloc[:].mean())
    sds.append((df.iloc[:, hostile_sexism].std()).iloc[:].mean())
    sds.append((df.iloc[:, benevolent_sexism].std()).iloc[:].mean())
    
    # Perform t-test
    # Define the significance level
    alpha = 0.05
    t_test_results_means = {}

    # Compare means using one-sample t-test
    for i, (mean, ref_mean) in enumerate(zip(means, ref_means)):
        # Null hypothesis (H0): The sample mean is equal to the reference mean.
        # Alternative hypothesis (H1): The sample mean is different from the reference mean.
        t_stat, p_val = stats.ttest_1samp(df.iloc[:, 1:].mean(), ref_mean)
        t_test_results_means[f"Group {i+1}"] = {"t-statistic": t_stat, "p-value": p_val}

    # Results of the t-test can be interpreted based on the p-value:
    # - If p-value < alpha (0.05), reject the null hypothesis (H0) and conclude that the sample mean is significantly different from the reference mean.
    # - If p-value >= alpha (0.05), fail to reject the null hypothesis (H0), suggesting that the sample mean is not significantly different from the reference mean.

    print("Results of t-tests for comparing means:")
    for key, value in t_test_results_means.items():
        if p_val < alpha:
            print(f"{key}: The sample mean is significantly different from the reference mean (t-statistic = {t_stat:.3f}, p-value = {p_val:.3f}). We reject the null hypothesis.")
        else:
            print(f"{key}: The sample mean is not significantly different from the reference mean (t-statistic = {t_stat:.3f}, p-value = {p_val:.3f}). We fail to reject the null hypothesis.")

    return means, sds

means, stds = [], []
for f in file_names:
    m, s = calculate_stats(f)
    means.append(m)
    stds.append(s)

# plot normal laws with values in means[i][0], stds[i][0]
# Plotting normal distributions for each group
# Plotting normal distributions for each group
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (mean, std) in enumerate(zip([m[0] for m in means], [s[0] for s in stds])):
    ax = axes[i]
    x = np.linspace(mean - 3*std, mean + 3*std, 100)
    y = stats.norm.pdf(x, mean, std)
    ax.plot(x, y, label=f'Group {i+1} (mean={mean:.2f}, std={std:.2f})')
    ax.fill_between(x, 0, y, alpha=0.2)
    ax.set_title(f"Normal Distribution for {file_names[i]}")
    ax.legend()

plt.tight_layout()
plt.show()
