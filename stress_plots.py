import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, kruskal, spearmanr, ranksums

# Load dataset
data = pd.read_csv("C:/Users/nadav lisha/Desktop/Sleep_health_and_lifestyle_dataset.csv")

# Define key variables
occupations = data['Occupation'].dropna().unique()
heart_rate_col = 'Heart Rate'
stress_level_col = 'Stress Level'

# -------------------------------
# STATS: Normality (Shapiro-Wilk)
# -------------------------------
print(f"Checking normality for {heart_rate_col} and {stress_level_col} across occupations")

for occupation in occupations:
    group_heart_rate = data[data['Occupation'] == occupation][heart_rate_col].dropna()
    group_stress_level = data[data['Occupation'] == occupation][stress_level_col].dropna()

    stat, p = shapiro(group_heart_rate)
    print(f"Occupation: {occupation} --> Shapiro-Wilk p-value for {heart_rate_col}: {p:.4f}")

    stat, p = shapiro(group_stress_level)
    print(f"Occupation: {occupation} --> Shapiro-Wilk p-value for {stress_level_col}: {p:.4f}")

# -------------------------------
# STATS: Kruskal-Wallis Test
# -------------------------------
print("\nKruskal-Wallis test for Heart Rate and Stress Level across occupations:")

kruskal_results_heart_rate = []
kruskal_results_stress_level = []

for occupation in occupations:
    group_heart_rate = data[data['Occupation'] == occupation][heart_rate_col].dropna()
    group_stress_level = data[data['Occupation'] == occupation][stress_level_col].dropna()

    kruskal_results_heart_rate.append(group_heart_rate)
    kruskal_results_stress_level.append(group_stress_level)

kruskal_heart_rate = kruskal(*kruskal_results_heart_rate)
print(f"Kruskal-Wallis test for {heart_rate_col}: H = {kruskal_heart_rate.statistic:.4f}, p = {kruskal_heart_rate.pvalue:.4f}")

kruskal_stress_level = kruskal(*kruskal_results_stress_level)
print(f"Kruskal-Wallis test for {stress_level_col}: H = {kruskal_stress_level.statistic:.4f}, p = {kruskal_stress_level.pvalue:.4f}")

# -------------------------------
# STATS: Dunn’s Post-hoc Test
# -------------------------------
def perform_dunns_test(group_data, col_name):
    p_values = []
    occupations = group_data['Occupation'].dropna().unique()
    for i in range(len(occupations)):
        for j in range(i + 1, len(occupations)):
            group_i = group_data[group_data['Occupation'] == occupations[i]][col_name].dropna()
            group_j = group_data[group_data['Occupation'] == occupations[j]][col_name].dropna()
            stat, p = ranksums(group_i, group_j)
            p_values.append(((occupations[i], occupations[j]), p))
    return p_values

print("\nDunn's Test for Heart Rate:")
for pair, p in perform_dunns_test(data, heart_rate_col):
    print(f"{pair[0]} vs {pair[1]}: p = {p:.4f}")

print("\nDunn's Test for Stress Level:")
for pair, p in perform_dunns_test(data, stress_level_col):
    print(f"{pair[0]} vs {pair[1]}: p = {p:.4f}")

# -------------------------------
# STATS: Spearman Correlation
# -------------------------------
print("\nSpearman correlation between Heart Rate and Stress Level by Occupation:")
for occupation in occupations:
    group = data[data['Occupation'] == occupation].dropna(subset=[heart_rate_col, stress_level_col])
    rho, p = spearmanr(group[heart_rate_col], group[stress_level_col])
    print(f"{occupation}: Spearman's rho = {rho:.4f}, p = {p:.4f}")

# -------------------------------
# PLOTTING FUNCTIONS (CI style)
# -------------------------------

# Compute means and 95% CI
def compute_means_ci(df, col):
    summary = df.groupby("Occupation")[col].agg(['mean', 'count', 'std'])
    summary['ci95'] = 1.96 * summary['std'] / np.sqrt(summary['count'])
    return summary.reset_index()

# Plot: Horizontal error bars
def plot_ci_horizontal(summary_df, x_col, x_label, title, filename):
    plt.figure(figsize=(10, 6))
    plt.errorbar(summary_df[x_col], summary_df['Occupation'],
                 xerr=summary_df['ci95'], fmt='o', capsize=5)
    plt.xlabel(x_label)
    plt.ylabel("Occupation")
    plt.title(f"Mean {x_label} by Occupation (with 95% CIs)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, format='jpg', dpi=300)
    plt.show()

# Plot: Combined bivariate plot
def plot_combined_bivariate(hr_summary, stress_summary, filename):
    merged = pd.merge(hr_summary, stress_summary, on='Occupation', suffixes=('_hr', '_stress'))

    plt.figure(figsize=(8, 6))
    for _, row in merged.iterrows():
        plt.errorbar(row['mean_hr'], row['mean_stress'],
                     xerr=row['ci95_hr'], yerr=row['ci95_stress'],
                     fmt='o', color='steelblue', capsize=5)
        plt.text(row['mean_hr'] + 0.2, row['mean_stress'], row['Occupation'],
                 fontsize=8, verticalalignment='center')

    plt.xlabel("Mean Heart Rate")
    plt.ylabel("Mean Stress Level")
    plt.title("Occupation Means: Heart Rate vs. Stress Level")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, format='jpg', dpi=300)
    plt.show()

# -------------------------------
# Generate & Save Plots
# -------------------------------
hr_summary = compute_means_ci(data, heart_rate_col)
stress_summary = compute_means_ci(data, stress_level_col)

plot_ci_horizontal(hr_summary, 'mean', 'Heart Rate', 'Heart Rate', 'heart_rate_ci.jpg')
plot_ci_horizontal(stress_summary, 'mean', 'Stress Level', 'Stress Level', 'stress_level_ci.jpg')
plot_combined_bivariate(hr_summary, stress_summary, 'bivariate_heart_stress.jpg')
