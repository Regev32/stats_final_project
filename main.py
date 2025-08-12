import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy.stats import rankdata, norm
from statsmodels.stats.multitest import multipletests
from statsmodels.multivariate.manova import MANOVA
from statsmodels.miscmodels.ordinal_model import OrderedModel

# ─── Plot: Means with 95% t-based CIs ─────────────────────────────────────────
def plot_ci(df, column):
    g = (df[['Occupation', column]]
         .dropna()
         .groupby('Occupation')
         .agg(mean=(column, 'mean'),
              sd=(column, 'std'),
              n=(column, 'count')))
    g['se'] = g['sd'] / np.sqrt(g['n'])
    g['tcrit'] = stats.t.ppf(0.975, df=g['n'] - 1)
    g['lo'] = g['mean'] - g['tcrit'] * g['se']
    g['hi'] = g['mean'] + g['tcrit'] * g['se']
    g = g.reset_index()

    plt.figure(figsize=(8, 10))
    plt.errorbar(
        g['mean'], g['Occupation'],
        xerr=[g['mean'] - g['lo'], g['hi'] - g['mean']],
        fmt='o', capsize=4
    )
    plt.ylabel("Occupation")
    plt.xlabel(f"{column} (units)")
    plt.title(f"Mean {column} by Occupation (with 95% t-CIs)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ─── Univariate normality per group ───────────────────────────────────────────
def assess_normality(data, column, name, bins=30, threshold=30, figsize=(15, 5)):
    data = np.array(data, dtype=float)
    n = len(data)
    if n <= threshold:
        test_name = "Shapiro-Wilk"
        statistic, p_value = stats.shapiro(data)
    else:
        test_name = "Kolmogorov-Smirnov"
        statistic, p_value = stats.kstest(
            data, 'norm', args=(np.mean(data), np.std(data))
        )
    is_normal = (p_value >= 0.05) if np.isfinite(p_value) else False

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(f'Normality of {column} (n={n}) for {name} — Normal? {is_normal}',
                 fontsize=16, fontweight='bold')

    xs = np.linspace(np.nanmin(data), np.nanmax(data), 200)
    sns.histplot(data, bins=bins, kde=True, stat='density',
                 alpha=0.7, ax=axes[0])
    axes[0].plot(xs, stats.norm.pdf(xs, np.mean(data), np.std(data)),
                 'r--', linewidth=2, label='Normal pdf')
    axes[0].set_title('Histogram + KDE')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    stats.probplot(data, dist="norm", plot=axes[1])
    axes[1].set_title("Q–Q Plot")
    axes[1].grid(True, alpha=0.3)

    axes[2].boxplot(data, patch_artist=True)
    axes[2].set_title('Box Plot')
    axes[2].set_ylabel('Value')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()

    return {
        'test_name': test_name,
        'statistic': statistic,
        'p_value': p_value,
        'is_normal': is_normal,
        'sample_size': n
    }

# ─── Directional Dunn’s tests (one-sided, BH over k(k−1)) ────────────────────
def directional_dunn_test(df, group_col, value_col,
                          alpha=0.05, min_group_size=None,
                          figsize=(12, 10)):
    data = df[[group_col, value_col]].dropna()
    if min_group_size is not None:
        counts = data[group_col].value_counts()
        valid = counts[counts >= min_group_size].index
        data = data[data[group_col].isin(valid)]

    groups = list(data[group_col].unique())
    data = data.copy()
    data['rank'] = rankdata(data[value_col])  # pooled, average ranks for ties

    N = len(data)
    denom_const = N * (N + 1) / 12.0
    stats_df = data.groupby(group_col)['rank'].agg(['mean', 'count'])
    R = stats_df['mean']
    n = stats_df['count']

    p_matrix = pd.DataFrame(np.nan, index=groups, columns=groups)
    for i in groups:
        for j in groups:
            if i == j:
                continue
            z = (R[i] - R[j]) / np.sqrt(denom_const * (1/n[i] + 1/n[j]))
            p_matrix.loc[i, j] = norm.sf(z)  # one-sided H_A: i > j

    pv = p_matrix.values.flatten()
    pv = pv[~np.isnan(pv)]
    corrected = multipletests(pv, alpha=alpha, method='fdr_bh')[1]

    p_matrix_corrected = pd.DataFrame(np.nan, index=groups, columns=groups)
    idx = 0
    for i in groups:
        for j in groups:
            if i == j:
                continue
            p_matrix_corrected.loc[i, j] = corrected[idx]
            idx += 1

    annot = p_matrix_corrected.copy().map(lambda x: f"p={x:.3f}" if pd.notnull(x) else "")
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        p_matrix_corrected.astype(float),
        annot=annot, fmt="", cmap="Blues_r",
        cbar_kws={"label": "corrected one-sided p-value (fdr_bh)"},
        linewidths=0.5, linecolor="lightgray"
    )
    ax.set_title(f"Directional Dunn’s Test ({value_col}: group_i > group_j) with fdr_bh correction")
    ax.set_xlabel("Group j")
    ax.set_ylabel("Group i")

    sig_mask = p_matrix_corrected < alpha
    for i_idx, i in enumerate(groups):
        for j_idx, j in enumerate(groups):
            if i != j and pd.notnull(sig_mask.loc[i, j]) and sig_mask.loc[i, j]:
                ax.add_patch(Rectangle((j_idx, i_idx), 1, 1,
                                       fill=False, edgecolor="black", lw=2))
    plt.tight_layout()
    plt.show()

# ─── Load data once ──────────────────────────────────────────────────────────
df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

# ─── Main loop over univariate outcomes ──────────────────────────────────────
columns = ["Sleep Duration", "Quality of Sleep"]
for column in columns:
    print(f"\n\n=== Processing '{column}' ===\n")
    # 1) Filter n>=5
    df_col = df[['Occupation', column]].dropna()
    counts = df_col['Occupation'].value_counts()
    valid_occs = sorted(counts[counts >= 5].index)
    df_col = df_col[df_col['Occupation'].isin(valid_occs)]
    grouped = df_col.groupby('Occupation')[column]
    samples = [grouped.get_group(occ).values for occ in valid_occs]

    # 2) Normality per group
    normality_results = {}
    for occ, sample in zip(valid_occs, samples):
        res = assess_normality(sample, column, occ)
        normality_results[occ] = {
            'test':      res['test_name'],
            'statistic': res['statistic'],
            'p_value':   res['p_value'],
            'is_normal': res['is_normal'],
            'n':         res['sample_size'],
        }
    normality_df = pd.DataFrame.from_dict(normality_results, orient='index')
    print(normality_df[['n', 'test', 'statistic', 'p_value', 'is_normal']])

    # 3) Omnibus test (always Kruskal–Wallis to match Methods)
    H, p = stats.kruskal(*samples)
    print(f"\nKruskal–Wallis H = {H:.3f}, p = {p:.3f}")

    # 4) Pairwise directional Dunn’s test with fdr_bh
    directional_dunn_test(
        df_col,
        group_col='Occupation',
        value_col=column,
        alpha=0.05,
        min_group_size=5,
        figsize=(14, 12)
    )

    # 5) Plot means ± 95% t-CI
    plot_ci(df_col, column)

# ─── Association of occupation-level means (Duration vs Quality) ─────────────
df_sd_qs = df[['Occupation', 'Sleep Duration', 'Quality of Sleep']].dropna()
counts = df_sd_qs['Occupation'].value_counts()
valid = counts[counts >= 5].index
df_sd_qs = df_sd_qs[df_sd_qs['Occupation'].isin(valid)]

grouped = df_sd_qs.groupby('Occupation')
summary = grouped.agg(
    mean_D=('Sleep Duration', 'mean'),
    sem_D=('Sleep Duration', 'sem'),
    mean_Q=('Quality of Sleep', 'mean'),
    sem_Q=('Quality of Sleep', 'sem'),
)

tcrit = stats.t.ppf(0.975, df=grouped.size() - 1)
summary['lower_D'] = summary['mean_D'] - tcrit * summary['sem_D']
summary['upper_D'] = summary['mean_D'] + tcrit * summary['sem_D']
summary['lower_Q'] = summary['mean_Q'] - tcrit * summary['sem_Q']
summary['upper_Q'] = summary['mean_Q'] + tcrit * summary['sem_Q']

plt.figure(figsize=(8, 8))
plt.errorbar(
    summary['mean_D'], summary['mean_Q'],
    xerr=[summary['mean_D'] - summary['lower_D'], summary['upper_D'] - summary['mean_D']],
    yerr=[summary['mean_Q'] - summary['lower_Q'], summary['upper_Q'] - summary['mean_Q']],
    fmt='o', capsize=5
)
for occ, row in summary.iterrows():
    plt.text(row['mean_D'], row['mean_Q'], occ, va='bottom', ha='right')
plt.xlabel("Mean Sleep Duration (h)")
plt.ylabel("Mean Quality of Sleep")
plt.title("Occupation means: Duration vs. Quality")
plt.grid(True)
plt.tight_layout()
plt.show()

r, p = stats.pearsonr(summary['mean_D'], summary['mean_Q'])
n_groups = summary.shape[0]
z = np.arctanh(r)
se_z = 1 / np.sqrt(n_groups - 3)
z_lo, z_hi = z - 1.96 * se_z, z + 1.96 * se_z
r_lo, r_hi = np.tanh([z_lo, z_hi])
print(f"Pearson r = {r:.3f}, p = {p:.3f}, 95% CI = ({r_lo:.3f}, {r_hi:.3f})")

maov = MANOVA.from_formula('Q("Sleep Duration") + Q("Quality of Sleep") ~ Occupation', data=df_sd_qs)
print(maov.mv_test())

# ─── Ordinal Logit: PAL ~ RCS(Sleep Duration) + covariates ───────────────────
SLEEP_COL = "Sleep Duration"
PAL_COL = "Physical Activity Level"
AGE_COL = "Age"
GENDER_COL = "Gender"
OCC_COL = "Occupation"

counts_all = df[OCC_COL].value_counts()
valid_occs_all = set(counts_all[counts_all >= 5].index)
df_model = df[df[OCC_COL].isin(valid_occs_all)].copy()

def rcs_basis(x, knots, lower, upper):
    """
    Restricted cubic spline (Harrell) basis for a 1D x.
    Returns columns: ['sd', 'sd_rcs1', 'sd_rcs2', ...]
    With m internal knots, produces (m-1) nonlinear columns.
    """
    x = np.asarray(x, dtype=float)
    k = np.array(knots, dtype=float)
    k1, km = lower, upper

    def tp(u):
        u = np.maximum(u, 0.0)
        return u * u * u

    denom = (km - k1)
    Z = []
    for kj in k[:-1]:
        a = (km - kj) / denom
        b = (kj - k1) / denom
        z = tp(x - kj) - a * tp(x - k1) + b * tp(x - km)
        Z.append(z)

    Z = np.column_stack(Z) if len(Z) else np.empty((len(x), 0))
    cols = ['sd'] + [f'sd_rcs{i+1}' for i in range(Z.shape[1])]
    out = np.column_stack([x, Z])
    return pd.DataFrame(out, columns=cols)

def wald_test_joint(res_like, names):
    if not names:
        return None
    pindex = list(res_like.params.index)
    L = np.zeros((len(names), len(pindex)))
    for i, nm in enumerate(names):
        L[i, pindex.index(nm)] = 1.0
    return res_like.wald_test(L, scalar=True)

lb, ub = np.percentile(df_model[SLEEP_COL].astype(float), [5, 95])
internal_knots = sorted([t for t in [6.0, 7.0, 8.0] if lb < t < ub])

X_spline = rcs_basis(df_model[SLEEP_COL].to_numpy(), internal_knots, lb, ub)
# **** IMPORTANT: align indices to avoid misalignment with dummies/ok ****
X_spline.index = df_model.index

X_cov = pd.get_dummies(df_model[[AGE_COL, GENDER_COL, OCC_COL]], drop_first=True)
X = pd.concat([X_spline, X_cov], axis=1)

y_num = pd.to_numeric(df_model[PAL_COL], errors="coerce")
ok = y_num.notna()

X = X.replace([np.inf, -np.inf], np.nan).apply(pd.to_numeric, errors="coerce").astype(float)
ok &= X.notna().all(axis=1)

y_num = y_num[ok]
X = X.loc[ok]

levels = np.sort(y_num.unique())
y = pd.Categorical(y_num, categories=levels, ordered=True)

mod = OrderedModel(y, X, distr="logit")
try:
    res = mod.fit(method="bfgs", disp=False, cov_type="HC1")
    rob = res
except TypeError:
    res = mod.fit(method="bfgs", disp=False)
    rob = res

print(rob.summary())

sleep_all = [c for c in X.columns if c.startswith("sd")]
sleep_nonlin = [c for c in X.columns if c.startswith("sd_rcs")]

wt_overall = wald_test_joint(rob, sleep_all)
print("\nWald test (overall SleepDuration effect):", wt_overall)

if sleep_nonlin:
    wt_nonlin = wald_test_joint(rob, sleep_nonlin)
    print("Wald test (nonlinearity of SleepDuration):", wt_nonlin)

sleep_cols = list(X_spline.columns)
beta_sleep = rob.params.reindex(sleep_cols)
V_sleep = rob.cov_params().loc[sleep_cols, sleep_cols]

def rcs_row(s, knots=internal_knots, lower=lb, upper=ub, cols=X_spline.columns):
    r = rcs_basis(np.array([s]), knots, lower, upper)
    r = r.reindex(columns=cols, fill_value=0.0)
    return r.to_numpy().ravel()

def or_per_hour(s):
    x_s = rcs_row(s)
    x_sp1 = rcs_row(s + 1.0)
    d = (x_sp1 - x_s)
    log_or = float(d @ beta_sleep.to_numpy())
    se = float(np.sqrt(d @ V_sleep.to_numpy() @ d))
    return np.exp(log_or), np.exp(log_or - 1.96 * se), np.exp(log_or + 1.96 * se)

grid = np.linspace(max(lb, 4.5), min(ub, 9.0), 20)
or_curve = pd.DataFrame([(*or_per_hour(s), s) for s in grid],
                        columns=["OR_per_hour", "CI_lo", "CI_hi", "s"])
print("\nHead of OR(s+1 vs s) curve:")
print(or_curve.head())

x_short = rcs_row(5.5)
x_ref   = rcs_row(7.5)
d = (x_short - x_ref)
log_or = float(d @ beta_sleep.to_numpy())
se = float(np.sqrt(d @ V_sleep.to_numpy() @ d))
OR = np.exp(log_or)
CI_lo = np.exp(log_or - 1.96 * se)
CI_hi = np.exp(log_or + 1.96 * se)
print(f"\nContrast <6 h (5.5) vs 7–8 h (7.5): OR={OR:.3f}  95% CI=({CI_lo:.3f}, {CI_hi:.3f})")
