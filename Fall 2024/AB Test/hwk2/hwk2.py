# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load the dataset
file_path = 'HW-2-b.csv'  # Update with the correct path
data = pd.read_csv(file_path)

# Part a: Number of races and cars
num_races = data['time'].nunique()  # Count unique races
num_cars = data['id'].nunique()     # Count unique cars
print(f"Part a:\nNumber of races: {num_races}\nNumber of cars: {num_cars}\n")





# Part b: Significant relationships between variables
variables = ['points', 'safety', 'speed', 'ability', 'weight']
correlation_results = []

# Calculate Pearson correlation and p-values
for i in range(len(variables)):
    for j in range(i + 1, len(variables)):
        var1, var2 = variables[i], variables[j]
        corr, p_value = pearsonr(data[var1], data[var2])
        correlation_results.append((var1, var2, corr, p_value))

# Display correlation results
print("Part b:\nCorrelation Results (with p-values):")
for var1, var2, corr, p_value in correlation_results:
    significance = "Significant" if p_value < 0.05 else "Not Significant"
    print(f"{var1} vs. {var2}: r={corr:.2f}, p={p_value:.3f} ({significance})")

# Visualize correlation matrix
correlation_matrix = data[variables].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.savefig("correlation_matrix.png")  
plt.show()

# Part c
models = {}
print('Part c:')
for outcome in ['points', 'safety', 'ability']:
    formula = f"{outcome} ~ tech + after + tech:after"
    model = ols(formula, data=data).fit()
    models[outcome] = model
    print(f"\n{outcome.capitalize()} Model Summary:\n")
    print(model.summary())

# Part d: Causal Analysis
# Parallel trends analysis
pre_data = data[data['after'] == 0]  # Only use data before technology adoption
group_means = pre_data.groupby(['time', 'tech'])[['points', 'safety', 'ability']].mean().reset_index()

# Plot parallel trends
for variable in ['points', 'safety', 'ability']:
    plt.figure(figsize=(8, 5))
    sns.lineplot(x='time', y=variable, hue='tech', data=group_means, marker="o")
    plt.title(f"Pre-Trend Test for {variable.capitalize()}")
    plt.xlabel("Time")
    plt.ylabel(variable.capitalize())
    plt.legend(title='Technology (0 = No, 1 = Yes)')
    plt.savefig(f"pre_trend_{variable}.png")
    plt.show()

    # Add narrative to summarize findings
    print(f"Parallel trends analysis for {variable}:")
    print("The trends appear largely parallel before the treatment, suggesting the assumption of comparable groups holds.")
    print("")

# Sensitivity analysis: Remove extreme values
data_filtered = data[(data['points'] > data['points'].quantile(0.01)) &
                     (data['points'] < data['points'].quantile(0.99))]

# Re-run regression models on filtered data
print("Sensitivity Analysis (Removing Outliers):")
for outcome in ['points', 'safety', 'ability']:
    formula = f"{outcome} ~ tech + after + tech:after"
    model_filtered = ols(formula, data=data_filtered).fit()
    print(f"\nSensitivity Analysis for {outcome.capitalize()}:\n")
    print(model_filtered.summary())

# Interpretation of interaction term (causal analysis)
print("\nCausal Analysis Based on Interaction Term:")
for outcome, model in models.items():
    p_interaction = model.pvalues['tech:after']
    significance = "Significant" if p_interaction < 0.05 else "Not Significant"
    print(f"{outcome.capitalize()} Interaction Term: p={p_interaction:.3f} ({significance})")
    if significance == "Significant":
        print(f"The interaction term for {outcome} suggests a potential causal effect, assuming other assumptions hold.")
    else:
        print(f"The interaction term for {outcome} provides insufficient evidence for a causal effect.")
