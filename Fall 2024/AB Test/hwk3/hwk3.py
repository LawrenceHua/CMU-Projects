import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind

# Load the dataset
df = pd.read_csv('HW-3-b.csv')

# Part a) Number of movies and distributions
print("\n===== Part a) Number of Movies and Distributions =====\n")
# Count unique movies
num_movies = df['packagename'].nunique()
print(f"Number of unique movies: {num_movies}")

# Distribution of `order`
plt.figure(figsize=(8, 6))
df['order'].hist(bins=30, edgecolor='black')
plt.title('Distribution of Order')
plt.xlabel('Average Slot Order')
plt.ylabel('Frequency')
plt.savefig('order_distribution.png')  # Save the figure
plt.close()

# Distribution of `n_lease`
plt.figure(figsize=(8, 6))
df['n_lease'].hist(bins=30, edgecolor='black')
plt.title('Distribution of Number of Leases')
plt.xlabel('Number of Leases')
plt.ylabel('Frequency')
plt.savefig('n_lease_distribution.png')  # Save the figure
plt.close()

# Part b) Price elasticity and slot interaction
print("\n===== Part b) Price Elasticity and Slot Interaction =====\n")
# Create interaction term
df['price_order_interaction'] = df['price'] * df['order']

# Fit a Poisson regression model
model = smf.glm(
    formula='n_lease ~ price + order + price_order_interaction + C(mac_type) + C(movie_group)',
    data=df,
    family=sm.families.Poisson()
).fit()

# Output the summary
print("\nPoisson Regression Model (Full):\n")
print(model.summary())

# Bundling slots for simplified analysis
df['slot_group'] = pd.cut(df['order'], bins=[0, 2, 4, 6, 8, 10], labels=['1-2', '3-4', '5-6', '7-8', '9-10'])

# Fit the grouped model
grouped_model = smf.glm(
    formula='n_lease ~ price * C(slot_group) + C(mac_type) + C(movie_group)',
    data=df,
    family=sm.families.Poisson()
).fit()

# Output the summary for grouped analysis
print("\nPoisson Regression Model (Grouped Slots):\n")
print(grouped_model.summary())

# Part c) Does empirical evidence support the hypothesis?
print("\n===== Part c) Hypothesis Testing =====\n")
# Extract relevant coefficients
coeffs = model.params[['price', 'price_order_interaction']]
elasticity = coeffs['price'] + coeffs['price_order_interaction'] * df['order'].mean()
print(f"\nEstimated elasticity at average order: {elasticity}")

# Test hypothesis: Compare left vs right slots
left_slots = df[df['order'] <= 2]['n_lease']
right_slots = df[df['order'] > 2]['n_lease']

# Perform t-test to compare groups
t_stat, p_value = ttest_ind(left_slots, right_slots, equal_var=False)
print(f"\nT-statistic: {t_stat}, P-value: {p_value}")

# Conclusion
if p_value < 0.05:
    print("The empirical evidence supports the hypothesis that left-placed movies have lower price elasticity.")
else:
    print("The empirical evidence does not support the hypothesis that left-placed movies have lower price elasticity.")
