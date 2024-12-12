import pandas as pd

data = pd.read_csv('HW-1-a.csv') 

#A
# Count total students
unique_students = data['stu_id'].nunique()

# Count students in each condition
students_per_condition = data['treatment'].value_counts()
print("\nPart A:")
print("Total # of students:", unique_students)
print("Students per condition:\n", students_per_condition)

#B
# Calculate improvement in scores
data['improvement'] = data['postscore'] - data['prescore']

# Generate descriptive statistics for improvement
improvement_stats = data['improvement'].describe()
print("\nPart B:")
print("Descriptive statistics for improvement in test scores:\n", improvement_stats)

#C
# Calculate mean improvement by treatment condition
mean_improvement_by_treatment = data.groupby('treatment')['improvement'].mean()

# Calculate the difference in mean improvement between B and ANL
reduction_in_improvement_ANL = mean_improvement_by_treatment['B'] - mean_improvement_by_treatment['ANL']
print("\nPart C:")
print("Mean improvement for B:", mean_improvement_by_treatment['B'])
print("Mean improvement for ANL:", mean_improvement_by_treatment['ANL'])
print("Reduction in improvement for ANL vs B:", reduction_in_improvement_ANL)

#D
# Calculate the difference in mean improvement between B and AML
increase_in_improvement_AML = mean_improvement_by_treatment['AML'] - mean_improvement_by_treatment['B']
print("\nPart D:")
print("Mean improvement for AML:", mean_improvement_by_treatment['AML'])
print("Mean improvement for B:", mean_improvement_by_treatment['B'])
print("Change in improvement for AML vs B:", increase_in_improvement_AML)

#E
from scipy.stats import ttest_ind

t_test_B_ANL = ttest_ind(data[data['treatment'] == 'B']['improvement'], 
                         data[data['treatment'] == 'ANL']['improvement'], equal_var=False)
t_test_B_AML = ttest_ind(data[data['treatment'] == 'B']['improvement'], 
                         data[data['treatment'] == 'AML']['improvement'], equal_var=False)
t_test_ANL_AML = ttest_ind(data[data['treatment'] == 'ANL']['improvement'], 
                           data[data['treatment'] == 'AML']['improvement'], equal_var=False)
print("\nPart E:")
print("T-test results for B vs ANL:")
print("  t-statistic:", t_test_B_ANL.statistic)
print("  p-value:", t_test_B_ANL.pvalue)

print("\nT-test results for B vs AML:")
print("  t-statistic:", t_test_B_AML.statistic)
print("  p-value:", t_test_B_AML.pvalue)

print("\nT-test results for ANL vs AML:")
print("  t-statistic:", t_test_ANL_AML.statistic)
print("  p-value:", t_test_ANL_AML.pvalue)