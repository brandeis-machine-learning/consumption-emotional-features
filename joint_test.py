import statsmodels.api as sm
from scipy.stats import chi2
import pandas as pd

GROUP_TEST = False

data = pd.read_csv('regression/20231208/proposed.csv', index_col=None)
features = data.keys()[4:]
print(data.shape)

if GROUP_TEST:
    test_var = ['1_all_positive_duration','2_all_positive_duration','3_all_positive_duration']
    # test_var = ['1_all_negative_duration','2_all_negative_duration','3_all_negative_duration']
    # test_var = ['1_pos_k','2_pos_k','3_pos_k']
    # test_var = ['1_neg_k','2_neg_k','3_neg_k']
    # test_var = ['1_pos_neg','2_pos_neg','3_pos_neg']
    # test_var = ['1_minus','2_minus','3_minus']
else:
    test_var = ['pre_all_positive_duration', 'pre_all_negative_duration', 'pre_pos_k', 'pre_neg_k',
                '1_all_positive_duration', '1_all_negative_duration', '2_all_positive_duration', '2_all_negative_duration', '3_all_positive_duration', '3_all_negative_duration',
                '1_pos_k', '1_neg_k', '2_pos_k', '2_neg_k', '3_pos_k', '3_neg_k']

rest_var = []
for var in features:
    if var not in test_var:
        rest_var.append(var)


# Assuming you have a linear regression model
model = sm.OLS(data['viewcount'], sm.add_constant(data[test_var + rest_var])).fit()



# ================= F test =================
# # Create a contrast matrix to test the joint significance
# contrast_matrix = []
# if GROUP_TEST:
#     contrast_matrix.append( [0, 1, -1, 0] + [0] * len(rest_var) ) # var1 == var2
#     contrast_matrix.append( [0, 0, -1, 1] + [0] * len(rest_var) ) # var2 == var3
# else:
#     for i in range(len(test_var)//2):
#         cur_constraint = [0] * (len(test_var) + len(rest_var) + 1)
#         cur_constraint[i*2 + 1] = 1
#         cur_constraint[i*2 + 2] = -1
#         contrast_matrix.append(cur_constraint) # pos==neg

# # Perform the Wald test
# wald_test = model.f_test(contrast_matrix)

# # Print the results
# print(wald_test)


# ================= Wald test =================
# null_hypothesis = 'pre_pos_k = 0, 1_pos_k = 0, 2_pos_k = 0, 3_pos_k = 0'
null_hypothesis = 'pre_neg_k = 0, 1_neg_k = 0, 2_neg_k = 0, 3_neg_k = 0'
# null_hypothesis = '1_pos_neg = 0, 2_pos_neg = 0, 3_pos_neg = 0, 1_minus = 0, 2_minus = 0, 3_minus = 0'
wald_test = model.wald_test(null_hypothesis)
print("Wald Test Results:")
print(wald_test)