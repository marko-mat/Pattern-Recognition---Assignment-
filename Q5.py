# Marko Matijevic - 40282591
import numpy as np
import pandas as pd

data = {
    'w1': {'x1': [-5.01, -5.43, 1.08, 0.86, -2.67, 4.94, -2.51, -2.25, 5.56, 1.03],
           'x2': [-8.12, -3.48, -5.52, -3.78, 0.63, 3.29, 2.09, -2.13, 2.86, -3.33],
           'x3': [-3.68, -3.54, 1.66, -4.11, 7.39, 2.08, -2.59, -6.94, -2.26, 4.33]},
    'w2': {'x1': [-0.91, 1.30, -7.75, -4.11, 6.14, 3.60, 5.37, 7.18, -7.39, -7.50],
           'x2': [-0.18, -2.06, -4.54, 0.50, 5.72, 1.26, -4.63, 1.46, 1.17, -6.32],
           'x3': [-0.05, -3.53, -0.95, 4.92, -4.85, 4.36, -3.65, -6.66, 6.30, -0.31]},
}

#dataframes for each
w1_df = pd.DataFrame(data['w1'])
w2_df = pd.DataFrame(data['w2'])

#SAMPLE INPUT
x = data['w1']['x1'][0]
#--------------------------------------------------------------------------------------------------------------------
#
#                                                   FUNCTIONS
#
#--------------------------------------------------------------------------------------------------------------------
def custom_mean(X):
    return np.sum(X, axis=0) / X.shape[0]
def custom_cov(X): #formula: (x_i - mean) (x_i - mean)^T
    n = X.shape[0]
    mean_X = custom_mean(X)
    cov_matrix = np.zeros((X.shape[1], X.shape[1]))
    for i in range(n):
        diff = X[i, :] - mean_X
        cov_matrix += np.outer(diff, diff)
    cov_matrix /= n
    return cov_matrix

def g_i(x, mu_i, cov_i):
    if np.isscalar(mu_i):  #single feature
        var_i = cov_i
        term1 = -0.5 * ((x - mu_i) ** 2) / var_i
        term2 = -0.5 * np.log(2 * np.pi * var_i)
        term3 = 0
        term4 = 0
    else:  #multivariate case
        d = len(mu_i)
        cov_inv = np.linalg.inv(cov_i)
        diff = x - mu_i
        term1 = -0.5 * np.dot(np.dot(diff.T, cov_inv), diff)
        term2 = -0.5 * d * np.log(2 * np.pi)
        term3 = -0.5 * np.log(np.linalg.det(cov_i))
        term4 = np.log(0.5)

    return term1 + term2 +term3 + term4


#--------------------------------------------------------------------------------------------------------------------
#
#                                        means and variances -  x1
#
#--------------------------------------------------------------------------------------------------------------------

mu1_x1 = np.sum(data['w1']['x1']) / len(data['w1']['x1'])
mu2_x1 = np.sum(data['w2']['x1']) / len(data['w2']['x1'])
var1_x1 = custom_cov(np.array(data['w1']['x1']).reshape(-1, 1))[0, 0]
var2_x1 = custom_cov(np.array(data['w2']['x1']).reshape(-1, 1))[0, 0]
#----------------------------------------------------------------------------------------------------------------------
#
#                                  mean and covariance matrices -  x1 and x2
#
#--------------------------------------------------------------------------------------------------------------------

X_w1_x12 = np.array([data['w1']['x1'], data['w1']['x2']]).T
X_w2_x12 = np.array([data['w2']['x1'], data['w2']['x2']]).T

mean_w1_x12 = custom_mean(X_w1_x12)
mean_w2_x12 = custom_mean(X_w2_x12)
cov_w1_x12 = custom_cov(X_w1_x12)
cov_w2_x12 = custom_cov(X_w2_x12)
#---------------------------------------------------------------------------------------------------------------------
#
#                                  mean  and covariance -  x1, x2, and x3
#
#--------------------------------------------------------------------------------------------------------------------
X_w1_x123 = np.array([data['w1']['x1'], data['w1']['x2'], data['w1']['x3']]).T
X_w2_x123 = np.array([data['w2']['x1'], data['w2']['x2'], data['w2']['x3']]).T

mean_w1_x123= custom_mean(X_w1_x123)
mean_w2_x123 = custom_mean(X_w2_x123)
cov_w1_x123= custom_cov(X_w1_x123)
cov_w2_x123 = custom_cov(X_w2_x123)
#---------------------------------------------------------------------------------------------------------------------
#
#                                                        X1
#
#---------------------------------------------------------------------------------------------------------------------
print("\nFor X1")
w1_x1 = np.array(data['w1']['x1'])
w2_x1 = np.array(data['w2']['x1'])
predicted_w1 = [1 if g_i(x, mu1_x1, var1_x1) > g_i(x, mu2_x1, var2_x1) else 2 for x in w1_x1]
predicted_w2 = [1 if g_i(x, mu1_x1, var1_x1) > g_i(x, mu2_x1, var2_x1) else 2 for x in w2_x1]
#print(predicted_w1)
#print(predicted_w2)
#misclassified calculations
misclassified_w1 = np.sum(np.array(predicted_w1) != 1)
misclassified_w2 = np.sum(np.array(predicted_w2) != 2)
total_misclassified = misclassified_w1 + misclassified_w2
#print(total_misclassified)
total_samples = len(w1_x1) + len(w2_x1)
#print(total_samples)
training_error = (total_misclassified / total_samples) * 100
print(f"Empirical Training Error: {training_error}%")
#Bhattacharyya distance
DB = (1/8) * ((mu1_x1 - mu2_x1) ** 2) * (1/var1_x1 + 1/var2_x1) + 0.5 * np.log((var1_x1 + var2_x1) / (2 * np.sqrt(var1_x1 * var2_x1)))
#Bhattacharyya bound on error
DBerr_X1 = 0.5 * np.exp(-DB)
print(f"Bhattacharyya Bound on Error: {DBerr_X1:.3f}")

#---------------------------------------------------------------------------------------------------------------------
#
#                                                    X1 & X2
#
#---------------------------------------------------------------------------------------------------------------------

print("\nFor X1 & X2")
samples_w1_x12 = np.array(list(zip(data['w1']['x1'], data['w1']['x2'])))
samples_w2_x12 = np.array(list(zip(data['w2']['x1'], data['w2']['x2'])))
predicted_w1_x12 = [1 if g_i(x, mean_w1_x12, cov_w1_x12) >
                    g_i(x, mean_w2_x12, cov_w2_x12) else 2 for x in samples_w1_x12]
predicted_w2_x12 = [1 if g_i(x, mean_w1_x12, cov_w1_x12) >
                    g_i(x, mean_w2_x12, cov_w2_x12) else 2 for x in samples_w2_x12]
#misclassified calculations
misclassified_w1_x12 = np.sum(np.array(predicted_w1_x12) != 1)
misclassified_w2_x12 = np.sum(np.array(predicted_w2_x12) != 2)
total_misclassified_x12 = misclassified_w1_x12 + misclassified_w2_x12
total_samples_x12 = len(samples_w1_x12) + len(samples_w2_x12)
training_error_x12 = (total_misclassified_x12 / total_samples_x12) * 100
print(f"Empirical Training Error: {training_error_x12}%")

#Bhattacharyya distance
mean_diff_x12 = mean_w1_x12 - mean_w2_x12
cov_mean_x12 = (cov_w1_x12 + cov_w2_x12) / 2
cov_mean_inv_x12 = np.linalg.inv(cov_mean_x12)
DB_x12 = 0.125 * mean_diff_x12.T @ cov_mean_inv_x12 @ mean_diff_x12 + \
         0.5 * np.log(np.linalg.det(cov_mean_x12) / np.sqrt(np.linalg.det(cov_w1_x12) * np.linalg.det(cov_w2_x12)))

#Bhattacharyya bound on error
DBerr_X12 = 0.5 * np.exp(-DB_x12)
print(f"Bhattacharyya Bound on Error: {DBerr_X12:.3f}")
#---------------------------------------------------------------------------------------------------------------------
#
#                                                    X1 & X2 & X3
#
#---------------------------------------------------------------------------------------------------------------------
print("\nFor X1 & X2 & X3")

samples_w1_x123 = np.array(list(zip(data['w1']['x1'], data['w1']['x2'], data['w1']['x3'])))
samples_w2_x123 = np.array(list(zip(data['w2']['x1'], data['w2']['x2'], data['w2']['x3'])))
predicted_w1_x123 = [1 if g_i(x, mean_w1_x123, cov_w1_x123) >
                     g_i(x, mean_w2_x123, cov_w2_x123) else 2 for x in samples_w1_x123]
predicted_w2_x123 = [1 if g_i(x, mean_w1_x123, cov_w1_x123) >
                     g_i(x, mean_w2_x123, cov_w2_x123) else 2 for x in samples_w2_x123]
#misclassified calculations
misclassified_w1_x123 = np.sum(np.array(predicted_w1_x123) != 1)
misclassified_w2_x123 = np.sum(np.array(predicted_w2_x123) != 2)
total_misclassified_x123 = misclassified_w1_x123 + misclassified_w2_x123
total_samples_x123 = len(samples_w1_x123) + len(samples_w2_x123)
training_error_x123 = (total_misclassified_x123 / total_samples_x123) * 100
print(f"Empirical Training Error: {training_error_x123}%")

#Bhattacharyya distance
mean_diff_x123 = mean_w1_x123 - mean_w2_x123
cov_mean_x123 = (cov_w1_x123 + cov_w2_x123) / 2
cov_mean_inv_x123 = np.linalg.inv(cov_mean_x123)
DB_x123 = 0.125 * mean_diff_x123.T @ cov_mean_inv_x123 @ mean_diff_x123 + \
          0.5 * np.log(np.linalg.det(cov_mean_x123) / np.sqrt(np.linalg.det(cov_w1_x123) * np.linalg.det(cov_w2_x123)))

#Bhattacharyya bound on error
DBerr_X123 = 0.5 * np.exp(-DB_x123)
print(f"Bhattacharyya Bound on Error: {DBerr_X123:.3f}")