from gplearn.genetic import SymbolicRegressor
from sklearn.dummy import check_random_state
from sklearn.metrics import r2_score
import numpy as np
import sympy
import pandas as pd

rng = check_random_state(0)

# # Training samples
# X_train = rng.uniform(-1, 1, 100).reshape(50, 2)
# y_train = X_train[:, 0]**2 - X_train[:, 1]**2 + X_train[:, 1] - 1
# # Testing samples
# X_test = rng.uniform(-1, 1, 100).reshape(50, 2)
# y_test = X_test[:, 0]**2 - X_test[:, 1]**2 + X_test[:, 1] - 1

# 从Excel文件中读取数据
# sinA	sinD	SL	Re	dY	CN
# fullname = './datasets/Steady_CN_0-90.xlsx'
# fullname = './datasets/Steady_CN_90-180.xlsx'
# Steady_mZ0_0-90
# fullname = './datasets/Steady_mZ0_0-90.xlsx'
# Steady_mZ0_90-180
# fullname = './datasets/Steady_mZ0_90-180.xlsx'
fullname = './datasets/sFCenter1.xlsx'

df = pd.read_excel(fullname)
XY=df.values
m, n = XY.shape
dim=n-1
X_train, y_train = XY[:,0:dim], XY[:,dim]
X_test = X_train
y_test = y_train
print("X_train.shape=", X_train.shape)
print("y_train.shape=", y_train.shape)
print("X_test.shape=", X_test.shape)
print("y_test.shape=", y_test.shape)

est_gp = SymbolicRegressor(population_size=1000,
                           generations=200, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0)

gplearn_expression=est_gp.fit(X_train, y_train)

print(gplearn_expression)

# ========== 模型性能评估 ==========
def relative_error(y_true, y_pred, eps=1e-10):
    """计算平均相对误差 (Mean Relative Error)，避免除零。"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.abs(y_true) + eps
    return np.mean(np.abs(y_true - y_pred) / denom) * 100

def rrmse(y_true, y_pred, eps=1e-10):
    """
    计算 RRMSE (Relative Root Mean Square Error):
    RRMSE = (1 / max_i |y_i|) * sqrt( (1/n) * sum_i (y_i - ŷ_i)^2 )
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    rmse = np.sqrt(np.sum((y_true - y_pred) ** 2) / n)
    max_abs_y = np.max(np.abs(y_true))
    if max_abs_y < eps:
        return np.nan  # 避免除零
    return rmse / max_abs_y

y_train_pred = est_gp.predict(X_train)
y_test_pred = est_gp.predict(X_test)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
re_train = relative_error(y_train, y_train_pred)
re_test = relative_error(y_test, y_test_pred)
rrmse_train = rrmse(y_train, y_train_pred)
rrmse_test = rrmse(y_test, y_test_pred)

print("\n--- 模型性能评估 ---")
print(f"训练集 - 决定系数 R²: {r2_train:.6f},  相对误差(%): {re_train:.4f},  RRMSE: {rrmse_train:.6f}")
print(f"测试集 - 决定系数 R²: {r2_test:.6f},  相对误差(%): {re_test:.4f},  RRMSE: {rrmse_test:.6f}")


# Simplify the expression
converter = {
    'sub': lambda x, y : x - y,
    'div': lambda x, y : x/y,
    'mul': lambda x, y : x*y,
    'add': lambda x, y : x + y,
    'neg': lambda x    : -x,
    'pow': lambda x, y : x**y
}
ret=sympy.sympify(str(gplearn_expression), locals=converter)
# Print the simplified expression
print(ret)

# Output【结果正确】:
# sub(add(-0.999, X1), mul(sub(X1, X0), add(X0, X1)))
# X1 - (-X0 + X1)*(X0 + X1) - 0.999

print("按照习惯表达，将x0,x1,...替换为x1,x2,... \n新的模型表达式:")
import re
expr=str(gplearn_expression)
expr_new = re.sub(r'X(\d+)', lambda m: 'X' + str(int(m.group(1)) + 1), expr)
print(expr_new)
ret=sympy.sympify(expr_new, locals=converter)
# Print the simplified expression
print(ret)