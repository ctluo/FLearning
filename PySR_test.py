import numpy as np
import pandas as pd
from pysr import PySRRegressor
import matplotlib.pyplot as plt
import time
import os

# X = 2 * np.random.randn(100, 5)
# y = 2.5382 * np.cos(X[:, 3]) + X[:, 1]*X[:, 0] ** 2 - 0.5


# basename =  './HQL20250312_basic_Cx'
# basename =  './datasets/Steady_CN_0-90'
# basename = './datasets/Steady_CN_90-180'
# Steady_mZ0_0-90
# basename = './datasets/Steady_mZ0_0-90'
# Steady_mZ0_90-180
# basename = './datasets/Steady_mZ0_90-180'
basename = './datasets/AeroHeating_Fused'


fullname = basename + '.xlsx' 
# fullname = './quanjianshujubiao.xlsx' 
df = pd.read_excel(fullname)
# 
print('Data shape: ', df.shape)
print(df) # only for debug
colnames=df.columns.to_list()
dim=len(colnames)-1
X=df.iloc[:, :dim].values
y=df.iloc[:, -1].values
print('X shape: ', X.shape)

model = PySRRegressor(
    procs=8,
    populations=16,
    # ^ 2 populations per core, so one is always running.
    population_size=32,
    # ^ Slightly larger populations, for greater diversity.
    ncycles_per_iteration=50,
    # ^ Generations between migrations.
    niterations=200,  # Run forever if niterations is too large
    early_stop_condition=(
        "stop_if(loss, complexity) = loss < 1e-6 && complexity < 30"
        # Stop early if we find a good and simple equation
    ),
    timeout_in_seconds=60 * 60 * 0.1, 
    # ^ Alternatively, stop after 0.1 hours have passed.
    maxsize=50,
    # ^ Allow greater complexity.
    maxdepth=20,
    # ^ But, avoid deep nesting.
    binary_operators=["*", "+", "-", "/","^"],
    unary_operators=["square", "cube", "exp","sin","cos"], # , "cos2(x)=cos(x)^2"],
    
    # # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # constraints={
    #     "/": (-1, 9),
    #     "square": 9,
    #     "cube": 9,
    #     "exp": 9,
    #     "^": (-1, 9),
    # },
    # # ^ Limit the complexity within each argument.
    # # "inv": (-1, 9) states that the numerator has no constraint,
    # # but the denominator has a max complexity of 9.
    # # "exp": 9 simply states that `exp` can only have
    # # an expression of complexity 9 as input.
    # nested_constraints={
    #     "square": {"square": 1, "cube": 1, "exp": 0},
    #     "cube": {"square": 1, "cube": 1, "exp": 0},
    #     "exp": {"square": 1, "cube": 1, "exp": 0},
    #     "^": {"square": 1, "cube": 1, "exp": 0},
    # },
    # # ^ Nesting constraints on operators. For example,
    # # "square(exp(x))" is not allowed, since "square": {"exp": 0}.
    # complexity_of_operators={"/": 2, "exp": 3,"^": 4},
    # # ^ Custom complexity of particular operators.
    # complexity_of_constants=2,
    # # ^ Punish constants more than variables
    # select_k_features=5,
    # # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    # # ^ Train on only the 4 most important features
    progress=False, # True,
    # ^ Can set to false if printing to a file.
    weight_randomize=0.1,
    # ^ Randomize the tree much more frequently
    cluster_manager=None,
    # ^ Can be set to, e.g., "slurm", to run a slurm
    # cluster. Just launch one script from the head node.
    precision=64,
    # ^ Higher precision calculations.
    warm_start= False,  # True,
    # ^ Start from where left off.
    turbo=True,
    # ^ Faster evaluation (experimental)
    # extra_sympy_mappings={"cos2": lambda x: sympy.cos(x)**2},
    # extra_torch_mappings={sympy.cos: torch.cos},
    # ^ Not needed as cos already defined, but this
    # is how you define custom torch operators.
    # extra_jax_mappings={sympy.cos: "jnp.cos"},
    # ^ For JAX, one passes a string.
)

# 运行10轮
# rounds = 10
rounds = 10
for i in range(rounds):
    print('\n\n第{}轮学习开始...'.format(i+1))
    try:
        model.fit(X, y)
        expr=model.sympy().simplify()
        print('\n\n模型表达式如下：')
        print(expr)
        y_pred = model.predict(X)
        y_true = y

        # 计算模型的性能评估指标,包括均方根误差(RMSE)、相对误差(Relative Error)和决定系数(R2)
        # 同时输出这些指标到控制台和文本文件
        with open('./Report.txt', 'w') as f:
            # 将模型表达式写入文件
            
            print('\n\n模型表达式如下：',file=f)
            print(expr, file=f)
            print('\n\n模型的性能评估指标如下：',file=f)

            print('均方根误差(RMSE): {}'.format(np.sqrt((((y_true-y_pred)) ** 2).mean())), file=f)
            rError=np.sqrt((((y_true-y_pred)) ** 2).mean())/abs(y_true).max()
            percentage = "{:.2%}".format(rError)
            print('相对误差(Relative Error): {}'.format(percentage), file=f)
            Rsquared = 1-np.sum((y_true-y_pred) ** 2)/np.sum((y_true-y_true.mean()) ** 2)
            print('决定系数(R2): {}'.format(Rsquared), file=f)

        print('\n\n模型的性能评估指标如下：')
        RMSE=np.sqrt((((y_true-y_pred)) ** 2).mean())
        print('均方根误差(RMSE)：')
        print(RMSE)

        rError=RMSE/abs(y_true).max()
        print('相对误差(Relative Error)：')
        # 将相对误差格式化为百分比形式，保留两位小数
        percentage = "{:.2%}".format(rError)
        print(percentage)

        Rsquared = 1-np.sum((y_true-y_pred) ** 2)/np.sum((y_true-y_true.mean()) ** 2)
        print('决定系数(R2)：')
        print(Rsquared)

        # 画图对比预测值和真实值        
        fig, ax = plt.subplots()
        ax.scatter(y_true, y_pred)
        ax.set_title("Predicted vs Target")
        ax.set_xlabel("Target")
        ax.set_ylabel("Predicted")
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, 'k--', zorder=0)
        # ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        # plt.show()
        # 保存图像为png格式
        fig.savefig('./Predict_vs_Target.png', dpi=600)

        # 按照样本顺序画图对比预测值和真实值
        # import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        # 添加Marker为"*"
        # ax.scatter(range(len(y_true)), y_true, marker='*', label='Target')
        ax.stem(range(len(y_true)), y_true, linefmt="b-", markerfmt="b*", basefmt="k-", label="Target")
        # ax.scatter(range(len(y_true)), y_true, label='Target')
        # 添加Marker为"o"
        ax.scatter(range(len(y_pred)), y_pred, marker='o', facecolors='none', edgecolors='red', label='Predicted')
        # ax.scatter(range(len(y_pred)), y_pred, label='Predicted')
        ax.legend()
        ax.set_title("Predicted and Target")
        ax.set_xlabel("Data index")
        ax.set_ylabel("Value")
        # plt.show()
        fig.savefig('./Predict_and_Target.png', dpi=600)

        # Rsquared转换为字符串
        strRsquared = "{:.4f}".format(Rsquared)
        # 获取当前时间并转换为字符串：time_string
        
        time_string = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        
        # 参考以下julia代码，调用cmd命令：7z.exe, 将数据文件和报告文件压缩到一个7z文件中
        #  run(`7z a ./Backups/$baseName.$strRsquared.$time_string.7z $datafileName Report.txt Predict_and_Target.png Predict_vs_Target.png`)
        os.system(f'7z a ./Backups/{basename}_{strRsquared}_{time_string}.7z {basename}.xlsx Report.txt Predict_and_Target.png Predict_vs_Target.png')
    except Exception as e:
        print(f"Iteration {i}: Failed with error: {str(e)}")

print('\n\n程序执行完成！')
