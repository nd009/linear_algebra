# 线性回归项目

欢迎来到线性回归项目，线性代数是很多机器学习算法的基础，在这个项目中，你将不借助任何库，用你之前所学来解决一个线性回归问题。

# 项目内容
所有需要完成的任务都在 `linear_regression_project.ipynb` 中，其中包括编程题和证明题。

**若项目中的题目有困难没完成也没关系，我们鼓励你带着问题提交项目，评审会给予你诸多帮助。**


# 单元测试
项目已经部署了自动化测试，你能在每个需要你完成的函数下面看到形如以下测试代码：
`%run -i -e test.py LinearRegressionTestCase.test_...`
Ctrl + Enter 运行即可。

如果你的实现有问题，会有断言错误`AssertionError`被抛出。
请重新检查你的实现，并且修正bug，直到通过测试为止。

以下是一些带有特定反馈的断言错误说明：

- AssertionError: Matrix A shouldn't be modified
  + 你在实现augmentMatrix时修改了矩阵A
- AssertionError: Matrix A is singular
  + 你的gj_Solve实现在矩阵A是奇异矩阵时没有返回None
- AssertionError: Matrix A is not singular
  + 你的gj_Solve实现会在矩阵A不是奇异矩阵时返回None
- AssertionError: x have to be two-dimensional Python List
  + 你的gj_Solve返回的数据结构不正确，x必须是二维列表，而且是Nx1的列向量
- AssertionError: Regression result isn't good enough
  + 你的gj_Solve返回了计算结果，但是偏差过大

# 项目提交
你需要提交以下4个文件, 请注意每个文件的文件名和文件格式。

1. `linear_regression_project.ipynb`: 写有你代码及答案的  ipynb 文件

2. `proof.pdf`: 写有你的证明的 pdf 文件。

3. `linear_regression_project.html`: 由 Jupyter notebook 导出的 html 文件

3. `linear_regression_project.py`: 由 Jupyter notebook 导出的 python 文件

5. 请不要提交其他任何文件。

你可以使用 Github 或上传 zip 压缩文档的方式提交项目。如果使用 Github 请将所有提交的文件放在你提交的repo中。 如果上传压缩文档，请将所有提交的文件放在压缩文档中，并命名为 `submit.zip` 后提交。