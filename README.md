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

- AssertionError: Expected shape(M,N), but got shape(C,D)."
  + 返回的计算结果的形状不正确
- AssertionError: Matrix A shouldn't be modified.
  + 你在实现augmentMatrix时修改了矩阵A
- AssertionError: Matrix A is singular.
  + 你的gj_Solve实现在矩阵A是奇异矩阵时没有返回None
- AssertionError: Matrix A is not singular.
  + 你的gj_Solve实现会在矩阵A不是奇异矩阵时返回None
- AssertionError: Bad result.
  + 你的gj_Solve返回了不正确的计算结果

# 项目提交
请在提交前确认你的项目已经满足所有[评审标准](https://review.udacity.com/#!/rubrics/871/view), 项目评审人会根据这份标准来给你相应的审阅。

你需要提交以下4个文件, 请注意每个文件的文件名和文件格式。

1. `linear_regression_project.ipynb`: 写有你代码及答案的  ipynb 文件

3. `linear_regression_project.html`: 由 Jupyter notebook 导出的 html 文件

3. `linear_regression_project.py`: 由 Jupyter notebook 导出的 python 文件

2. `proof.pdf`: 写有你的证明的 pdf 文件 （如果你在 ipython notebook中使用 LATEX 完成证明，则不需要提交此文件。）

5. 请不要提交其他任何文件。

你可以使用 Github 或上传 zip 压缩文档的方式提交项目。如果使用 Github 请将所有提交的文件放在你提交的repo中。 如果上传压缩文档，请将所有提交的文件放在压缩文档中，并命名为 `submit.zip` 后提交。

--- 

# Linear Regression Project(in developing, not validable yet)

Welcome to linear regression project.  Linear regression is the basic of many machine learning algorithms. In this project, you will imply what you learn to solve a linear regression problem, without using any external libraries. 

# What to do
All tasks are listed in  `linear_regression_project.ipynb`，including coding and proving tasks.

**You're encouragd to submit problem even if you haven't finished all tasks. You should submit with spefici questions and explain what you have tried and why it doesn't work. Reviewers will guide you accordingly.**


# Unit test
You can (and should) use unit tests to ensure all your implementations meet requirements. You can find the following code after every coding task. 

`%run -i -e test.py LinearRegressionTestCase.test_...`

If there is an error in your implementation, the `AssertionError` will be thrown. Please modify your code accordingly, until you've passed all unit tests.

The following are some examples of the assersion error. 

- AssertionError: Matrix A shouldn't be modified
  + Your augmentMatrix modifies matrix A. 
  
- AssertionError: Matrix A is singular
  + Your gj_Solve doesn't return None when A is singular. 
- AssertionError: Matrix A is not singular
  + Your gj_Solve returns None when A is not singular.
- AssertionError: x have to be a two-dimensional Python list
  + Your gj_Solve returns with incorrect data structure. X should be a two a list of lists. 

- AssertionError: Regression result isn't good enough
  + Your gj_Solve has too much error. 

# Project submission
Before submission, please ensure that your project meets all the requirements in this [rubric](https://review.udacity.com/#!/rubrics/854/view). Reviewer will give reviews according to this rubric.

You should submit the follow four files. Please pay attention to the file names and file types. 

1. `linear_regression_project.ipynb`: the ipynb file with your code and answers. 

3. `linear_regression_project.html`: the html file exported by Jupyter notebook.

3. `linear_regression_project.py`: the python file exported by Jupyter notebook.

2. `proof.pdf`: the pdf file with your proof. （If you use LATEX in ipython notebook to write the proof, you don't need to submit this file. ）

5. Please DO NOT submit any other files.

You can use Github or upload zip file to submit the project. If you use Github, please include all your files in the repo. If you submit with zip file, please compress all your files in `submit.zip` and then upload. 
