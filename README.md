# linear_algebra

---

本项目已经部署了自动化测试，你能在每个需要你完成的函数下面看到形如以下测试代码：
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
