"""
===========================
Plotting Template Estimator
===========================

An example plot of :class:`skltemplate.template.TemplateEstimator`
"""
import numpy as np
from matplotlib import pyplot as plt
from skltemplate import ReducedRankRegression

X = np.arange(1000).reshape(100, 10)
y = np.zeros((100, 10))

X = np.random.uniform(0,1, size=(100,40))
y = np.random.normal(size=(100,10))

estimator = ReducedRankRegression(rank=2)
estimator.fit(X, y)
plt.plot(y)
plt.plot(estimator.predict(X), alpha=.8, linestyle = '--')
plt.show()
print(estimator.get_params())
