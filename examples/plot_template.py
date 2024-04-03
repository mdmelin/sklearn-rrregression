"""
===========================
Plotting Template Estimator
===========================

An example plot of :class:`skltemplate.template.TemplateEstimator`
"""
import numpy as np
from matplotlib import pyplot as plt
from skltemplate import ReducedRankRegression

X = np.arange(100).reshape(100, 1)
y = np.zeros((100, ))
estimator = ReducedRankRegression()
estimator.fit(X, y)
plt.plot(y)
plt.plot(estimator.predict(X), alpha=.8, linestyle = '--')
plt.show()
