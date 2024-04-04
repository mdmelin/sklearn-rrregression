"""
This is a module to be used as a reference for building other modules
"""
from random import sample
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances


class ReducedRankRegression(Ridge):
    """ A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.

    Examples
    --------
    >>> from skltemplate import TemplateEstimator
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> estimator = TemplateEstimator()
    >>> estimator.fit(X, y)
    TemplateEstimator()
    """
    from sklearn.decomposition import PCA

    #def __init__(self, rank='full', # default full rank
    #             alpha=0, # default no regularization
    #             *, 
    #             fit_intercept=True,
    #             copy_X=True, 
    #             max_iter=None,
    #             tol=0.0001,
    #             solver='auto',
    #             positive=False,
    #             random_state=None):

    def __init__(self, rank='full', ridge_params_dict=None): # default full rank
        if ridge_params_dict is None:
            ridge_params_dict = dict(alpha=0,) # default no regularization - equivlent to OLS
                                     #fit_intercept=fit_intercept)  # expose this parameter

        super().__init__(**ridge_params_dict)
        self.ridge_params_dict = ridge_params_dict
        self.rank = rank #TODO: handle logic for rank between 0 and 1, explain the proper amount of variance

    def fit(self, X, y, sample_weight=None):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        if self.rank == 'full':
            rank = X.shape[1] #TODO: check this
        elif self.rank >= 1: #TODO: handle rank between 0 and 1
            assert isinstance(self.rank, int), "if rank >= 0, it must be an integer"
            rank = self.rank 

        b_ridge = super().fit(X, y, sample_weight=sample_weight).coef_.T # TODO: get self here or no? 
        #b_ridge = self.coef_
        y_hat_ridge = super().predict(X) 
        pca = PCA(n_components=rank)
        pca.fit(y_hat_ridge)
        b_proj = b_ridge @ pca.components_.T # the encoding matrix (projects predictors from full space to space spanned by first n ranks)
        b_rrr = b_proj @ pca.components_ # the reconstituted reduced rank regression matrix (same size as b_ridge)

        self.coef_ = b_rrr.T
        self.encoder_ = b_proj
        self.decoder_ = pca.components_
        #y_hat = X @ b_rrr
        #y_hat = X @ b_proj @ pca.components_
        return self

    #def predict(self, X):
    #    """ A reference implementation of a predicting function.

    #    Parameters
    #    ----------
    #    X : {array-like, sparse matrix}, shape (n_samples, n_features)
    #        The training input samples.

    #    Returns
    #    -------
    #    y : ndarray, shape (n_samples,)
    #        Returns an array of ones.
    #    """
    #    X = check_array(X, accept_sparse=True)
    #    check_is_fitted(self, 'is_fitted_')
    #    return np.ones(X.shape[0], dtype=np.int64)


class TemplateClassifier(ClassifierMixin, BaseEstimator):
    """ An example classifier which implements a 1-NN algorithm.

    For more information regarding how to build your own classifier, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """
    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param

    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]


class TemplateTransformer(TransformerMixin, BaseEstimator):
    """ An example transformer that returns the element-wise square root.

    For more information regarding how to build your own transformer, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
    """
    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param

    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X, accept_sparse=True)

        self.n_features_ = X.shape[1]

        # Return the transformer
        return self

    def transform(self, X):
        """ A reference implementation of a transform function.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        # Check is fit had been called
        check_is_fitted(self, 'n_features_')

        # Input validation
        X = check_array(X, accept_sparse=True)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        return np.sqrt(X)
