"""
Neighborhood Components Analysis (NCA)
"""

from __future__ import absolute_import
import warnings
import time
import sys
import numpy as np
from scipy.optimize import minimize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.fixes import logsumexp


EPS = np.finfo(float).eps


class NCA():
  """Neighborhood Components Analysis (NCA)

  NCA is a distance metric learning algorithm which aims to improve the
  accuracy of nearest neighbors classification compared to the standard
  Euclidean distance. The algorithm directly maximizes a stochastic variant
  of the leave-one-out k-nearest neighbors(KNN) score on the training set.
  It can also learn a low-dimensional linear transformation of data that can
  be used for data visualization and fast classification.

  Read more in the :ref:`User Guide <nca>`.

  Parameters
  ----------
  n_components : int or None, optional (default=None)
      Dimensionality of reduced space (if None, defaults to dimension of X).

  max_iter : int, optional (default=100)
    Maximum number of iterations done by the optimization algorithm.

  tol : float, optional (default=None)
      Convergence tolerance for the optimization.

  verbose : bool, optional (default=False)
    Whether to print progress messages or not.

  random_state : int or numpy.RandomState or None, optional (default=None)
      A pseudo random number generator object or a seed for it if int. If
      ``init='random'``, ``random_state`` is used to initialize the random
      transformation. If ``init='pca'``, ``random_state`` is passed as an
      argument to PCA when initializing the transformation.

  Attributes
  ----------
  n_iter_ : `int`
      The number of iterations the solver has run.

  components_ : `numpy.ndarray`, shape=(n_components, n_features)
      The learned linear transformation ``L``.

  """

  def __init__(self, n_components=None,
               max_iter=100, tol=None, verbose=False, preprocessor=None,
               random_state=None):
    self.n_components = n_components
    self.max_iter = max_iter
    self.tol = tol
    self.verbose = verbose
    self.random_state = random_state
    self.preprocessor = preprocessor
    super(NCA, self)

  def fit(self, X, y):
        """
        X: data matrix, (n x d)
        y: scalar labels, (n)
        """
        n_components = self.n_components
        random_state = self.random_state
        # 测量总训练时间
        train_time = time.time()
        # 初始化 A
        # A = np.array([[1., 0., 1., 0.], [0., 1., 0., 1.]])
        # A = np.array([[1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
        #               [0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.],
        #               [1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.]])
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        lda.fit(X, y)
        A = lda.scalings_.T[:n_components]
        # pca = PCA(n_components=n_components,
        #           random_state=random_state)
        # pca.fit(X)
        # A = pca.components_
        # A = np.eye(n_components, X.shape[-1])
        print(A)

        # 运行 NCA
        mask = y[:, np.newaxis] == y[np.newaxis, :]
        print(mask)
        optimizer_params = {'method': 'L-BFGS-B',
                            'fun': self._loss_grad_lbfgs,
                            'args': (X, mask, -1.0),
                            'jac': True,
                            'x0': A.ravel(),
                            'options': dict(maxiter=self.max_iter),
                            'tol': self.tol
                            }
        # 调用优化器
        self.n_iter_ = 0
        opt_result = minimize(**optimizer_params)

        self.components_ = opt_result.x.reshape(-1, X.shape[1])
        print('self.components_',self.components_)
        self.n_iter_ = opt_result.nit

        train_time = time.time() - train_time
        print('Training took {:8.2f}s.'.format(train_time))
        return self

  def transform(self, X):
    return X.dot(self.components_.T)

  def fit_transform(self, X, y):
      return self.fit(X, y).transform(X)

  def _loss_grad_lbfgs(self, A, X, mask, sign=1.0):

    if self.n_iter_ == 0 and self.verbose:
      header_fields = ['Iteration', 'Objective Value', 'Time(s)']
      header_fmt = '{:>10} {:>20} {:>10}'
      header = header_fmt.format(*header_fields)
      cls_name = self.__class__.__name__
      print('[{cls}]'.format(cls=cls_name))
      print('[{cls}] {header}\n[{cls}] {sep}'.format(cls=cls_name,
                                                     header=header,
                                                     sep='-' * len(header)))

    start_time = time.time()

    A = A.reshape(-1, X.shape[1])
    X_embedded = np.dot(X, A.T)  # (n_samples, n_components)
    # Compute softmax distances
    p_ij = pairwise_distances(X_embedded, squared=True)
    np.fill_diagonal(p_ij, np.inf)
    p_ij = np.exp(-p_ij - logsumexp(-p_ij, axis=1)[:, np.newaxis])
    print('p_ij', p_ij)
    # (n_samples, n_samples)

    # Compute loss
    masked_p_ij = p_ij * mask
    p = masked_p_ij.sum(axis=1, keepdims=True)  # (n_samples, 1)
    loss = p.sum()

    # Compute gradient of loss w.r.t. `transform`
    weighted_p_ij = masked_p_ij - p_ij * p
    weighted_p_ij_sym = weighted_p_ij + weighted_p_ij.T
    np.fill_diagonal(weighted_p_ij_sym, - weighted_p_ij.sum(axis=0))
    gradient = 2 * (X_embedded.T.dot(weighted_p_ij_sym)).dot(X)

    if self.verbose:
        start_time = time.time() - start_time
        values_fmt = '[{cls}] {n_iter:>10} {loss:>20.6e} {start_time:>10.2f}'
        print(values_fmt.format(cls=self.__class__.__name__,
                                n_iter=self.n_iter_, loss=loss,
                                start_time=start_time))
        sys.stdout.flush()

    self.n_iter_ += 1
    return sign * loss, sign * gradient.ravel()
