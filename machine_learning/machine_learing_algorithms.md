# Linear Models
## Simple linear regression

LinearRegression fits a linear model with coefficients w = (w_1, ..., w_p) to minimize the residual sum of squares between the observed responses in the dataset, and the responses predicted by the linear approximation.

### On SkLearn
    from sklearn.linear_model import LinearRegression

    regression = LinearRegression()
    regression.fit(x, y)

    print('Slope : %f' %{regression.coef_}}
    print('Intercept: %f' {regression.intercept_})

    reg.score(x, y) #r square

    prediction = regression.predict(v)

### On PySpark

    from pyspark.ml.regression import LinearRegression
    from pyspark.ml.regression import LinearRegression

    data = [
            [Vectors.dense([2, 4, 1]), 6],  
            [Vectors.dense([3, 2, 4]), 9],
            [Vectors.dense([2, 3, 4]), 3],
            [Vectors.dense([6, 1, 4]), 5]
           ]

    df = spark.createDataFrame(data,['features', 'label'])

    lr = LinearRegression()
    linear_model = lr.fit(df)

    print("Coefficients: %s" % str(linear_model.coefficients))
    print("Intercept: %s" % str(linear_model.intercept))


    data_test = [
                 [Vectors.dense([2, 4, 1])],  
                 [Vectors.dense([3, 2, 4])],
                 [Vectors.dense([2, 3, 4])],
                 [Vectors.dense([6, 1, 4])]
               ]

    df_test = spark.createDataFrame(data_test, ['features'])
    df_result = linear_model.transform(df_test)
    df_result.show()

However, coefficient estimates for Ordinary Least Squares rely on the independence
of the model terms. When terms are correlated and the columns of the design matrix
X have an approximate linear dependence, the design matrix becomes close to singular
and as a result, the least-squares estimate becomes highly sensitive to random errors
in the observed response, producing a large variance. This situation of multicollinearity
can arise, for example, when data are collected without an experimental design.

**Concepts:**
* Multi-variate Regression


#### 4.2.2  Ridge Regression

    >>> from sklearn import linear_model
    >>> reg = linear_model.Ridge (alpha = .5)
    >>> reg.fit ([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
    Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,
          normalize=False, random_state=None, solver='auto', tol=0.001)
    >>> reg.coef_
    array([ 0.34545455,  0.34545455])
    >>> reg.intercept_
    0.13636...

#### 4.2.3 Lasso Regression

The Lasso is a linear model that estimates sparse coefficients. It is useful in some contexts due to its tendency to prefer solutions with fewer parameter values, effectively reducing the number of variables upon which the given solution is dependent. For this reason, the Lasso and its variants are fundamental to the field of compressed sensing. Under certain conditions, it can recover the exact set of non-zero weights (see Compressive sensing: tomography reconstruction with L1 prior (Lasso)).

    >>> from sklearn import linear_model
    >>> reg = linear_model.Lasso(alpha = 0.1)
    >>> reg.fit([[0, 0], [1, 1]], [0, 1])
    Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
       normalize=False, positive=False, precompute=False, random_state=None,
       selection='cyclic', tol=0.0001, warm_start=False)
    >>> reg.predict([[1, 1]])
    array([ 0.8])


### 4.1 Naive Bayes

#### 4.1.1 Gaussian Naive Bayes

> class sklearn.naive_bayes.GaussianNB(priors=None)


**GaussianNB** implements the Gaussian Naive Bayes algorithm for classification. The likelihood of the features is assumed to be Gaussian.LinearRegression fits a linear model with coefficients w = (w_1, ..., w_p) to minimize the residual sum of squares between the observed responses in the dataset, and the responses predicted by the linear approximation.

    from sklearn.linear_model import LinearRegression()

    regression = LinearRegression()
    regression.fit(x, y)

    regression.coef_
    regression.intercept_

    reg.score(x, y) #r square

However, coefficient estimates for Ordinary Least Squares rely on the independence
of the model terms. When terms are correlated and the columns of the design matrix
X have an approximate linear dependence, the design matrix becomes close to singular
and as a result, the least-squares estimate becomes highly sensitive to random errors
in the observed response, producing a large variance. This situation of multicollinearity
can arise, for example, when data are collected without an experimental design.

**Concepts:**
* Multi-variate (Regressão Multivariada)

#### 4.2.2  Ridge Regression

    >>> from sklearn import linear_model
    >>> reg = linear_model.Ridge (alpha = .5)
    >>> reg.fit ([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
    Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,
          normalize=False, random_state=None, solver='auto', tol=0.001)
    >>> reg.coef_
    array([ 0.34545455,  0.34545455])
    >>> reg.intercept_
    0.13636...

#### 4.2.3 Lasso Regression

The Lasso is a linear model that estimates sparse coefficients. It is useful in some contexts due to its tendency to prefer solutions with fewer parameter values, effectively reducing the number of variables upon which the given solution is dependent. For this reason, the Lasso and its variants are fundamental to the field of compressed sensing. Under certain conditions, it can recover the exact set of non-zero weights (see Compressive sensing: tomography reconstruction with L1 prior (Lasso)).

    >>> from sklearn import linear_model
    >>> reg = linear_model.Lasso(alpha = 0.1)
    >>> reg.fit([[0, 0], [1, 1]], [0, 1])
    Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
       normalize=False, positive=False, precompute=False, random_state=None,
       selection='cyclic', tol=0.0001, warm_start=False)
    >>> reg.predict([[1, 1]])
    array([ 0.8])


** Parameters:**
* **priors** : array-like, shape (n_classes,)
Prior probabilities of the classes. If specified the priors are not adjusted according to the data.

**Attributes:**

* **class_prior_** : array, shape (n_classes,) -
  probability of each class.
* **class_count_ :** array, shape (n_classes,) -
number of training samples observed in each class.
* **theta_ :** array, shape (n_classes, n_features) -
mean of each feature per class
* **sigma_ :** array, shape (n_classes, n_features) -
variance of each feature per class


#### 4.1.2 Multinomial Naive Bayes

MultinomialNB implements the naive Bayes algorithm for multinomially distributed data, and is one of the two classic naive Bayes variants used in text classification (where the data are typically represented as word vector counts, although tf-idf vectors are also known to work well in practice). The distribution is parametrized by vectors \theta_y = (\theta_{y1},\ldots,\theta_{yn}) for each class y, where n is the number of features (in text classification, the size of the vocabulary) and \theta_{yi} is the probability P(x_i \mid y) of feature i appearing in a sample belonging to class y.

#### 4.1.3 Bernoulli Naive Bayes

BernoulliNB implements the naive Bayes training and classification algorithms for data that is distributed according to multivariate Bernoulli distributions; i.e., there may be multiple features but each one is assumed to be a binary-valued (Bernoulli, boolean) variable. Therefore, this class requires samples to be represented as binary-valued feature vectors; if handed any other kind of data, a BernoulliNB instance may binarize its input (depending on the binarize parameter).

### 4.2 sklearn.linear_model

#### 4.2.1 LinearRegression



### 4.3 sklearn.tree

#### 4.3.1 DecisionTreeClassifier

> class sklearn.tree.DecisionTreeClassifier(criterion=’gini’, splitter=’best’,
    max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
    max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
    min_impurity_split=None, class_weight=None, presort=False)

    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier()
    clf.fit(features_train, labels_train)

-
**criterion:** string, optional (default=”gini”):

The function to measure the quality of a split. Supported criteria are “gini” for
the Gini impurity and “entropy” for the information gain.

**max_depth:** int or None, optional (default=None):

The maximum depth of the tree. If None, then nodes are expanded until all leaves
are pure or until all leaves contain less than min_samples_split samples.

**min_samples_split:** int, float, optional (default=2)

The minimum number of samples required to split an internal node:

  If int, then consider min_samples_split as the minimum number.
  If float, then min_samples_split is a percentage and
  ceil(min_samples_split * n_samples) are the minimum number of samples for
  each split.

**min_samples_leaf:** int, float, optional (default=1)

The minimum number of samples required to be at a leaf node: If int, then consider
min_samples_leaf as the minimum number. If float, then min_samples_leaf is a
percentage and ceil(min_samples_leaf * n_samples) are the minimum number of
samples for each node.


### 4.4 slknearn.neighbors

Neighbors-based classification is a type of instance-based learning or non-generalizing
learning: it does not attempt to construct a general internal model, but simply
stores instances of the training data. Classification is computed from a simple
majority vote of the nearest neighbors of each point: a query point is assigned
the data class which has the most representatives within the nearest neighbors
of the point.

scikit-learn implements two different nearest neighbors classifiers:
KNeighborsClassifier implements learning based on the k nearest neighbors of each
query point, where k is an integer value specified by the user.
RadiusNeighborsClassifier implements learning based on the number of neighbors
within a fixed radius r of each training point, where r is a floating-point value
specified by the user.

The k-neighbors classification in KNeighborsClassifier is the more commonly used
of the two techniques. The optimal choice of the value k is highly data-dependent:
in general a larger k suppresses the effects of noise, but makes the classification
boundaries less distinct.

In cases where the data is not uniformly sampled, radius-based neighbors
classification in RadiusNeighborsClassifier can be a better choice. The user
specifies a fixed radius r, such that points in sparser neighborhoods use fewer
nearest neighbors for the classification. For high-dimensional parameter spaces,
this method becomes less effective due to the so-called “curse of dimensionality”.

The basic nearest neighbors classification uses uniform weights: that is, the value
assigned to a query point is computed from a simple majority vote of the nearest
neighbors. Under some circumstances, it is better to weight the neighbors such that
nearer neighbors contribute more to the fit. This can be accomplished through the
weights keyword. The default value, weights = 'uniform', assigns uniform weights
to each neighbor. weights = 'distance' assigns weights proportional to the inverse
of the distance from the query point. Alternatively, a user-defined function of
the distance can be supplied which is used to compute the weights.


#### 4.4.1 KNeighborsClassifier

    class sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, weights=’uniform’,
    algorithm=’auto’, leaf_size=30, p=2, metric=’minkowski’, metric_params=None,
    n_jobs=1, \**kwargs)


### 4.5 Support Vector Machines
Support vector machines (SVMs) are a set of supervised learning methods used for
classification, regression and outliers detection.

The advantages of support vector machines are:
* Effective in high dimensional spaces.
* Still effective in cases where number of dimensions is greater than the number of samples.
* Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
* Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

* If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.
* SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).


The support vector machines in scikit-learn support both dense (numpy.ndarray and
  convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample
  vectors as input. However, to use an SVM to make predictions for sparse data, i
  t must have been fit on such data. For optimal performance, use C-ordered
  numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.


### 4.5.1 SVC

> class sklearn.svm.SVC(C=1.0, kernel=’rbf’, degree=3, gamma=’auto’, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=’ovr’, random_state=None)[source]

C-Support Vector Classification.
The implementation is based on libsvm. The fit time complexity is more than
quadratic with the number of samples which makes it hard to scale to dataset
with more than a couple of 10000 samples.
The multiclass support is handled according to a one-vs-one scheme.
For details on the precise mathematical formulation of the provided kernel
functions and how gamma, coef0 and degree affect each other, see the corresponding
section in the narrative documentation: Kernel functions.


**Principal Parameters:**

* **C:** float, optional (default=1.0) - Penalty parameter C of the error term.

* **kernel:** string, optional (default=’rbf’)
Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples).

* **degree:** int, optional (default=3) - Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.

* **probability:**  boolean, optional (default=False) - Whether to enable probability estimates. This must be enabled prior to calling fit, and will slow down that method.

## 4.6 sklearn.cluster

### 4.6.1 Kmeans

> class sklearn.cluster.KMeans(n_clusters=8, init=’k-means++’, n_init=10, max_iter=300, tol=0.0001, precompute_distances=’auto’, verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm=’auto’)

The KMeans algorithm clusters data by trying to separate samples in n groups of equal variance, minimizing a criterion known as the inertia or within-cluster sum-of-squares. This algorithm requires the number of clusters to be specified. It scales well to large number of samples and has been used across a large range of application areas in many different fields.
The k-means algorithm divides a set of N samples X into K disjoint clusters C, each described by the mean \mu_j of the samples in the cluster. The means are commonly called the cluster “centroids”; note that they are not, in general, points from  X, although they live in the same space. The K-means algorithm aims to choose centroids that minimise the inertia, or within-cluster sum of squared criterion:

\sum_{i=0}^{n}\min_{\mu_j \in C}(||x_j - \mu_i||^2)

* Inertia, or the within-cluster sum of squares criterion, can be recognized as a measure of how internally coherent clusters are. It suffers from various drawbacks:
Inertia makes the assumption that clusters are convex and isotropic, which is not always the case. It responds poorly to elongated clusters, or manifolds with irregular shapes.

* Inertia is not a normalized metric: we just know that lower values are better and zero is optimal. But in very high-dimensional spaces, Euclidean distances tend to become inflated (this is an instance of the so-called “curse of dimensionality”). Running a dimensionality reduction algorithm such as PCA prior to k-means clustering can alleviate this problem and speed up the computations.
