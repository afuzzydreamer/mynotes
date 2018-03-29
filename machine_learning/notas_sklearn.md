# SCIKIT-LEARN

1. [Preprocessing](## 1. sklearn.preprocessing)   
  1.1 [preprocessing.scale](### 1.1 preprocessing.scale)   
  1.2 [StandardScaler](### 1.2 StandardScaler)   



## 1. sklearn.preprocessing

### 1.1 scale

> sklearn.preprocessing.scale(X, axis=0, with_mean=True, with_std=True, copy=True)

**Standardization** of datasets is a common requirement for many machine learning estimators implemented in scikit-learn;

they might behave badly if the individual features do not more or less look like standard normally distributed data: Gaussian with zero mean and unit variance.

    >>> from sklearn import preprocessing
    >>> import numpy as np
    >>> X_train = np.array([[ 1., -1.,  2.],
    >>>                     [ 2.,  0.,  0.],
    >>>                     [ 0.,  1., -1.]])
    >>> X_scaled = preprocessing.scale(X_train)
    >>>
    >>>
    >>> X_scaled                                          
    >>> array([[ 0.  ..., -1.22...,  1.33...],
    >>> [ 1.22...,  0.  ..., -0.26...],
    >>> [-1.22...,  1.22..., -1.06...]])


**Notes**

This implementation will refuse to center scipy.sparse matrices since it would make them non-sparse and would potentially crash the program with memory exhaustion problems.
Instead the caller is expected to either set explicitly with_mean=False (in that case, only variance scaling will be performed on the features of the CSC matrix) or to call X.toarray() if he/she expects the materialized dense array to fit in memory.
To avoid memory copy the caller should pass a CSC matrix.
For a comparison of the different scalers, transformers, and normalizers, see

### 1.2 StandardScaler

> class sklearn.preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True

StandardScaler removes the mean and scales the data to unit variance. However, the outliers have an influence when computing the empirical mean and standard deviation which shrink the range of the feature values as shown in the left figure below. Note in particular that because the outliers on each feature have different magnitudes, the spread of the transformed data on each feature is very different: most of the data lie in the [-2, 4] range for the transformed median income feature while the same data is squeezed in the smaller [-0.2, 0.2] range for the transformed number of households.
StandardScaler therefore cannot guarantee balanced feature scales in the presence of outliers.

**Atributes:** ()

* **scale_:** ndarray, shape (n_features,) - Per feature relative scaling of the data.

* **mean_:** array of floats with shape [n_features]

* **var_:** array of floats with shape [n_features]

    The mean value for each feature in the training set.


* **n_samples_seen_:** The number of samples processed by the estimator. Will be reset on new calls to fit, but increments across partial_fit calls.



**Methods:**


* **fit(X[, y]):**	Compute the mean and std to be used for later scaling.
* **fit_transform(X[, y]):**	Fit to data, then transform it.
* **get_params([deep]):**	Get parameters for this estimator.
* **inverse_transform(X[, copy]):**	Scale back the data to the original representation
* **partial_fit(X[, y]):**	Online computation of mean and std on X for later scaling.
* **set_params(**params):**	Set the parameters of this estimator.
* **transform(X[, y, copy]):**	Perform standardization by centering and scaling


**Examples**

    >>> from sklearn.preprocessing import StandardScaler
    >>>
    >>> data = [[0, 0], [0, 0], [1, 1], [1, 1]]
    >>> scaler = StandardScaler()
    >>> print(scaler.fit(data))
    StandardScaler(copy=True, with_mean=True, with_std=True)
    >>> print(scaler.mean_)
    [ 0.5  0.5]
    >>> print(scaler.transform(data))
    [[-1. -1.]
     [-1. -1.]
     [ 1.  1.]
     [ 1.  1.]]
    >>> print(scaler.transform([[2, 2]]))
    [[ 3.  3.]]


### 1.3 MinMaxScaler

MinMaxScaler rescales the data set such that all feature values are in the range [0, 1] as shown in the right panel below. However, this scaling compress all inliers in the narrow range [0, 0.005] for the transformed number of households.
As StandardScaler, MinMaxScaler is very sensitive to the presence of outliers.

### 1.4 MaxAbsScaler

MaxAbsScaler differs from the previous scaler such that the absolute values are mapped in the range [0, 1]. On positive only data, this scaler behaves similarly to MinMaxScaler and therefore also suffers from the presence of large outliers.


### 1.5 RobustScaler

Unlike the previous scalers, the centering and scaling statistics of this scaler are based on percentiles and are therefore not influenced by a few number of very large marginal outliers. Consequently, the resulting range of the transformed feature values is larger than for the previous scalers and, more importantly, are approximately similar: for both features most of the transformed values lie in a [-2, 3] range as seen in the zoomed-in figure. Note that the outliers themselves are still present in the transformed data. If a separate outlier clipping is desirable, a non-linear transformation is required (see below).


### 1.6 QuantileTransformer

QuantileTransformer applies a non-linear transformation such that the probability density function of each feature will be mapped to a uniform distribution. In this case, all the data will be mapped in the range [0, 1], even the outliers which cannot be distinguished anymore from the inliers.

As RobustScaler, QuantileTransformer is robust to outliers in the sense that adding or removing outliers in the training set will yield approximately the same transformation on held out data. But contrary to RobustScaler, QuantileTransformer will also automatically collapse any outlier by setting them to the a priori defined range boundaries (0 and 1).


### 1.7 Normalizer

The Normalizer rescales the vector for each sample to have unit norm, independently of the distribution of the samples. It can be seen on both figures below where all samples are mapped onto the unit circle. In our example the two selected features have only positive values; therefore the transformed data only lie in the positive quadrant. This would not be the case if some original features had a mix of positive and negative values.

http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py


## 2. Metrics

### 2.1 Accuracy

> sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)

The accuracy_score function computes the accuracy, either the fraction (default) or the count (normalize=False) of correct predictions.

In multilabel classification, the function returns the subset accuracy. If the entire set of predicted labels for a sample strictly match with the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0.

    >>> import numpy as np
    >>> from sklearn.metrics import accuracy_score
    >>> y_pred = [0, 2, 1, 3]
    >>> y_true = [0, 1, 2, 3]
    >>> accuracy_score(y_true, y_pred)
    0.5
    >>> accuracy_score(y_true, y_pred, normalize=False)
    2


### 2.2 Mean squared error

The mean_squared_error function computes mean square error, a risk metric corresponding to the expected value of the squared (quadratic) error or loss.

    >>> from sklearn.metrics import mean_squared_error
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> mean_squared_error(y_true, y_pred)
    0.375
    >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    >>> mean_squared_error(y_true, y_pred)  
    0.7083...


### 2.3 Median absolute error

The median_absolute_error is particularly interesting because it is robust to outliers. The loss is calculated by taking the median of all absolute differences between the target and the prediction.

The median_absolute_error does not support multioutput.

Here is a small example of usage of the median_absolute_error function:

    >>> from sklearn.metrics import median_absolute_error
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> median_absolute_error(y_true, y_pred)
    0.5


### 2.4 R² score, the coefficient of determination

The r2_score function computes R², the coefficient of determination. It provides a measure of how well future samples are likely to be predicted by the model. Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0.

    >>> from sklearn.metrics import r2_score
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> y_true = [3, -0.5, 2, 7]
    >>> r2_score(y_true, y_pred)  
    0.948...
    >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    >>> r2_score(y_true, y_pred, multioutput='variance_weighted')
    >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    ...
    0.938...
    >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    >>> r2_score(y_true, y_pred, multioutput='uniform_average')
    ...
    0.936...
    >>> r2_score(y_true, y_pred, multioutput='raw_values')
    ...
    array([ 0.965...,  0.908...])
    >>> r2_score(y_true, y_pred, multioutput=[0.3, 0.7])
    ...
    0.925...

### 2.5 F1 score

Compute the F1 score, also known as balanced F-score or F-measure
The F1 score can be interpreted as a weighted average of the precision and recall,
where an F1 score reaches its best value at 1 and worst score at 0. The relative
contribution of precision and recall to the F1 score are equal.
The formula for the F1 score is:

F1 = 2 * (precision * recall) / (precision + recall)

In the multi-class and multi-label case, this is the weighted average
of the F1 score of each class.

    >>> from sklearn.metrics import f1_score
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_pred = [0, 2, 1, 0, 0, 1]
    >>> f1_score(y_true, y_pred, average='macro')  
    0.26...
    >>> f1_score(y_true, y_pred, average='micro')  
    0.33...
    >>> f1_score(y_true, y_pred, average='weighted')  
    0.26...
    >>> f1_score(y_true, y_pred, average=None)
    array([ 0.8,  0. ,  0. ])


## 3. sklearn.cross_validation

    from sklearn.cross_validation import train_test_split
    ages_train, ages_test, net_worths_train, net_worths_test =
           train_test_split(features, labels)

## 4. Altorithms

### 4.1 Naive Bayes

#### 4.1.1 Gaussian Naive Bayes

> class sklearn.naive_bayes.GaussianNB(priors=None)


**GaussianNB** implements the Gaussian Naive Bayes algorithm for classification. The likelihood of the features is assumed to be Gaussian.

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

LinearRegression fits a linear model with coefficients w = (w_1, ..., w_p) to minimize the residual sum of squares between the observed responses in the dataset, and the responses predicted by the linear approximation.

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


## 5 features_extraction

The sklearn.feature_extraction module can be used to extract features in a format
supported by machine learning algorithms from datasets consisting of formats such
as text and image.

**Note:** Feature extraction is very different from Feature selection: the former
consists in transforming arbitrary data, such as text or images, into numerical
features usable for machine learning. The latter is a machine learning technique
applied on these features.




### 5.1 CountVectorizer (Bag of Words)
> class sklearn.feature_extraction.text.CountVectorizer(input=’content’,
encoding=’utf-8’, decode_error=’strict’, strip_accents=None, lowercase=True,
preprocessor=None, tokenizer=None, stop_words=None, token_pattern=’(?u)\b\w\w+\b’,
ngram_range=(1, 1), analyzer=’word’, max_df=1.0, min_df=1, max_features=None,
vocabulary=None, binary=False, dtype=<class ‘numpy.int64’>)

Convert a collection of text documents to a matrix of token counts
This implementation produces a sparse representation of the counts using
scipy.sparse.csr_matrix.

If you do not provide an a-priori dictionary and you do not use an analyzer that
does some kind of feature selection then the number of features will be equal to
the vocabulary size found by analyzing the data.


    from sklearn.features_extraction.text import CountVectorizer
    text_list = [text1, text2, text3]

    vectorizer = CountVectorizer()

    vectorizer.fit(text_list)

    bag_of_words = vectorizer.transform(email_list)

    quantity = vectorizer.vocabulary_.get("word")

**The Bag of Words representation:**

Text Analysis is a major application field for machine learning algorithms.
However the raw data, a sequence of symbols cannot be fed directly to the
algorithms themselves as most of them expect numerical feature vectors with a
fixed size rather than the raw text documents with variable length.

In order to address this, scikit-learn provides utilities for the most common
ways to extract numerical features from text content, namely:

* **tokenizing** strings and giving an integer id for each possible token,
for instance by using white-spaces and punctuation as token separators.
* **counting** the occurrences of tokens in each document.
* **normalizing** and weighting with diminishing importance tokens that occur in the majority of samples / documents.

In this scheme, features and samples are defined as follows:
each **individual token occurrence frequency** (normalized or not) is treated as a **feature**.
the vector of all the token frequencies for a given **document** is considered a multivariate sample.

A corpus of documents can thus be represented by a matrix with one row per
document and one column per token (e.g. word) occurring in the corpus.

We call vectorization the general process of turning a collection of text
documents into numerical feature vectors. This specific strategy
(tokenization, counting and normalization) is called the Bag of Words or
“Bag of n-grams” representation. Documents are described by word occurrences
while completely ignoring the relative position information of the words in the
document.

**Sparsity:**

As most documents will typically use a very small subset of the words used in
the corpus, the resulting matrix will have many feature values that are zeros
(typically more than 99% of them).
For instance a collection of 10,000 short text documents (such as emails) will
use a vocabulary with a size in the order of 100,000 unique words in total while
each document will use 100 to 1000 unique words individually.
In order to be able to store such a matrix in memory but also to speed up algebraic
operations matrix / vector, implementations will typically use a sparse
representation such as the implementations available in the scipy.sparse package.


**Common Vectorizer usage:**

**CountVectorizer** implements both tokenization and occurrence counting in a single class:

    >>>> from sklearn.feature_extraction.text import CountVectorizer

This model has many parameters, however the default values are quite reasonable (please see the reference documentation for the details):


    >>> vectorizer = CountVectorizer()
    >>> vectorizer                     
    CountVectorizer(analyzer=...'word', binary=False, decode_error=...'strict',
            dtype=<... 'numpy.int64'>, encoding=...'utf-8', input=...'content',
            lowercase=True, max_df=1.0, max_features=None, min_df=1,
            ngram_range=(1, 1), preprocessor=None, stop_words=None,
            strip_accents=None, token_pattern=...'(?u)\\b\\w\\w+\\b',
            tokenizer=None, vocabulary=None)


Let’s use it to tokenize and count the word occurrences of a minimalistic corpus
of text documents:

    >>> corpus = [
    ...     'This is the first document.',
    ...     'This is the second second document.',
    ...     'And the third one.',
    ...     'Is this the first document?',
    ... ]
    >>> X = vectorizer.fit_transform(corpus)
    >>> X                              
    <4x9 sparse matrix of type '<... 'numpy.int64'>'
        with 19 stored elements in Compressed Sparse ... format>

The default configuration tokenizes the string by extracting words of at least 2 letters.
The specific function that does this step can be requested explicitly:

    -
    >>> analyze = vectorizer.build_analyzer()
    >>> analyze("This is a text document to analyze.") == (
    ...     ['this', 'is', 'text', 'document', 'to', 'analyze'])
    True

Each term found by the analyzer during the fit is assigned a unique integer index
corresponding to a column in the resulting matrix. This interpretation of the columns
can be retrieved as follows:

    >>> vectorizer.get_feature_names() == (
    ...     ['and', 'document', 'first', 'is', 'one',
    ...      'second', 'the', 'third', 'this'])
    True

    >>> X.toarray()           
    array([[0, 1, 1, 1, 0, 0, 1, 0, 1],
           [0, 1, 0, 1, 0, 2, 1, 0, 1],
           [1, 0, 0, 0, 1, 0, 1, 1, 0],
           [0, 1, 1, 1, 0, 0, 1, 0, 1]]...)

The converse mapping from feature name to column index is stored in the vocabulary_
attribute of the vectorizer:

    >>> vectorizer.vocabulary_.get('document')
    1

Hence words that were not seen in the training corpus will be completely ignored
in future calls to the transform method:

    >>> vectorizer.transform(['Something completely new.']).toarray()
    ...                           
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0]]...)

Note that in the previous corpus, the first and the last documents have exactly
the same words hence are encoded in equal vectors. In particular we lose the information
that the last document is an interrogative form. To preserve some of the local ordering
information we can extract 2-grams of words in addition to the 1-grams (individual words):

    >>> bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),
    ...                                     token_pattern=r'\b\w+\b', min_df=1)
    >>> analyze = bigram_vectorizer.build_analyzer()
    >>> analyze('Bi-grams are cool!') == (
    ...     ['bi', 'grams', 'are', 'cool', 'bi grams', 'grams are', 'are cool'])
    True

The vocabulary extracted by this vectorizer is hence much bigger and can now resolve
ambiguities encoded in local positioning patterns:

    >>> X_2 = bigram_vectorizer.fit_transform(corpus).toarray()
    >>> X_2
    ...                           
    array([[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
           [0, 0, 1, 0, 0, 1, 1, 0, 0, 2, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0],
           [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1]]...)

In particular the interrogative form “Is this” is only present in the last document:

    >>> feature_index = bigram_vectorizer.vocabulary_.get('is this')
    >>> X_2[:, feature_index]     
    array([0, 0, 0, 1]...)

Getting internal representation for a word

    >>> vectorizer.vocabulary_.get("world")

Getting the features names
    >>> vocab_list = vectorizer.get_feature_names()


### 5.2 TFIDF (Term Frequence Inverse Document Frequency)

> class sklearn.feature_extraction.text.TfidfVectorizer(input=’content’,
encoding=’utf-8’, decode_error=’strict’, strip_accents=None, lowercase=True,
preprocessor=None, tokenizer=None, analyzer=’word’, stop_words=None,
token_pattern=’(?u)\b\w\w+\b’, ngram_range=(1, 1), **max_df**=1.0, min_df=1,
max_features=None, vocabulary=None, binary=False, dtype=<class ‘numpy.int64’>,
norm=’l2’, use_idf=True, smooth_idf=True, sublinear_tf=False)


Convert a collection of raw documents to a matrix of TF-IDF features.
Equivalent to CountVectorizer followed by TfidfTransformer.

In a large text corpus, some words will be very present
(e.g. “the”, “a”, “is” in English) hence carrying very little meaningful information
about the actual contents of the document. If we were to feed the direct count
data directly to a classifier those very frequent terms would shadow the frequencies
of rarer yet more interesting terms.
In order to re-weight the count features into floating point values suitable for
usage by a classifier it is very common to use the tf–idf transform.
Tf means term-frequency while tf–idf means term-frequency times inverse
document-frequency: \text{tf-idf(t,d)}=\text{tf(t,d)} \times \text{idf(t)}.
Using the TfidfTransformer’s default settings, TfidfTransformer(norm='l2',
use_idf=True, smooth_idf=True, sublinear_tf=False) the term frequency, the number
of times a term occurs in a given document, is multiplied with idf component.

This normalization is implemented by the TfidfTransformer class:

    >>> from sklearn.feature_extraction.text import TfidfTransformer
    >>> transformer = TfidfTransformer(smooth_idf=False)
    >>> transformer   
    TfidfTransformer(norm=...'l2', smooth_idf=False, sublinear_tf=False,
                     use_idf=True)

Let’s take an example with the following counts. The first term is present 100%
of the time hence not very interesting. The two other features only in less than
50% of the time hence probably more representative of the content of the documents:


    >>> counts = [[3, 0, 1],
    ...           [2, 0, 0],
    ...           [3, 0, 0],
    ...           [4, 0, 0],
    ...           [3, 2, 0],
    ...           [3, 0, 2]]
    ...
    >>> tfidf = transformer.fit_transform(counts)
    >>> tfidf                         
    <6x3 sparse matrix of type '<... 'numpy.float64'>'
        with 9 stored elements in Compressed Sparse ... format>

    >>> tfidf.toarray()                        
    array([[ 0.81940995,  0.        ,  0.57320793],
           [ 1.        ,  0.        ,  0.        ],
           [ 1.        ,  0.        ,  0.        ],
           [ 1.        ,  0.        ,  0.        ],
           [ 0.47330339,  0.88089948,  0.        ],
           [ 0.58149261,  0.        ,  0.81355169]])


## 6.Feature selection
> sklearn.feature_selection

The sklearn.feature_selection module implements feature selection algorithms.
It currently includes univariate filter selection methods and the recursive
feature elimination algorithm.





### 6.1 SelectKBest

> class sklearn.feature_selection.SelectKBest(score_func=<function f_classif>, k=10)

Select features according to the k highest scores.

**Parameters:**

* **score_func:** callable
Function taking two arrays X and y, and returning a pair of arrays (scores, pvalues)
or a single array with scores. Default is f_classif (see below “See also”).
The default function only works with classification tasks.

* **k:** k : int or “all”, optional, default=10

**Attributes:**

* **scores_:** array-like, shape=(n_features,) - Scores of features.
* **pvalues_:** array-like, shape=(n_features,) - p-values of feature scores, None if score_func returned only scores.

* Notes
Ties between features with equal scores will be broken in an unspecified way.

**Methods**

* **fit(X, y):**	Run score function on (X, y) and get the appropriate features.
* **fit_transform(X[, y]):**	Fit to data, then transform it.
* **get_params([deep]):**	Get parameters for this estimator.
* **get_support([indices]):**	Get a mask, or integer index, of the features selected
* **inverse_transform(X):**	Reverse the transformation operation
* **set_params(**params):**	Set the parameters of this estimator.
* **transform(X):**	Reduce X to the selected features.

### 6.2 SelectPercentile
> class sklearn.feature_selection.SelectPercentile(score_func=<function f_classif>, percentile=10

Select features according to a percentile of the highest scores.

## 7. sklearn.decomposition

The sklearn.decomposition module includes matrix decomposition algorithms,
including among others PCA, NMF or ICA. Most of the algorithms of this module
can be regarded as dimensionality reduction techniques.

### 7.1 PCA
> class sklearn.decomposition.PCA(n_components=None, copy=True, whiten=False,
> svd_solver=’auto’, tol=0.0, iterated_power=’auto’, random_state=None)

Principal component analysis (PCA)
Linear dimensionality reduction using Singular Value Decomposition of the data
to project it to a lower dimensional space.
It uses the LAPACK implementation of the full SVD or a randomized truncated SVD
by the method of Halko et al. 2009, depending on the shape of the input data and
the number of components to extract.
It can also use the scipy.sparse.linalg ARPACK implementation of the truncated SVD.
Notice that this class does not support sparse input. See TruncatedSVD for an
alternative with sparse data.


**Exampes:**

    -
    >>> import numpy as np
    >>> from sklearn.decomposition import PCA
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> pca = PCA(n_components=2)
    >>> pca.fit(X)
    PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)
    >>> print(pca.explained_variance_ratio_)  
    [ 0.99244...  0.00755...]
    >>> print(pca.singular_values_)  
    [ 6.30061...  0.54980...]

    >>> pca = PCA(n_components=2, svd_solver='full')
    >>> pca.fit(X)                 
    PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
      svd_solver='full', tol=0.0, whiten=False)
    >>> print(pca.explained_variance_ratio_)  
    [ 0.99244...  0.00755...]
    >>> print(pca.singular_values_)  
    [ 6.30061...  0.54980...]


    >>> pca = PCA(n_components=1, svd_solver='arpack')
    >>> pca.fit(X)
    PCA(copy=True, iterated_power='auto', n_components=1, random_state=None,
      svd_solver='arpack', tol=0.0, whiten=False)
    >>> print(pca.explained_variance_ratio_)  
    [ 0.99244...]
    >>> print(pca.singular_values_)  
    [ 6.30061...]

    print(pca.components_[0])
    print(pca.components_[1])


## 7.sklearn.model_selection

## 7.1 KFold
K-Folds cross-validator
Provides train/test indices to split data in train/test sets.
Split dataset into k consecutive folds (without shuffling by default).
Each fold is then used once as a validation while the k - 1 remaining folds form the training set.

KFold divides all the samples in k groups of samples, called folds
(if k = n, this is equivalent to the Leave One Out strategy), of equal
sizes (if possible). The prediction function is learned using k - 1 folds,
and the fold left out is used for test.

Example of 2-fold cross-validation on a dataset with 4 samples:

        >>> import numpy as np
        >>> from sklearn.model_selection import KFold

        >>> X = ["a", "b", "c", "d"]
        >>> kf = KFold(n_splits=2)
        >>> for train, test in kf.split(X):
        ...     print("%s %s" % (train, test))
        [2 3] [0 1]
        [0 1] [2 3]

Each fold is constituted by two arrays: the first one is related to the training set, and the second one to the test set. Thus, one can create the training/test sets using numpy indexing:

        >>> X = np.array([[0., 0.], [1., 1.], [-1., -1.], [2., 2.]])
        >>> y = np.array([0, 1, 0, 1])
        >>> X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

**Example**

    >>> from sklearn.model_selection import KFold
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([1, 2, 3, 4])
    >>> kf = KFold(n_splits=2)
    >>> kf.get_n_splits(X)
    2
    >>> print(kf)  
    KFold(n_splits=2, random_state=None, shuffle=False)
    >>> for train_index, test_index in kf.split(X):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [2 3] TEST: [0 1]
    TRAIN: [0 1] TEST: [2 3]

    from sklearn.cross_validation import KFold

    kf = KFold(len(authors), 2)

    for train_index, text_index in kf:
        features_train = [word_data[ii] for ii in train_index]
        features_test = [word_data[ii] for ii in test_index]
        authors_train = [authos[ii] for ii in train_index]
        authors_test  = [authors[ii] for ii in test_indices]


# 7.2 GridSearch
> class sklearn.model_selection.GridSearchCV(estimator, param_grid, scoring=None,
fit_params=None, n_jobs=1, iid=True, refit=True, cv=None, verbose=0,
pre_dispatch=‘2*n_jobs’, error_score=’raise’, return_train_score=True)

Exhaustive search over specified parameter values for an estimator.

Important members are fit, predict.

GridSearchCV implements a “fit” and a “score” method. It also implements “predict”, “predict_proba”, “decision_function”, “transform” and “inverse_transform” if they are implemented in the estimator used.

The parameters of the estimator used to apply these methods are optimized by cross-validated grid-search over a parameter grid.


**Notes**

The parameters selected are those that maximize the score of the left out data,
unless an explicit score is passed in which case it is used instead.

If n_jobs was set to a value higher than one, the data is copied for each point
in the grid (and not n_jobs times). This is done for efficiency reasons if
individual jobs take very little time, but may raise errors if the dataset is
large and not enough memory is available. A workaround in this case is to set
pre_dispatch. Then, the memory is copied only pre_dispatch many times.
A reasonable value for pre_dispatch is 2 * n_jobs.


    >>> from sklearn import svm, datasets
    >>> from sklearn.model_selection import GridSearchCV
    >>> iris = datasets.load_iris()
    >>> parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    >>> svc = svm.SVC()
    >>> clf = GridSearchCV(svc, parameters)
    >>> clf.fit(iris.data, iris.target)
    ...                             
    GridSearchCV(cv=None, error_score=...,
           estimator=SVC(C=1.0, cache_size=..., class_weight=..., coef0=...,
                         decision_function_shape='ovr', degree=..., gamma=...,
                         kernel='rbf', max_iter=-1, probability=False,
                         random_state=None, shrinking=True, tol=...,
                         verbose=False),
           fit_params=None, iid=..., n_jobs=1,
           param_grid=..., pre_dispatch=..., refit=..., return_train_score=...,
           scoring=..., verbose=...)
    >>> sorted(clf.cv_results_.keys())
    ...                             
    ['mean_fit_time', 'mean_score_time', 'mean_test_score',...
     'mean_train_score', 'param_C', 'param_kernel', 'params',...
     'rank_test_score', 'split0_test_score',...
     'split0_train_score', 'split1_test_score', 'split1_train_score',...
     'split2_test_score', 'split2_train_score',...
     'std_fit_time', 'std_score_time', 'std_test_score', 'std_train_score'...]

**Methods**
* **decision_function(X)** - Call decision_function on the estimator with the best found parameters.
* **fit(X[, y, groups])** -	Run fit with all sets of parameters.
* **get_params([deep])** - Get parameters for this estimator.
inverse_transform(Xt) 	Call inverse_transform on the estimator with the best found params.
* **predict(X)** - Call predict on the estimator with the best found parameters.
* **predict_log_proba(X)** - Call predict_log_proba on the estimator with the best found parameters.
* **predict_proba(X)** - Call predict_proba on the estimator with the best found parameters.
* **score(X[, y])** -	Returns the score on the given data, if the estimator has been refit.
* **set_params(**params)** - 	Set the parameters of this estimator.
transform(X) 	Call transform on the estimator with the best found parameters.


# NLTK (Natural Language Toolkit)

NLTK is a leading platform for building Python programs to work with human language
data. It provides easy-to-use interfaces to over 50 corpora and lexical resources
such as WordNet, along with a suite of text processing libraries for classification,
tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for
industrial-strength NLP libraries, and an active discussion forum.

Thanks to a hands-on guide introducing programming fundamentals alongside topics
in computational linguistics, plus comprehensive API documentation, NLTK is suitable
for linguists, engineers, students, educators, researchers, and industry users alike.
NLTK is available for Windows, Mac OS X, and Linux. Best of all, NLTK is a free,
open source, community-driven project.

NLTK has been called “a wonderful tool for teaching, and working in, computational
linguistics using Python,” and “an amazing library to play with natural language.”

### 1.Concepts

* **Stemmer: **(lemizador): root of word
* **Corpus**
* **Stopwords**

### 2. Corpus

    from nltk.corpus import stopwords
    sw = stopwords.words("english")


### nltk.stem

    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer("english")
    stemmer.stem("responsivity")

### nltk.word_tokenize
    >>> import nltk
    >>> sentence = """At eight o'clock on Thursday morning
    ... Arthur didn't feel very good."""
    >>> tokens = nltk.word_tokenize(sentence)
    >>> tokens
    ['At', 'eight', "o'clock", 'on', 'Thursday', 'morning',
    'Arthur', 'did', "n't", 'feel', 'very', 'good', '.']
