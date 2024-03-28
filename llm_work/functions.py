import numpy as np
from sklearn.datasets import make_blobs
from numpy.random import uniform, normal, shuffle
from scipy.stats import multivariate_normal

def _generate_2d_rotation(theta=0):
    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

    return R

def generate_gaussian_parity(
    n_samples,
    centers=None,
    class_label=None,
    cluster_std=0.25,
    bounding_box=(-1.0, 1.0),
    angle_params=None,
    random_state=None,
):
    """
    Generate 2-dimensional Gaussian XOR distribution.
    (Classic XOR problem but each point is the
    center of a Gaussian blob distribution)
    Parameters
    ----------
    n_samples : int
        Total number of points divided among the four
        clusters with equal probability.
    centers : array of shape [n_centers,2], optional (default=None)
        The coordinates of the ceneter of total n_centers blobs.
    class_label : array of shape [n_centers], optional (default=None)
        class label for each blob.
    cluster_std : float, optional (default=1)
        The standard deviation of the blobs.
    bounding_box : tuple of float (min, max), default=(-1.0, 1.0)
        The bounding box within which the samples are drawn.
    angle_params: float, optional (default=None)
        Number of radians to rotate the distribution by.
    random_state : int, RandomState instance, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.
    y : array of shape [n_samples]
        The integer labels for cluster membership of each sample.
    """

    if random_state != None:
        np.random.seed(random_state)

    if centers == None:
        centers = np.array([(-0.5, 0.5), (0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)])

    if class_label == None:
        class_label = [0, 1, 1, 0]

    blob_num = len(class_label)

    # get the number of samples in each blob with equal probability
    samples_per_blob = np.random.multinomial(
        n_samples, 1 / blob_num * np.ones(blob_num)
    )

    X = np.zeros((1,2), dtype=float)
    y = np.zeros((1), dtype=float)
    ii = 0
    for center, sample in zip(centers, samples_per_blob):
        X_, _ = make_blobs(
            n_samples=sample*10,
            n_features=2,
            centers=[center],
            cluster_std=cluster_std
        )
        col1 = (X_[:,0] > bounding_box[0]) & (X_[:,0] < bounding_box[1])
        col2 = (X_[:,1] > bounding_box[0]) & (X_[:,1] < bounding_box[1])
        X_ = X_[col1 & col2]
        X = np.concatenate((X,X_[:sample,:]), axis=0)
        y_ = np.array([class_label[ii]]*sample)
        y = np.concatenate((y, y_), axis=0)
        ii += 1

    X, y = X[1:], y[1:]

    if angle_params != None:
        R = _generate_2d_rotation(angle_params)
        X = X @ R

    return X, y.astype(int)

def sample_from_G(n, p=0.5):
    n0 = int(n*p)
    n1 = int(n*(1-p))

    # sample from class 0
    x0 = multivariate_normal.rvs(
        mean=[0, 0],
        cov=np.eye(2),
        size=n0
    )
    y0 = np.zeros((n0, ))

    # sample from class 1
    x1 = multivariate_normal.rvs(
        mean=[1, 0],
        cov=np.array([[1, 0.5], [0.5, 1]]),
        size=n1
    )
    y1 = np.ones((n1, ))

    x = np.concatenate((x0, x1))
    y = np.concatenate((y0, y1))
    idx = np.random.permutation(n)
    return x[idx], y[idx]

def sample_from_F(n, p=0.5, pi=0.1):
    n0 = int(n*p)
    n1 = int(n*(1-p))

    # sample from class 0
    x0 = multivariate_normal.rvs(
        mean=[0, 0],
        cov=np.eye(2),
        size=n0
    )
    y0 = np.zeros((n0, ))

    # sample from class 1
    m = n1
    m0 = int(m * pi)
    m1 = int(m * (1-pi))
    x01 = multivariate_normal.rvs(
        mean=[0, 0],
        cov=np.eye(2),
        size=m0
    )
    x11 = multivariate_normal.rvs(
        mean=[5, 1],
        cov=np.array([[1, 0.2], [0.2, 1]]),
        size=m1
    )
    y1 = np.ones((n1, ))

    x = np.concatenate((x0, x01, x11))
    y = np.concatenate((y0, y1))
    idx = np.random.permutation(n)
    return x[idx], y[idx]