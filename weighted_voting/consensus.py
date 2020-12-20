# article: Learning from multiple annotators: A Survey
# authors: Sharan Vaswani and Mohamed Osama Ahmed
# algorithm implementation: 3.2 Approach proposed by Rayker et.al
# Vikas C Raykar, Shipeng Yu, Linda H Zhao, Gerardo Hermosillo Valadez, Charles
# Florin, Luca
# Bogoni, and Linda Moy.  Learning from crowds.  The Journal of Machine
# Learning Research, 99:1297–
# 1322, 2010.
import warnings
import numpy as np

class WarnOnlyOnce:
    """
    This class contains method which is showing
    warning only one time when calling function.
    """
    warnings = set()

    @classmethod
    def warn(cls, message):
        """
        This method returns warning only one time when calling function.
        """
        h = hash(message)
        if h not in cls.warnings:
            warnings.showwarning(f"Warning: {message}",
                category=UserWarning,
                filename="",
                lineno="")
            cls.warnings.add(h)


def tpr(pred, true):
    """
    true positive rate is measured for each user and each class
    Parameters
    ----------
    pred : numpy.ndarray
    true : numpy.ndarray
    Returns
    -------
    float with tpr for each user and each class
    """
    corr = pred.astype(np.float32) * true
    corr_sums = np.nansum(corr, axis=1)
    true_sums = np.nansum(true, axis=1)
    return np.divide(corr_sums, true_sums, out=np.ones_like(corr_sums), where=true_sums != 0)


def tnr(pred, true):
    """
    true negative rate is measured for each user and each class
    Parameters
    ----------
    pred : numpy.ndarray
    true : numpy.ndarray
    Returns
    -------
    float with tnr for each user and each class
    """
    return tpr(1 - pred, 1 - true)


def log_avoid_zero(x, additional_warning_message="data"):
    """
    This function counts logarithm with replacing value
    of 0 by 10 ** -10.
    Parameters
    ----------
    x : float
    Returns
    -------
    float with log(x) replacing value of log(0) by log(10 ** -10).
    """
    if np.any(x == 0):
        x = x.astype(np.float)
        x[np.where(x == 0)] = 10 ** -10
        WarnOnlyOnce.warn("Replacing value of 0 by 10 ** -10 in the logarithm to enable further computation. \
        This situation occurred while calculating the log of {} and may indicate poor quality of data used." \
            .format(additional_warning_message))
        return np.log(x)
    else:
        return np.log(x)


def user_classes_performance(user_samples_classes, mv_classes, num_classes, num_users, num_samples):
    """
    The performance of each user for each class is measured
    in terms of the sensitivity (tpr), the specificity (tnr),
    fall-out (fpr) and miss rate (fnr).
    Parameters
    ----------
    user_samples_classes : numpy.ndarray
    Array of one hot encoded classes that user assigned to each of the samples. \
    Cells with nan mean that the user didn't evaluate the sample.
    Dimensions: classes, users, samples.
    mv_classes : numpy.ndarray
    Array with classes per sample. Rows represent samples and columns classes.
    num_classes : int
    A number representing the number of classes.
    num_users : int
    A number representing the number of users.
    num_samples : int
    A number representing the number of samples.
    Returns
    -------
    numpy.ndarray with:
        user's tpr, tnr, fpr, fnr for each class.
    """

    user_classes_perf = np.zeros((2, 2, num_users, num_classes))
    mv_classes_new = np.broadcast_to(mv_classes, (num_users, num_samples, num_classes))
    user_classes_nan = user_samples_classes.transpose(1, 2, 0)
    mv_classes_nan = np.empty((num_users, num_samples, num_classes))
    mv_classes_nan[:] = np.NaN
    user_samples = ~np.isnan(user_classes_nan)
    mv_classes_nan[user_samples] = mv_classes_new[user_samples]
    user_classes_perf[1][1] = tpr(user_classes_nan, mv_classes_nan)
    user_classes_perf[0][0] = tnr(user_classes_nan, mv_classes_nan)
    user_classes_perf[0][1] = 1 - user_classes_perf[0][0]
    user_classes_perf[1][0] = 1 - user_classes_perf[1][1]
    return user_classes_perf


def new_classes_likelihood(user_samples_classes, user_classes_perf, prior, num_classes, num_users, num_samples):
    """
    Conditional likelihoods are measured for each sample in each class.
    Parameters
    ----------
    user_samples_classes : numpy.ndarray
    Array of one hot encoded classes that user assigned to each of the samples. \
    Cells with nan mean that the user didn't evaluate the sample.
    Dimensions: classes, users, samples.
    user_classes_perf : numpy.ndarray
    User's tpr, tnr, fpr, fnr for each class.
    prior : numpy.ndarray
    Array with classes priors.
    num_classes : int
    A number representing the number of classes.
    num_users : int
    A number representing the number of users.
    num_samples : int
    A number representing the number of samples.
    Returns
    -------
    numpy.ndarray with:
        conditional likelihoods for each sample in each class
    """
    new_classes_like = np.zeros((2, num_samples, num_classes))
    user_samples_rep = ~np.isnan(user_samples_classes)
    like_class_1 = (1 - np.nan_to_num(user_samples_classes)) * log_avoid_zero(np.tile(user_classes_perf[0][0].T.reshape(num_classes, num_users, 1), num_samples),
        additional_warning_message="user's tnr for each class",) + np.nan_to_num(user_samples_classes) * log_avoid_zero(np.tile(user_classes_perf[0][1].T.reshape(num_classes, num_users, 1), num_samples),
        additional_warning_message="user's fpr for each class",)
    like_class_1 = like_class_1 * user_samples_rep
    like_class_1 = like_class_1.sum(axis=1)
    like_class_2 = (1 - np.nan_to_num(user_samples_classes)) * log_avoid_zero(np.tile(user_classes_perf[1][0].T.reshape(num_classes, num_users, 1), num_samples),
        additional_warning_message="user's fnr for each class",) + np.nan_to_num(user_samples_classes) * log_avoid_zero(np.tile(user_classes_perf[1][1].T.reshape(num_classes, num_users, 1), num_samples),
        additional_warning_message="user's tpr for each class",)
    like_class_2 = like_class_2 * user_samples_rep
    like_class_2 = like_class_2.sum(axis=1)
    new_classes_like[0] = like_class_1.T + log_avoid_zero(1 - prior, additional_warning_message="(1- prior) samples class")
    new_classes_like[1] = like_class_2.T + log_avoid_zero(prior, additional_warning_message="prior samples class")
    return new_classes_like


def consensus_voting(user_classes, user_samples=None, th=0.5, max_iters=100):
    """
    Chooses classes for samples labeled by users using weighted voting.
    Classes are chosen using estimated users labeling quality by each class.
    The estimation of users' qualities is iterativly improved basing on
    majority voting weighted by previous users' qualities.
    Parameters
    ----------
    user_samples : numpy.ndarray
        default = None
	Boolean matrix indicating which samples were labeled by each user.
	Users should be represented by rows and samples by columns.
    user_classes : numpy.ndarray
        Array of classes that user assigned to each of the samples.
        Should user_samples = None, then user_classes have cells with nan,
        meaning that the user haven't evaluated the sample.
        Dimensions: classes, users, samples.
    th : float
        Threshold for majority voting, class will be assigned to sample if
        mean voting value for this class will exceed this threshold.
    max_iters : int
        Maximum number of iterations of users qualities estimation the
        algorithm will make.
    Returns
    -------
    dict with:
        user_classes_performance - user's tnr, fpr, fnr, tpr
        for each class
        user_tpr - matrix of user's true positive ratio
        for each class
        user_tnr - matrix of user's true negative ratio
        for each class
        samples_classes - classes chosen for each sample
        by weighted voting
        samples_logits - nonnormalized predictions
        of each samples' classess   
    """
    num_classes = user_classes.shape[0]
    if user_samples is None and len(user_classes.shape) == 3:
        user_samples_classes = user_classes
        num_users = user_samples_classes.shape[1]
        num_samples = user_samples_classes.shape[2]
    elif len(user_classes.shape) == 3 and not user_samples is None:
        num_samples = user_samples.shape[1]
        num_users = user_samples.shape[0]
        user_samples_classes = np.empty((num_classes, num_users, num_samples))
        user_samples_classes[:] = np.NaN
        user_samples_classes[:, user_samples] = user_classes[:, user_samples]
    else:
        warnings.warn("The input data is invalid")
    samples_logits = np.zeros((num_samples, num_classes))
    mv_old = np.zeros((num_samples, num_classes))
    mv_classes, labelled_samples = consensus_majority(user_samples_classes, num_classes, num_samples, th)
    ite = 0
    prior = (mv_classes[labelled_samples]).mean(axis=0)
    bool_to_start = True
    while (bool_to_start == True) or (np.any(mv_classes != mv_old) and (ite < max_iters)):
        user_classes_perf = user_classes_performance(user_samples_classes, mv_classes, num_classes, num_users, num_samples)
        new_classes_like_arr = new_classes_likelihood(user_samples_classes,
            user_classes_perf,
            prior,
            num_classes,
            num_users,
            num_samples,)
        mv_old = mv_classes
        mv_classes = (new_classes_like_arr[1, :, :] > new_classes_like_arr[0, :, :]).astype(int)
        mv_classes[~(np.array(labelled_samples))] = 0
        samples_logits = new_classes_like_arr[1, :, :] - new_classes_like_arr[0, :, :]
        ite = ite + 1
        bool_to_start = False 
    return {
        "user_classes_performance": user_classes_perf,
        "samples_classes": mv_classes,
        "user_tpr": user_classes_perf[1][1].mean(axis=1),
        "user_tnr": user_classes_perf[0][0].mean(axis=1),
        "samples_logits": samples_logits,
    }


def consensus_majority(user_samples_classes, num_classes, num_samples, th=0.5):
    """
    Chooses classes for samples labeled by users using majority voting.
    Classes are chosen using majority voting, i. e. every user
    is assumed to have the same quality.
    Parameters
    ----------
    user_samples_classes : numpy.ndarray
        Array of one hot encoded classes that user assigned to each of the samples. \
        Cells with nan mean that the user didn't evaluate the sample.
        Dimensions: classes, users, samples.
    num_classes : int
        A number representing the number of classes.
    num_samples : int
        A number representing the number of samples.
    th : float
        Threshold for majority voting, class will be assigned to sample
        if mean voting value for this class will exceed this threshold.
    Returns
    -------
    numpy.ndarray with classes per sample.
        Rows represent samples and columns classes.
    numpy.ndarray with information about samples.
        True - sample has been rated / False - sample hasn't been rated.
    """

    labelled_samples = np.any(~np.isnan(user_samples_classes), axis=(0, 1))
    user_classes_labelled = user_samples_classes[:, :, labelled_samples]
    mv_classes = np.zeros((num_samples, num_classes))
    labelled_mv_classes = (np.nanmean(user_classes_labelled, axis=1) > th).astype(int).T
    mv_classes[labelled_samples] = labelled_mv_classes
    return mv_classes, labelled_samples
