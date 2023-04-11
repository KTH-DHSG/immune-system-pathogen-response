from copy import copy
import numpy as np
import pandas as pd
from scipy.stats import entropy
import anndata
from dataset_lib import get_mask_anndata, get_samples
from sklearn.feature_selection import mutual_info_regression


class DistributionDiscretizer:
    def __init__(self, nbins=8, double_sided=True, tolerance=0.05, bounds=[0, 1]):
        """
        Initialize the DistributionDiscretizer object.

        @param nbins: Number of bins to use for discretization.
        @param double_sided: Whether to use a double-sided discretization.
        @param tolerance: Tolerance for bin border comparison.
        @param bounds: Bounds for the discretization.
        """
        self.nbins = nbins
        self.bin_border = np.linspace(bounds[0], bounds[1], nbins)
        self.bin_border = np.concatenate((self.bin_border, [np.inf]))
        self.double_sided = double_sided
        self.tolerance = tolerance
        if double_sided:
            self.bin_border = np.concatenate((-self.bin_border[1:], self.bin_border))
            self.bin_border = np.sort(self.bin_border)

    def get_bin_count(self, data):
        """
        Get the number of data points in each bin for the given data.

        @param data: The input data to be binned.
        @return: The count of data points in each bin.
        """
        if len(data.shape) == 1:
            data = data.reshape((-1, 1))
        bin_count = np.zeros((self.bin_border.shape[0] - 1, data.shape[1]))
        for i in range(self.bin_border.shape[0] - 1):
            bin_count[i, :] = np.sum((self.bin_border[i] - self.tolerance <= data)
                                     & (data < self.bin_border[i + 1] + self.tolerance), axis=0)
        return bin_count

    def get_bin_prob(self, data):
        """
        Get the probability distribution of data points in each bin for the given data.

        @param data: The input data to be binned.
        @return: The probability distribution of data points in each bin.
        """
        bin_count = self.get_bin_count(data)
        return bin_count / np.sum(bin_count, axis=0)

    def calc_entropy(self, data):
        """
        Calculate the entropy of the discretized data.

        @param data: The input data.
        @return: The entropy of the discretized data.
        """
        bin_count = self.get_bin_count(data)
        return entropy(bin_count, base=2, axis=0)

    def calc_difference(self, data1, data2):
        """
        Calculate the difference in entropy between two datasets.

        @param data1: The first dataset.
        @param data2: The second dataset.
        @return: The difference in entropy between the two datasets.
        """
        ent1 = self.calc_entropy(data1)
        ent2 = self.calc_entropy(data2)
        return ent2 - ent1

    def calc_emd(self, data1, data2):
        """
        Calculate the Earth Mover's Distance (EMD) between two datasets.

        @param data1: The first dataset.
        @param data2: The second dataset.
        @return: The Earth Mover's Distance between the two datasets.
        """
        bin_prob1 = self.get_bin_prob(data1)
        bin_prob2 = self.get_bin_prob(data2)
        emd = np.sum(np.abs(np.cumsum(bin_prob1 - bin_prob2, axis=0)), axis=0) / (bin_prob2.shape[0]-1)
        if self.double_sided:
            emd *= 2
        return emd

    def reset_bounds(self, bounds):
        """
        Reset the bounds for the discretization.

        @param bounds: New bounds for the discretization.
        """
        self.bin_border = np.linspace(bounds[0], bounds[1], self.nbins)
        self.bin_border = np.concatenate((self.bin_border, [np.inf]))


def get_binary_mutual_info(class0_data, class1_data):
    """
    Calculate the binary mutual information between two datasets.

    @param class0_data: The first dataset.
    @param class1_data: The second dataset.
    @return: The binary mutual information between the two datasets.
    """
    label_0 = np.zeros((class0_data.shape[0], 1))
    label_1 = np.ones((class1_data.shape[0], 1))

    class0_data_ = np.concatenate((class0_data, label_0), axis=1)
    class1_data = np.concatenate((class1_data, label_1), axis=1)

    data_tot_ = np.concatenate((class0_data_, class1_data), axis=0)
    mi = mutual_info_regression(data_tot_[:, :-1], data_tot_[:, -1])
    return mi


def get_binary_logdiff(class0_data, class1_data, epsilon=0.1):
    """
    Calculate the log difference between two datasets.

    @param class0_data: The first dataset.
    @param class1_data: The second dataset.
    @param epsilon: Small constant to avoid division by zero.
    @return: The log difference between the two datasets.
    """
    mean_cl0 = np.mean(class0_data, axis=0)
    mean_cl1 = np.mean(class1_data, axis=0)
    logdiff = (np.log(mean_cl0 + epsilon) - np.log(mean_cl1 + epsilon)) / np.abs(np.log(epsilon))
    return logdiff


class BinClassifierEMD:
    def __init__(self, distrib_func: DistributionDiscretizer=None, nfeatures=20,
                 column_names=None, prob_filter=None, class_names=None):
        """
        Initialize the BinClassifierEMD object.

        @param distrib_func: Distribution discretization function.
        @param nfeatures: Number of features to consider in the classification.
        @param column_names: Names of the columns in the input dataset.
        @param prob_filter: Probability filter to apply in the classification.
        @param class_names: Names of the classes in the classification.
        """
        if class_names is None:
            class_names = ['cl0', 'cl1']

        if distrib_func is None:
            distrib_func = DistributionDiscretizer(nbins=8)

        self.distrib_func = distrib_func
        self.nfeatures = nfeatures
        self.column_names = column_names
        self.class_names = class_names
        self.class0_probs = None
        self.class1_probs = None

        self.selected_columns = None
        self.selected_column_class0_probs = None
        self.selected_column_class1_probs = None

        self.filter = None

        if prob_filter is not None:
            self.filter = prob_filter / np.sum(prob_filter)

    def set_filter(self, prob_filter=None):
        self.filter = None
        if prob_filter is not None:
            self.filter = prob_filter / np.sum(prob_filter)
            if self.selected_column_class0_probs is not None:
                self.filter_probability_dists()

    def set_class_names(self, names):
        self.class_names = names

    def get_class_names(self):
        return self.class_names

    def filter_probability_dists(self):
        """
        Low pass filtering of the discretized probability densities
        """
        self.selected_column_class0_probs = \
            np.apply_along_axis(lambda m: np.convolve(m, self.filter, mode='same'), axis=1,
                                arr=self.selected_column_class0_probs)

        self.selected_column_class1_probs = \
            np.apply_along_axis(lambda m: np.convolve(m, self.filter, mode='same'), axis=1,
                                arr=self.selected_column_class1_probs)

    def fit(self, class0_data, class1_data, selected_columns=None, reset_bounds=False):
        """
        find the top nfeatures in terms of differential expression between class0 and class1 data
        """
        if reset_bounds:
            min0 = np.min(class0_data, axis=0)
            min1 = np.min(class1_data, axis=0)
            max0 = np.max(class0_data, axis=0)
            max1 = np.max(class1_data, axis=0)
            self.distrib_func.reset_bounds([np.minimum(min0, min1), np.maximum(max0, max1)])

        if selected_columns is None:
            emd = self.distrib_func.calc_emd(class1_data, class0_data)
            self.selected_columns = np.argpartition(emd, -self.nfeatures)[-self.nfeatures:]
        else:
            self.selected_columns = selected_columns
            self.nfeatures = len(selected_columns)

        self.class0_probs = self.distrib_func.get_bin_prob(class0_data)
        self.class1_probs = self.distrib_func.get_bin_prob(class1_data)

        if self.pca is None:
            self.selected_column_class0_probs = self.class0_probs[:, self.selected_columns]
            self.selected_column_class1_probs = self.class1_probs[:, self.selected_columns]
        else:
            print('INACTIVE FUNCTIONALITY. TO BE IMPLEMENTED.')

        if self.filter is not None:
            self.filter_probability_dists()

    def get_class_likelihoods(self, data):
        """
        Return two columns that are respectively the log likelihoods that each datapoint belongs inside the class
        and outside the class
        """
        prob_in = np.zeros((data.shape[0], 1))
        prob_out = np.zeros((data.shape[0], 1))
        data_cpy = data[:, self.selected_columns]
        for i in range(data.shape[0]):
            prob_in[i] = np.sum(np.log2(
                (self.distrib_func.get_bin_count(data_cpy[i:i + 1, :]) * self.selected_column_class0_probs) + 0.01))
            prob_out[i] = np.sum(np.log2(
                (self.distrib_func.get_bin_count(data_cpy[i:i + 1, :]) * self.selected_column_class1_probs) + 0.01))

        return np.concatenate((prob_in, prob_out), axis=1)

    def get_emds(self, data):
        """
        Return two columns that are respectively the EMD of the singular cell distance from the inside class
        distribution and the outside class distribution
        """
        dist_in = np.zeros((data.shape[0], 1))
        dist_out = np.zeros((data.shape[0], 1))
        data_cpy = data[:, self.selected_columns]
        for i in range(data.shape[0]):
            point_bins = self.distrib_func.get_bin_prob(data_cpy[i:i + 1, :])
            dist_in[i] = np.sum(self.distrib_func.calc_emd(self.selected_column_class0_probs, point_bins))
            dist_out[i] = np.sum(self.distrib_func.calc_emd(self.selected_column_class1_probs, point_bins))

        return np.concatenate((dist_in, dist_out), axis=1)

    def classify(self, data):
        return np.diff(self.get_class_likelihoods(data)) > 0


class MultiClassifier:
    def __init__(self, bin_classifiers: dict, default=None):
        """
        Initialize the MultiClassifier object.

        @param bin_classifiers: Dictionary of binary classifiers.
        @param default: Default class label if none of the binary classifiers are applicable.
        """
        self.bin_classifiers = bin_classifiers
        self.classes = list(self.bin_classifiers.keys())
        self.default = default
        self.child_classifier = dict()

    def get_likelihood(self, data):
        """
        Get the likelihood of each class for the given data.

        @param data: The input data.
        @return: A DataFrame with likelihoods of each class for each data point.
        """
        likelihood = np.zeros((data.shape[0], len(self.classes)))
        for cl in range(len(self.classes)):
            likelihood[:, cl] = -np.diff(self.bin_classifiers[self.classes[cl]].get_class_likelihoods(data)).reshape(
                (-1,))
        likelihood_df = pd.DataFrame(likelihood, columns=self.classes)
        if self.default is not None:
            likelihood_df[self.default] = 0
        return likelihood_df

    def predict(self, data):
        """
        Predict the class labels for the given data.

        @param data: The input data.
        @return: An array of predicted class labels for each data point.
        """
        likelihood = self.get_likelihood(data)
        prediction = likelihood.idxmin(axis=1).to_numpy()
        adjusted_prediction = self.child_classification(prediction, data)
        return adjusted_prediction

    def child_classification(self, prediction, data):
        """
        Adjust the predicted class labels based on child classifiers.

        @param prediction: The initial prediction.
        @param data: The input data.
        @return: An array of adjusted predicted class labels for each data point.
        """
        adjusted_prediction = copy(prediction)
        for cl in self.child_classifier.keys():
            adjusted_prediction[prediction == cl] = self.child_classifier[cl].predict(data[prediction == cl, :])
        return adjusted_prediction

    def add_child(self, label, classifier):
        """
        Add a child classifier for the given label.

        @param label: The class label for the child classifier.
        @param classifier: The child classifier.
        """
        if isinstance(classifier, dict):
            self.child_classifier[label] = MultiClassifier(classifier, default=label)
        elif isinstance(classifier, MultiClassifier):
            self.child_classifier[label] = classifier


def get_adata_classifier(adata: anndata.AnnData, filter_dict0: dict, filter_dict1: dict,
                         n_features=10, n_samples=100, n_iterations=1, discretizer=None, scaler=None):
    """
        Get an AnnData classifier based on given filters and settings.

        @param adata: Input AnnData object.
        @param filter_dict0: Filter dictionary for the first class.
        @param filter_dict1: Filter dictionary for the second class.
        @param n_features: Number of features to consider in the classification.
        @param n_samples: Number of samples to use for training the classifier.
        @param n_iterations: Number of iterations to perform for feature selection.
        @param discretizer: Distribution discretization function.
        @param scaler: Scaler object to normalize the data.
        @return: A trained BinClassifierEMD object.
        """

    class0_cellrows = get_mask_anndata(adata, filter_dict0)
    class0_data = adata[class0_cellrows]

    class1_cellrows = get_mask_anndata(adata, filter_dict1)
    class1_data = adata[class1_cellrows]

    votes = []
    for it in range(n_iterations):
        'Train'
        class0_samples = get_samples(class0_data, n_samples, scaler=scaler, replacement=True)
        class1_samples = get_samples(class1_data, n_samples, scaler=scaler, replacement=True)
        classifier = BinClassifierEMD(discretizer, nfeatures=n_features)
        classifier.fit(class0_data=class0_samples, class1_data=class1_samples)
        votes.extend(list(classifier.selected_columns))

    vote_count = np.array([[i, votes.count(i)] for i in set(votes)])
    vote_count = vote_count[vote_count[:, 1].argsort(), :]

    class0_samples = get_samples(class0_data, n_samples, scaler=scaler, replacement=True)
    class1_samples = get_samples(class1_data, n_samples, scaler=scaler, replacement=True)

    classifier = BinClassifierEMD(discretizer, nfeatures=n_features)
    classifier.fit(class0_data=class0_samples, class1_data=class1_samples, selected_columns=vote_count[-n_features:, 0])

    return classifier


class NaiveClassifier:
    def __init__(self, distrib_func: DistributionDiscretizer = None, column_names = None):
        """
        Initializes the NaiveClassifier with an optional DistributionDiscretizer instance and column names.

        @param distrib_func: An instance of DistributionDiscretizer, default is None.
        @param column_names: Column names for the dataset, default is None.
        """
        if distrib_func is None:
            distrib_func = DistributionDiscretizer(nbins=8)

        self.distrib_func = distrib_func
        self.column_names = np.array(column_names)
        self.class_names = None
        self.class_probs = {}

    def fit(self, data_dict: dict):
        """
        Fits the NaiveClassifier on the provided data dictionary.

        @param data_dict: A dictionary with keys as class names and values as feature distributions for each class.
        """
        self.class_names = list(data_dict.keys())
        self.class_probs = {}
        for c in self.class_names:
            self.class_probs[c] = self.distrib_func.get_bin_prob(data_dict[c])

    def get_class_likelihoods(self, data):
        """
        Returns a dataframe with the likelihood of each data point belonging to each class.

        @param data: A 2D numpy array or pandas DataFrame with input data.
        @return: A pandas DataFrame with likelihoods for each class.
        """
        likelihoods = pd.DataFrame(columns=self.class_names)
        n_classes = len(self.class_names)
        for i in range(data.shape[0]):
            likelihood = np.zeros((n_classes,))
            for c in range(n_classes):
                likelihood[c] = np.sum(np.log2(
                    self.distrib_func.get_bin_count(data[i:i+1, :]) * self.class_probs[self.class_names[c]] + 0.01
                ))
            likelihoods.loc[i] = likelihood
        return likelihoods

    def classify(self, data):
        """
        Classifies the given data points using the trained NaiveClassifier.

        :param data: A 2D numpy array or pandas DataFrame with input data.
        :return: A 1D numpy array with predicted class labels.
        """
        return self.class_names[np.argmax(self.get_class_likelihoods(data).to_numpy(), axis=1)]
