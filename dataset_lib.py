import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import QuantileTransformer, StandardScaler, Normalizer
import anndata


def get_samples(dataset: anndata.AnnData, nsamples, replacement=False, scaler=None, return_labels=None,
                selected_columns=None):
    """
    Function to get a subset of samples from the dataset.

    @param dataset: An anndata.AnnData object containing the dataset
    @param nsamples: Number of samples to extract, or 'all' to extract all samples
    @param replacement: Whether to sample with replacement (True) or without replacement (False)
    @param scaler: An optional scaler to preprocess the samples
    @param return_labels: An optional label to return along with the samples (e.g., 'index')
    @param selected_columns: An optional list of selected columns to extract from the dataset

    @return: A subset of samples from the dataset, and optionally, their associated labels
    """
    nrows = dataset.shape[0]
    if nsamples == 'all':
        nsamples = nrows

    if replacement:
        samples_id = random.choices(range(0, nrows), k=nsamples)
    else:
        samples_id = random.sample(range(0, nrows), k=nsamples)

    samples = dataset[samples_id].to_df().fillna(0).reset_index().drop('index', axis=1)
    if selected_columns is not None:
        samples = samples[selected_columns]

    if scaler is not None:
        samples = scaler.transform(samples)

    if return_labels is not None:
        if return_labels == 'index':
            labels = dataset[samples_id].to_df().index
        else:
            labels = dataset[samples_id].obs[return_labels].reset_index().drop('index', axis=1).to_numpy().reshape(
                (-1,))
        return samples, labels
    else:
        return samples


def get_mask_anndata(dataset: anndata.AnnData, criteria: dict):
    """
    Function to create a mask for rows in the dataset based on the provided criteria.

    @param dataset: An anndata.AnnData object containing the dataset
    @param criteria: A dictionary defining the criteria for masking the dataset

    @return: A NumPy array containing a mask for the dataset based on the given criteria
    """
    mask = np.full((dataset.shape[0],), True)
    for d in criteria.keys():
        if type(criteria[d]) is list:
            temp_mask = np.full((dataset.shape[0],), False)
            for crit in criteria[d]:
                temp_mask = temp_mask | (dataset.obs[d] == crit)
        else:
            temp_mask = dataset.obs[d] == criteria[d]
        mask = mask & temp_mask
    return mask


def get_gene_mask_anndata(dataset: anndata.AnnData, gene_n, level=0):
    """
    Function to create a mask for rows in the dataset based on the gene expression level.

    @param dataset: An anndata.AnnData object containing the dataset
    @param gene_n: Gene index to create the mask based on its expression level
    @param level: Expression level threshold for masking

    @return: A NumPy array containing a mask for the dataset based on the given gene expression level
    """
    mask = np.full((dataset.shape[0],), True)
    mask[:] = dataset.X[:, gene_n] > level
    return mask


def get_freqs_anndata(dataset: anndata.AnnData, criteria: dict):
    """
    Function to compute the relative frequency of rows according to a given criteria.

    @param dataset: An anndata.AnnData object containing the dataset
    @param criteria: A dictionary defining the criteria for computing the relative frequency

    @return: A NumPy array containing the relative frequency of rows based on the given criteria
    """
    assert len(criteria) == 1
    count = []
    for d in criteria.keys():
        assert type(criteria[d]) is list
        for c in criteria[d]:
            count.append(np.sum(get_mask_anndata(dataset, {d: c})))
    count = np.array(count)
    freq = count / np.maximum(np.sum(count), 1)
    return freq


def get_scaler(dataset, nsamples=10000, scaler_type='quant', selected_columns=None):
    """
    Function to get a scaler to preprocess the dataset based on a given scaler_type.

    @param dataset: An anndata.AnnData object containing the dataset
    @param nsamples: Number of samples to extract for fitting the scaler
    @param scaler_type: The type of scaler to use ('quant', 'standard', 'normal', 'log')
    @param selected_columns: An optional list of selected columns to extract from the dataset

    @return: A fitted scaler object
    """
    samples = get_samples(dataset, nsamples, selected_columns=selected_columns)
    if scaler_type == 'quant':
        scaler = QuantileTransformer(output_distribution="uniform").fit(samples)
    elif scaler_type == 'standard':
        scaler = StandardScaler().fit(samples)
    elif scaler_type == 'normal':
        scaler = Normalizer().fit(samples)
    elif scaler_type == 'log':
        scaler = LogScalerNorm().fit(samples)
    else:
        scaler = None

    return scaler


class LogScalerNorm:
    """
    Custom LogScalerNorm class to normalize data using log transformation.
    """
    def __init__(self, data=None):
        self.normalizer = None
        if data is not None:
            self.fit(data)

    def fit(self, data: pd.DataFrame):
        """
        Fit the LogScalerNorm to the given dataset.

        @param data: A pandas.DataFrame containing the dataset to fit

        @return: A fitted LogScalerNorm object
        """
        self.normalizer = np.maximum(np.max(data, axis=0).to_numpy(), 1)
        return self

    def transform(self, data: pd.DataFrame):
        """
        Transform the dataset using the fitted LogScalerNorm.

        @param data: A pandas.DataFrame containing the dataset to transform

        @return: A transformed dataset as a NumPy array
        """
        transformed = (np.log2(data + 1)).to_numpy() / self.normalizer
        return transformed

    def inverse_transform(self, data: pd.DataFrame):
        """
        Inverse transform the dataset back to its original form.

        @param data: A pandas.DataFrame containing the transformed dataset to inverse transform

        @return: The original dataset as a NumPy array
        """
        inverse_normalized = data * self.normalizer
        inverse_transformed = (np.power(inverse_normalized, 2) - 1).to_numpy()
        return inverse_transformed
