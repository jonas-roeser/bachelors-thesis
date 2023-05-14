import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.utils import shuffle as shuffle_arrays

def train_test_val_dict(*arrays, val_size, test_size, shuffle=True, random_state=42, verbose=True):
    '''
    Split arrays into training, validation and testing arrays.
    
    ---------
    *arrays : array or dataframe
        Arbitrary number of arrays to split.
    val_size : int or float
        Size of validation data in absolute or relative terms.
    test_ size : int or float
        Size of testing data in absolute or relative terms.
    shuffle : bool, default True
        Whether to shuffle the dataset before splitting.
    random_state : int, default 42
        Determines random number generation for shuffling the data. None for non-deterministic results.
    verbose : bool, default True
        Whether to print array lengths to the console.
    
    Returns
    -------
    dict_list : list of dictionaries
        Training, validation and testing arrays split using val_size and test_size.
    '''
    array_len = len(arrays[0])

    if not all(len(array) == array_len for array in arrays):
        raise ValueError('Arrays must be of equal length!')
    
    if shuffle == True:
        arrays = shuffle_arrays(*arrays, random_state=random_state)

    if type(val_size) is float:
        val_size = int(array_len * val_size)
    
    if type(test_size) is float:
        test_size = int(array_len * test_size)
    
    train_size = array_len - (val_size + test_size)

    if verbose == True:
        print(f'{"Array":<10}{"Length (n)":>12}{"Length (%)":>12}')
        print(f'{34 * "-"}')
        print(f'{"training":<10}{train_size:>12}{(train_size / array_len):12.2%}')
        print(f'{"validation":<10}{val_size:>12}{(val_size / array_len):12.2%}')
        print(f'{"testing":<10}{test_size:>12}{(test_size / array_len):12.2%}')
        print(f'{34 * "-"}')
        print(f'{"total":<10}{array_len:>12}{(array_len / array_len):12.2%}')

    dict_list = []

    for array in arrays:
        dict_list.append({
            'train': array[:train_size],
            'val': array[train_size:-test_size],
            'test': array[-test_size:]
            })

    return dict_list


def get_predictions(model, X_tensors, y_split, dataset_labels=['train', 'val', 'test'], save_as=False):
    '''
    Generate predictions with machine learning model.
    
    Parameters
    ----------
    model : ?
        Model to generate predictions with.
    y_true : array
        True output.
    verbose : bool, default True
        Whether to print metrics to the console.
    save : 
    
    Returns
    -------
    predictions : DataFrame
        Generated predictions.
    '''
    predictions = pd.DataFrame({
        'y_true': pd.concat([y_split[key] for key in dataset_labels]),
        'y_pred': np.nan,
        'dataset': np.nan
        })
    
    y_pred = [model(X_tensors[key]) for key in dataset_labels]
    predictions.y_pred = [pred.item() for dataset in y_pred for pred in dataset]

    labels = [[key] * len(y_split[key]) for key in dataset_labels]
    predictions.dataset = [label for x in labels for label in x]

    predictions.sort_index(inplace=True)

    # Save predictions
    if save_as != True:
        predictions.to_csv(save_as)
    
    return predictions


def calc_metrics(predictions, dataset_labels=['train', 'val', 'test'], save_as=False, verbose=True):
    # adapt this function to calculate metrics for all three (train, val, test) and save using model name
    '''
    Calculate performance metrics for machine learning model.
    
    Parameters
    ----------
    predictions : DataFrame
        Contains index, y_true, y_pred and dataset (one of 'train', 'val' or 'test).
    verbose : bool, default True
        Whether to print metrics to the console.
    
    Returns
    -------
    perf_metrics : DataFrame
        Calculated performance metrics.
    '''
    perf_metrics = pd.DataFrame({
        'CPE': np.nan,
        'RMSE': np.nan,
        'MAE': np.nan,
        'R2': np.nan
        }, index=pd.Index(dataset_labels, name='Dataset'))
    
    for dataset in perf_metrics.index:
        y_true = predictions[predictions.dataset == dataset].y_true
        y_pred = predictions[predictions.dataset == dataset].y_pred

        perf_metrics.loc[dataset, 'CPE'] = sum(abs(y_true - y_pred)) / sum(y_true)
        perf_metrics.loc[dataset, 'RMSE'] = metrics.mean_squared_error(y_true, y_pred, squared=False)# (sum((y_true - y_pred) ** 2) / len(y_true)) ** 0.5
        perf_metrics.loc[dataset, 'MAE'] = metrics.mean_absolute_error(y_true, y_pred)# sum(abs(y_true - y_pred)) / len(y_true)
        perf_metrics.loc[dataset, 'R2'] = r2 = metrics.r2_score(y_true, y_pred)# 1 - (sum((y_true - y_pred) ** 2) / sum((y_true - np.mean(y_true)) ** 2))
        # Since actuals are close to 0, MAPE would result in a very high value and is therefore not used
    
    if verbose == True:
        index_width = max(len(perf_metrics.index.name), perf_metrics.index.str.len().max())
        value_widths = {metric:max(len(metric), column.astype(str).str.split('.').str[0].str.len().max()) for metric, column in zip(perf_metrics.columns, [perf_metrics[metric] for metric in perf_metrics.columns])}
        n_decimals = 3
        spacing = 3

        print(f'{perf_metrics.index.name : <{index_width}}' + ''.join([f'{metric : >{value_widths[metric] + spacing + n_decimals}}' for metric in perf_metrics.columns]))
        print('-' * index_width + '-' * (sum(value_widths.values()) + len(value_widths) * (spacing + n_decimals)))
        for dataset_label in dataset_labels:
            print(f'{dataset_label : <{index_width}}' + ''.join([f'{perf_metrics.loc[dataset_label][metric_label]:{value_widths[metric_label] + spacing + n_decimals}.{n_decimals}f}' for metric_label in perf_metrics.columns]))
    
    if save_as != False:
        perf_metrics.to_csv(save_as)

    return perf_metrics