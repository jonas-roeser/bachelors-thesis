import numpy as np
import pandas as pd
from sklearn import metrics as sklearn_metrics
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


def get_predictions(model, X_tensors, y_split, subset_keys=['train', 'val', 'test'], save_as=False):
    '''
    Generate predictions with machine learning model.
    
    Parameters
    ----------
    model : model
        Model to generate predictions with.
    X_tensors : dict of tensors
        Dictionary of input tensors with subset keys.
    y_split : dict of arrays
        Dictionary of output arrays with subset keys.
    subset_keys : list, default ['train', 'val', 'test']
        List of subset keys.
    save_as : bool, default Fals
        Whether to save predictions in csv format at the specified location.
    
    Returns
    -------
    predictions : DataFrame
        Generated model predictions sorted by index.
    '''
    predictions = pd.DataFrame({'y_true': pd.concat([y_split[subset_key] for subset_key in subset_keys])})
    
    # Add model predictions
    y_pred = [model(X_tensors[subset_key]) for subset_key in subset_keys]
    predictions['y_pred'] = [pred.item() for subset in y_pred for pred in subset]
    
    # Add subset keys
    subsets = [[subset_key] * len(y_split[subset_key]) for subset_key in subset_keys]
    predictions['subset'] = [key for subset in subsets for key in subset]

    # Sort predictions by index
    predictions.sort_index(inplace=True)

    # Save predictions
    if save_as != True:
        predictions.to_csv(save_as)
    
    # Return predictions
    return predictions


def get_metrics(predictions, subset_keys=['train', 'val', 'test'], save_as=False, verbose=True):
    '''
    Calculate performance metrics for machine learning model.
    
    Parameters
    ----------
    predictions : DataFrame
        Contains index, y_true, y_pred and subset.
    subset_keys : list, default ['train', 'val', 'test']
        List of subset keys.
    save_as : bool, default False
        Whether to save performance metrics in csv format at the specified location.
    verbose : bool, default True
        Whether to print metrics to the console.
    
    Returns
    -------
    perf_metrics : DataFrame
        Calculated performance metrics.
    '''
    # Create DataFrame for storing performance metrics
    perf_metrics = pd.DataFrame(index=pd.Index(subset_keys, name='Subset'))
    
    # Define functions for calculating performance metrics
    def cpe(y_true, y_pred):
        return sum(abs(y_true - y_pred)) / sum(y_true)
    
    def rmse(y_true, y_pred):
        return sklearn_metrics.mean_squared_error(y_true, y_pred, squared=False)
        # (sum((y_true - y_pred) ** 2) / len(y_true)) ** 0.5

    def mae(y_true, y_pred):
        return sklearn_metrics.mean_absolute_error(y_true, y_pred)
        # sum(abs(y_true - y_pred)) / len(y_true)

    def r2(y_true, y_pred):
        return sklearn_metrics.r2_score(y_true, y_pred)
        # 1 - (sum((y_true - y_pred) ** 2) / sum((y_true - np.mean(y_true)) ** 2))
    
    # Since actuals are close to 0, MAPE would result in a very high value and is therefore not used

    # Create list containing all functions for calculating performance metrics
    metric_functions = [cpe, rmse, mae, r2]

    # Calculate performance metrics for each subset
    for subset in perf_metrics.index:
        # Use y_true & y_pred from specific subset
        y_true = predictions[predictions.subset == subset].y_true
        y_pred = predictions[predictions.subset == subset].y_pred

        # Calculate each metric
        for metric_function in metric_functions:
            perf_metrics.loc[subset, metric_function.__name__.upper()] = metric_function(y_true, y_pred)

    # Use y_true & y_pred of total dataset
    y_true = predictions.y_true
    y_pred = predictions.y_pred

    # Calculate values of each metric for total dataset
    for metric_function in metric_functions:
        perf_metrics.loc['total', metric_function.__name__.upper()] = metric_function(y_true, y_pred)

    if verbose == True:
        # Include index in column width computation
        auxiliary_df = perf_metrics.reset_index()

        # Convert values into strings
        auxiliary_df = auxiliary_df.astype(str)

        # Use only characters before decimal point for column width computation
        auxiliary_df.replace(r'\..+','', regex=True, inplace=True)

        # Define number of digits after decimal point
        n_decimals = 3

        # Replace strings of decimal values with their length + 1 for decimal point + n_decimals after decimal point
        auxiliary_df.iloc[:, 1:] = auxiliary_df.iloc[:, 1:].applymap(lambda x: len(x) + 1 + n_decimals)

        # Replace strings of non-decimal values with their length
        auxiliary_df.iloc[:, [0]] = auxiliary_df.iloc[:, [0]].applymap(len)

        # Include column name lengths in width computation
        auxiliary_df.loc['colname_lengths'] = [len(colname) for colname in auxiliary_df.columns]

        # Compute minimum width for each column
        min_col_widths = auxiliary_df.max().to_list()

        # Define inter-column spacing
        spacing = 2

        # Add inter-column spacing to all columns but the first
        col_widths = [min_col_widths[0]] + [min_col_width + spacing for min_col_width in min_col_widths[1:]]

        # Define header formatting
        def format_header(i, header):
            if i == 0:
                return f'{header : <{col_widths[i]}}'
            else:
                return f'{header : >{col_widths[i]}}'
        
        # Define entry formatting
        def format_entry(i, entry):
            if i == 0:
                return f'{entry : <{col_widths[i]}}'
            else:
                return f'{entry:{col_widths[i]}.{n_decimals}f}'

        # Print column headers
        print(''.join([format_header(i, header) for i, header in enumerate(perf_metrics.reset_index())]))

        # Print row seperator
        print('-' * sum(col_widths))

        # Print all rows except last
        for index, row in perf_metrics.reset_index().iloc[:-1].iterrows():
            print(''.join([format_entry(i, entry) for i, entry in enumerate(row)]))

        # Print row separator
        print('-' * sum(col_widths))

        # Print last row
        print(''.join([format_entry(i, entry) for i, entry in enumerate(perf_metrics.reset_index().iloc[-1])]))

    # Save performance metrics
    if save_as != False:
        perf_metrics.to_csv(save_as)

    # Return preformance metrics
    return perf_metrics