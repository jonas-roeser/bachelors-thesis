import numpy as np
import pandas as pd
import random
import math
import requests
from urlsigner import sign_url
from pathlib import Path
from PIL import Image
import os
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn import metrics as sklearn_metrics

def subset_index(index, test_size=.2, val_size=0, k_folds=None, shuffle=True, random_state=42, return_keys=True, save_index_as=None, save_keys_as=None, save_sizes_as=None, verbose=True):
    '''
    Split arrays into training, validation and testing subsets.
    
    Parameters
    ---------
    index : index or array
        Index used for subsetting.
    test_ size : int or float, default .2
        Size of testing subset in absolute or relative terms.
    val_size : int or float, default 0
        Size of validation subset in absolute or relative terms.
    k_folds : int, default None
        Number of folds for cross-validation.
    shuffle : bool, default True
        Whether to shuffle the dataset before splitting.
    random_state : int, default 42
        Determines random number generation for shuffling the data. None results in non-deterministic results.
    return_keys : bool, default True
        Whether to return subset keys.
    save_index_as : str, default None
        Where to save subset index in csv format.
    save_keys_as : str, default None
        Where to save subset keys in csv format.
    save_sizes_as : str, default None
        Where to save subset sizes in csv format.
    verbose : bool, default True
        Whether to print subset sizes to the console.
    
    Returns
    -------
    subset_index : Series
        Series of subsets, properly indexed.
    '''
    # Use length of index as reference point
    index_len = len(index)

    # Turn index into list if not already a list
    if not isinstance(index, list):
        index = index.to_list()

    # Shuffle index based on random state
    if shuffle == True:
        random.Random(random_state).shuffle(index)

    # Prepare for splitting arrays into k-folds and testing set
    if k_folds is not None:
        # Calculate absolute size of testing set if not already given
        if type(test_size) is float:
            test_size = int(index_len * test_size)
        
        # Calculate absolute size of c set based on size of testing set
        fold_size = int((index_len - test_size) / k_folds)

        # Account for remainder of k_fold division by allowing size of testing set to increase
        test_size = index_len - k_folds * fold_size
        
        # Create list of subset keys
        subset_keys = []

        # Add subset key for every fold
        for fold in range(1, k_folds + 1):
            subset_keys.append(f'fold_{fold}')

        # Create dictionary for subset sizes
        size_dict = {subset_key: fold_size for subset_key in subset_keys}
        
        # Add subset key for testing set
        subset_keys.append('test')

        # Add size of testing set to dictionary
        size_dict['test'] = test_size
    
    # Prepare for splitting arrays into training, validation and testing sets
    else:
        # Calculate absolute size of validation set if not already given
        if type(val_size) is float:
            val_size = int(index_len * val_size)
        
        # Calculate absolute size of testing set if not already given
        if type(test_size) is float:
            test_size = int(index_len * test_size)
        
        # Calculate absolute size of training set based on size of validation and testing set
        train_size = index_len - (val_size + test_size)

        # Define subset keys
        subset_keys = ['train', 'val', 'test']

        # Create dictionary for subset sizes
        size_dict = {
            'train': train_size,
            'val': val_size,
            'test': test_size
            }

    # Create dataframe for storing subset sizes
    if verbose == True or save_sizes_as is not None:
        subset_sizes = pd.DataFrame(columns=['Size (n)', 'Size (%)'], index=pd.Index(subset_keys, name='Subset'))

        # Store size of each subset in data frame
        for subset_key in subset_keys:
            subset_sizes.loc[subset_key] = [size_dict[subset_key], size_dict[subset_key] / index_len]

        # Store total size in data frame
        subset_sizes.loc['total'] = [index_len, index_len / index_len]

    # Print information to console
    if verbose == True:
        # Include index in column width computation
        auxiliary_df = subset_sizes.reset_index()

        # Convert values into strings
        auxiliary_df = auxiliary_df.astype(str)

        # Use only characters before decimal point for column width computation
        auxiliary_df.replace(r'\..+','', regex=True, inplace=True)

        # Define number of digits after decimal point
        n_decimals = 2

        # Replace strings of decimal values with their length + 1 for decimal point + n_decimals after decimal point
        auxiliary_df.iloc[:, [2]] = auxiliary_df.iloc[:, [2]].applymap(lambda x: len(x) + 1 + n_decimals)

        # Replace strings of non-decimal values with their length
        auxiliary_df.iloc[:, :2] = auxiliary_df.iloc[:, :2].applymap(len)

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
            elif i == 2:
                return f'{entry:{col_widths[i]}.{n_decimals}%}'
            else:
                return f'{entry:{col_widths[i]}.0f}'

        # Print column headers
        print(''.join([format_header(i, header) for i, header in enumerate(subset_sizes.reset_index())]))

        # Print row seperator
        print('-' * sum(col_widths))

        # Print all rows except last
        for rownum, row in subset_sizes.reset_index().iloc[:-1].iterrows():
            print(''.join([format_entry(colnum, entry) for colnum, entry in enumerate(row)]))

        # Print row separator
        print('-' * sum(col_widths))

        # Print last row
        print(''.join([format_entry(i, entry) for i, entry in enumerate(subset_sizes.reset_index().iloc[-1])]))

    # Create subset index
    subset_list = [[subset_key] * size_dict[subset_key] for subset_key in subset_keys]

    # Unpack subset index
    subset_list = [subset_key for subset in subset_list for subset_key in subset]
    
    # Crate subset index from subset list
    subset_index = pd.DataFrame({'subset': subset_list}, index=index)

    # Sort subset index by index
    subset_index.sort_index(inplace=True)
    
    # Save subset index
    if save_index_as is not None:
        Path(os.path.dirname(save_index_as)).mkdir(parents=True, exist_ok=True)
        subset_index.to_csv(save_index_as)
    
    # Save subset keys
    if save_keys_as is not None:
        Path(os.path.dirname(save_keys_as)).mkdir(parents=True, exist_ok=True)
        pd.Series(subset_keys).to_csv(save_keys_as, index=False)
    
    # Save subset sizes
    if save_sizes_as is not None:
        Path(os.path.dirname(save_sizes_as)).mkdir(parents=True, exist_ok=True)
        subset_sizes.to_csv(save_sizes_as)
    
    # Return subset index and subset keys
    if return_keys == True:
        return subset_index, subset_keys
    
    # Return subset index
    else:
        return subset_index


class MultiModalDataset(Dataset):
    '''
    Custom class for multi model dataset.

    '''

    def __init__(self, data, target_name, to_drop=None, image_directory=None, image_transformation=transforms.ToTensor(), subset_index=None, subset=None, input_scaler=None, target_scaler=None, categorical_encoder=None, numerical_imputer=None, data_overview=None):
        '''        
        Parameters
        ----------
        data : str
            Path to text data.
        target_name : str
            Name of target variable.
        to_drop : str or list
            Variables to drop.
        image_directory : str or list of str, default None
            Image directory or list of image directories.
        image_transformation : callable, default transforms.ToTensor()
            Function for transforming image vectors.
        subset_index : str, default None
            Path to subset index.
        subset : str, default None
            Subset to return, if None returns all data.
        input_scaler : callable, default None
            Scaler for scaling the input vector.
        target_scaler : callable, default None
            Scaler for scaling the target vector.
        categorical_encoder : callable, default None
            Encoder for encoding categorical variables.
        numerical_imputer : callable, default None
            Imputer for imputing numerical variables.
        data_overview : str, default None
            Path to data overview.
        '''
        # Load text data
        self.data = pd.read_parquet(data)

        # Define target variable
        self.target_name = target_name

        # Define variables to drop
        if to_drop is not None and not isinstance(to_drop, list):
            to_drop = [to_drop]
        to_drop = to_drop if to_drop else []

        # Drop specified variables
        self.data = self.data.drop(columns=to_drop)

        # Define image directory
        if image_directory is not None and not isinstance(image_directory, list):
            image_directory = [image_directory]
        self.image_directory = image_directory if image_directory else []

        # Define subset index
        self.subset_index = subset_index

        # Define subset
        self.subset = subset

        # Use entire index as training index if not overwritten by subset index
        training_index = self.data.index

        # Load subset index
        if self.subset_index is not None:
            self.subset_index = pd.read_csv(self.subset_index, index_col=0)

            # Store training & non-training index for fitting scalers, encoders & imputers before it is overwritten
            training_index = self.subset_index[self.subset_index.subset == 'train'].index
            non_training_index = self.subset_index[self.subset_index.subset != 'train'].index

            # Limit subset index to subset
            if self.subset is not None:
                self.subset_index = self.subset_index[self.subset_index.subset == subset]

        # Define image transformations
        self.image_transformation = image_transformation

        # Define other transformations
        self.input_scaler = input_scaler
        self.target_scaler = target_scaler
        self.categorical_encoder = categorical_encoder
        self.numerical_imputer = numerical_imputer

        # Require data overview when either categorical encoder or numerical imputer is set
        if data_overview is None and (self.categorical_encoder is not None or self.numerical_imputer is not None):
            raise AttributeError('data overview is required when either categorical encoder or numerical imputer is set')

        # Get dictionary of columns for each variable type
        if data_overview is not None:
            # Load data overview
            data_overview = pd.read_csv(data_overview, index_col=0)

            # Remove target variable from data overview
            data_overview = data_overview.loc[data_overview.column != self.target_name]

            # Remove dropped variables from data overview
            data_overview = data_overview.loc[~data_overview.column.isin(to_drop)]

            # Create dictionary for storing variable types
            self.variable_types = {}
            
            # Store list of columns for each variable type
            for variable_type in data_overview.variable_type.unique():
                self.variable_types[variable_type] = data_overview.column[data_overview.variable_type == variable_type].to_list()

        # Create target vector
        self.y = self.data[[self.target_name]]
        
        # Apply target scaler
        if self.target_scaler is not None:
            # Fit scaler on training set
            self.target_scaler.fit(self.y.loc[training_index])

            # Scale values in entire set
            self.y = pd.DataFrame(self.target_scaler.transform(self.y), index=self.y.index)

        # Create input vector
        self.X_text = self.data.drop(columns=self.target_name)

        # Encode categorical variables
        if self.categorical_encoder is not None:
            # Fit encoder on training set
            self.categorical_encoder.fit(self.X_text.loc[training_index, self.variable_types['categorical']], self.y.loc[training_index])

            # Encode values in entire set: as per category_econders docs, y should be provided when transforming input vector of training set
            self.X_text.loc[training_index, self.variable_types['categorical']] = self.categorical_encoder.transform(self.X_text.loc[training_index, self.variable_types['categorical']], self.y.loc[training_index])

            # This becomes superflous when subset index is None, since in that case training_index = self.data.index 
            if subset_index is not None:
                # y should not be provided when tranforming input vector of non-training set
                self.X_text.loc[non_training_index,self.variable_types['categorical']] = self.categorical_encoder.transform(self.X_text.loc[non_training_index, self.variable_types['categorical']])

        # Impute numerical variables
        if self.numerical_imputer is not None:
            # Fit imputer on training set
            self.numerical_imputer.fit(self.X_text.loc[training_index, self.variable_types['numerical']])

            # Impute values in entire set
            self.X_text[self.variable_types['numerical']] = self.numerical_imputer.transform(self.X_text[self.variable_types['numerical']])

        # Apply input scaler
        if self.input_scaler is not None:
            # Fit scaler on training set
            self.input_scaler.fit(self.X_text.loc[training_index])

            # Scale values in entire set
            self.X_text = self.input_scaler.transform(self.X_text)

        # Convert input and target vectors to numpy arrays
        self.X_text = self.X_text.astype('float32')
        if isinstance(self.X_text, pd.Series) or isinstance(self.X_text, pd.DataFrame):
            self.X_text = self.X_text.to_numpy()
        self.y = self.y.astype('float32')
        if isinstance(self.y, pd.Series) or isinstance(self.y, pd.DataFrame):
            self.y = self.y.to_numpy()
        
        # Limit input and target vectors to subset index
        self.X_text = self.X_text[self.data.index.get_indexer(self.subset_index.index)]
        self.y = self.y[self.data.index.get_indexer(self.subset_index.index)]
        
        # Store input and target vectors as tensors
        self.X_text = torch.tensor(self.X_text, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    # Get length of data
    def __len__(self):
        # If subset index is used, base length on subset index, since it might not include all text data
        if self.subset_index is not None:
            return len(self.subset_index)
        
        # Otherwise base length on text data
        else:
            return len(self.text_data)

    # Get actual data
    def __getitem__(self, index):
        # Convert iloc index into loc index
        if self.subset_index is not None:
            loc_index = self.subset_index.index[index]
        else:
            loc_index = self.data.index[index]

        X_images = []
        # Create image vectors
        if self.image_directory:
            # Load one image from every directory
            for i, directory in enumerate(self.image_directory):
                # Load image using PIL
                image = Image.open(f'{directory}{loc_index}.png').convert('RGB')

                # Apply transformations to the image
                if self.image_transformation is not None:
                    image = self.image_transformation(image)

                # Store image vector as tensor (should already happen during transform)
                if not torch.is_tensor(image):
                    image = transforms.ToTensor()(image)
                
                # Add image to list
                X_images.append(image)

        # Return sample dictionary
        return self.X_text[index], *X_images, self.y[index]


def data_overview(df):
    '''
    Generate overview of a data frame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data frame for which to generate an overview.
    
    Returns
    -------
    data_overview : pd.DataFrame
        Overview of the data frame.
    '''
    # Create data overview
    data_overview = pd.DataFrame({'column': df.columns.values})

    # Get values for each column
    for i, column in enumerate(df):
        # Get data type
        data_overview.loc[i, 'dtype'] = df[column].dtype

        # Get number of unique values
        data_overview.loc[i, 'n_unique'] = df[column].nunique()

        # Get number of NA values
        data_overview.loc[i, 'n_missing'] = df[column].isna().sum()

    # Use most recent pandas data types (e.g. pd.NA)
    data_overview = data_overview.convert_dtypes()

    # Return data overview
    return data_overview


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


def get_satellite_image(location_lat, location_long, zoom_level, size, maptype, key, secret=None):
    '''
    Request satellite image from Google Maps Static API.
    
    Parameters
    ----------
    location_lat : float
        Latitude of image location.
    location_long : float
        Longitude of image location
    zoom_level : int
        Image zoom level between 0 and 21 (based on Mercator projection).
    size : str
        Image resolution in pixels, e.g. '640x640'.
    maptype : str
        Image map type, e.g. 'satellite'.
    key : str, default None
        Google Maps Static API key.
    secret : str
        Google Maps Static API secret.
    
    Returns
    -------
    image : bytes
        Image in byte format.
    '''
    # Set url for Google Maps Static API
    url = 'https://maps.googleapis.com/maps/api/staticmap'

    # Set request params
    params = {
        'center': f'{location_lat},{location_long}',
        'zoom': zoom_level,
        'size': size,
        'maptype': maptype,
        'key': key
        }

    # Sign URL
    if secret != None:
        url = sign_url(requests.Request('GET', url, params=params).prepare().url, secret=secret)
    
    # Request image from Google Maps Static API
    image = requests.get(url).content

    # Return image
    return image

# Define function for getting satellite image
def save_satellite_image(save_as, overwrite, *args):
    '''
    Save satellite image from Google Maps Static API.
    
    Parameters
    ----------
    save_as : str
        Location where to save image.
    overwrite : bool
        Whether to overwrite existing image.
    *args : float
        Arguments passed on to get_satellite_image().
    
    Returns
    -------
    None
        Saves image at the specified location.
    '''
    # Do not send requests for already existing files
    if overwrite == False and Path(save_as).is_file():
        pass
    else:
        # Request image from Google Maps Static API
        image = get_satellite_image(*args)

        # Create folder if it does not exist already
        Path(os.path.dirname(save_as)).mkdir(parents=True, exist_ok=True)

        # Save image
        with open(save_as, 'wb') as f:
            f.write(image)

def m_per_px(zoom_level, location_lat):
    '''
    Calculate image resolution in meters per pixel.

    Parameters
    ----------
    zoom_level : int
        Image zoom level between 0 and 21 (based on Mercator projection).
    location_lat : float
        Latitude of image location.

    Returns
    -------
    resolution : float
        Calculated image resolution in meters per pixel.
    '''
    # Set geographic constant as defined by WGS84
    a = 6378137 # equatorial radius

    # Calculate equatorial circumference
    C = 2 * math.pi * a

    # Convert latitude to radian
    lat_radian = math.pi / 180 * location_lat

    # Calculate image resolution
    resolution = C / 256 * math.cos(lat_radian) / 2**zoom_level

    # Return image resolution
    return resolution


def get_image_size(width_px, height_px, zoom_level, location_lat):
    '''
    Calculate image size in meters.
    
    Parameters
    ----------
    width_px : int
        Image width in pixels.
    height_px : int
        Image height in pixels.
    zoom_level : int
        Image zoom level between 0 and 21 (based on Mercator projection).
    location_lat : float
        Latitude of image location.
    
    Returns
    -------
    width_m, height_m : float
        Image dimensions in meters.
    '''
    # Calculate image width in meters
    width_m = width_px * m_per_px(zoom_level, location_lat)

    # Calculate image height in meters
    height_m = height_px * m_per_px(zoom_level, location_lat)

    # Return image dimensions in meters
    return width_m, height_m


def get_image_boundaries(location_lat, location_long, width_px, height_px, zoom_level):
    '''
    Calculate image geographic image boundaries based on location and dimensions.
    
    Parameters
    ----------
    location_lat : float
        Latitude of image location.
    location_long : float
        Longitude of image location
    width_px : int
        Image width in pixels.
    height_px : int
        Image height in pixels.
    zoom_level : int
        Image zoom level between 0 and 21 (based on Mercator projection).
    
    Returns
    -------
    lat_top, lat_bottom, long_left, long_right : float
        Image boundaries as coordinates.
    '''
    # Set geographic constants as defined by WGS84
    a = 6378137 # equatorial radius
    e = 0.00669437999014**(1 / 2) # eccentricity

    # Convert latitude to radian
    lat_radian = math.pi / 180 * location_lat

    # Calculate number of meters per degree of latitude
    m_per_lat = abs(math.pi * a * (1 - e**2) / (180 * (1 - e**2 * math.sin(lat_radian)**2)**(3 / 2)))

    # Calculate degrees of latitude per meter
    lat_per_m = 1 / m_per_lat

    # Calculate number of meters per degree of longitude
    m_per_long = abs(math.pi * a * math.cos(lat_radian) / (180 * (1 - e**2 * math.sin(lat_radian)**2)**(1 / 2)))

    # Calculate degrees of longitude per meter
    long_per_m = 1 / m_per_long

    # Calculate image size in meters
    width_m, height_m = get_image_size(width_px, height_px, zoom_level, location_lat)

    # Calculate latitude boundaries
    lat_top = location_lat + height_m / 2 * lat_per_m
    lat_bottom = location_lat - height_m / 2 * lat_per_m

    # Calculate longitude boundaries
    long_left = location_long - width_m  / 2 * long_per_m
    long_right = location_long + width_m  / 2 * long_per_m

    # Return image boundaries as coordinates
    return lat_top, lat_bottom, long_left, long_right