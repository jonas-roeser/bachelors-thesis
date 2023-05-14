# Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import geopandas as gpd
import geodatasets

# Define file path and file name
model_path = '../models/'
model_name = 'basicModel'
image_path = '../data/raw/satellite-images_new-york-city_2022/'

# Change font to LaTeX
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': [],

    # Fine-tune font-size
    'font.size': 12.0, # 10.0
    'figure.titlesize': 14.4, # 'large' (12.0)
    'figure.labelsize': 12.0, # 'large' (12.0)
    'axes.titlesize': 12.0, # 'large' (12.0)
    'axes.labelsize': 10.95, # 'medium' (10.0)
    'legend.title_fontsize': 10.95, # None (10.0)
    'legend.fontsize': 10.0, # 'medium' (10.0)
    'xtick.labelsize': 10.0, # 'medium' (10.0)
    'ytick.labelsize': 10.0 # 'medium' (10.0)
    })

def plot_history(model_name):
    # Loading data
    history = pd.read_csv(f'./models/{model_name}/history.csv',
        header=0,
        index_col='epoch')

    # Defining variables for plotting
    x = history.index + 1
    y1 = history.loss_train
    y2 = history.loss_val

    # Plotting data
    fig = plt.figure(figsize=(6.25,4), dpi=300)
    ax = plt.subplot(1, 1, 1)
    ax.plot(x, y1, linestyle='dashed', color='black', label='$RMSE_{train}$')
    ax.plot(x, y2, color='#c1272d', label='$RMSE_{val}$')

    # Defining axis limits
    max_x = max(x)
    min_x = min(x)
    max_y = max(max(y1), max(y2))
    min_y = min(min(y1), min(y2))

    # Setting axis limits
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y - 0.1 * (max_y - min_y), max_y + 0.1 * (max_y - min_y))

    # Creating labels and legend
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.legend(frameon=False)

    # Remove figure padding
    plt.tight_layout(pad=0.1) # pad=0 can lead to text being cut off
    left = max(fig.subplotpars.left, 1 - fig.subplotpars.right)
    spine_top_rel_height = ax.spines['top'].get_linewidth() / 72 / fig.get_size_inches()[1]
    fig.subplots_adjust( # does not work in .ipynb
        left=left,
        right=1 - left,
        top=1 - .5 * spine_top_rel_height if ax.get_title() == '' else fig.subplotpars.top)
    
    # Saving plot
    file_name = f'./models/{model_name}/history'
    plt.savefig(f'{file_name}.pdf', transparent=True, bbox_inches='tight')
    plt.close()

plot_history('basicModel')