import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings("ignore")


def showliner(x, y, x_label, y_label):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    plt.ylabel(x_label, fontsize=13)
    plt.xlabel(y_label, fontsize=13)
    plt.show()


def showheatmap(df):
    corrmat = df.corr()
    plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=0.9, square=True)
    plt.show()


