import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import torch
from mlex import DataReader
from pcpe_utils import get_pcpe_dtype_dict, pcpe_preprocessing_read_func
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from mlex import PreProcessingTransformer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlex import MarkovAnalyzer, MarkovAnalyzerPlotter
from pathlib import Path


script_path = Path(__file__).parent.absolute().joinpath('markov_results')

#sequences_compositions = ['temporal', 'account', 'individual']
sequences_compositions = ['account', 'individual']
#sequence_column_dict = {'temporal': None, 'account': 'CONTA_TITULAR', 'individual': 'CPF_CNPJ_TITULAR'}
sequence_column_dict = {'account': 'CONTA_TITULAR', 'individual': 'CPF_CNPJ_TITULAR'}

path = r'/data/pcpe/pcpe_04.csv'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
target_column = 'I-d'
filter_data = {}

reader = DataReader(path, target_columns=[target_column], filter_dict=filter_data, dtype_dict=get_pcpe_dtype_dict(), preprocessing_func=pcpe_preprocessing_read_func)
X, y = reader.get_X_y()



for sequence_composition in sequences_compositions:
    
    column_name = sequence_column_dict[sequence_composition]
    array_data = X[column_name]
    Markov_Analyzer = MarkovAnalyzer()
    frequencies, probability_matrix = Markov_Analyzer.analyze(array_data, column_name, script_path)
    
    Markov_Analyzer_Plotter = MarkovAnalyzerPlotter()
    Markov_Analyzer_Plotter.plot_markov_chain(probability_matrix, column_name, script_path)