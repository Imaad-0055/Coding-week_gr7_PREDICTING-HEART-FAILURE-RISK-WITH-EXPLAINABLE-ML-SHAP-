import pytest
import pandas as pd
import numpy as np
from scipy.stats import boxcox
import nbimporter
import nbformat
import sys
import os
notebook_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'notebook'))
sys.path.append(notebook_path)
from Notebooks import Feature_Engineering

def test_transform_input_data():
    # Données fictives avant transformation
    data_before_trans = pd.DataFrame({
        'platelets': [150000, 200000, 250000, 300000, 350000]
    })
    
    # Données fictives après transformation (simulées avec standardisation)
    data_after_transf = pd.DataFrame({
        'age': [0.1, -0.5, 1.2, -1.1, 0.3],
        'serum_creatinine': [0.4, -0.2, 1.5, -1.3, 0.1]
    })
    
    # Données d'entrée
    input_data = pd.DataFrame({
        'age': [60, 65, 70],
        'serum_creatinine': [1.0, 1.2, 1.5],
        'platelets': [100000, 400000, 500000],
        'sex': [1, 0, 1]  # Autre variable non transformée
    })
    
    # Définition des paramètres
    cont_features = ['age', 'serum_creatinine']
    other_features = ['sex']
    boxcox_lambdas = {'age': 0.5, 'serum_creatinine': -0.2}  # Exemples de valeurs lambda
    
    # Exécution de la transformation
    transformed_data = transform_input_data(input_data, data_before_trans, data_after_transf, 
                                            cont_features, other_features, boxcox_lambdas)
    
    # Vérifications
    assert 'age' in transformed_data.columns
    assert 'serum_creatinine' in transformed_data.columns
    assert 'sex' in transformed_data.columns
    assert transformed_data['sex'].equals(input_data['sex']), "La colonne 'sex' ne doit pas être modifiée."
    
    # Vérification de la Winsorization
    assert transformed_data['platelets'].min() >= np.percentile(data_before_trans['platelets'], 5), "Winsorization incorrecte (min)."
    assert transformed_data['platelets'].max() <= np.percentile(data_before_trans['platelets'], 95), "Winsorization incorrecte (max)."
    
    # Vérification de la standardisation
    assert np.isclose(transformed_data.mean().drop('sex'), 0, atol=1), "La moyenne après standardisation doit être proche de 0."
    assert np.isclose(transformed_data.std().drop('sex'), 1, atol=1), "L'écart-type après standardisation doit être proche de 1."
