import unittest
import pandas as pd
import os
import pytest
import numpy as np
import Exploratory_data_Analysis

class TestCSVLoading(unittest.TestCase):
    def setUp(self):
        self.file_path = 'heart_failure_clinical_records_dataset.csv'
    
    def test_file_exists(self):
        """Vérifie que le fichier CSV existe."""
        self.assertTrue(os.path.exists(self.file_path), "Le fichier CSV est introuvable.")
    
    def test_dataframe_loading(self):
        """Vérifie que le fichier CSV peut être chargé sans erreur."""
        try:
            df = pd.read_csv(self.file_path)
            self.assertFalse(df.empty, "Le DataFrame est vide.")
        except Exception as e:
            self.fail(f"Erreur lors du chargement du fichier CSV : {e}")
    
    def test_dataframe_structure(self):
        """Vérifie que les colonnes attendues sont présentes."""
        expected_columns = {
            'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
            'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium',
            'sex', 'smoking', 'time', 'DEATH_EVENT'
        }
        df = pd.read_csv(self.file_path)
        self.assertTrue(expected_columns.issubset(df.columns), "Les colonnes attendues ne sont pas toutes présentes.")

def test_detect_outliers_iqr():
    # Jeu de données sans valeurs aberrantes
    data_normal = pd.DataFrame({
        'A': [10, 12, 14, 11, 13, 15, 10],
        'B': [100, 102, 101, 99, 98, 103, 100]
    })
    assert all(len(v) == 0 for v in detect_outliers_iqr(data_normal).values())
    
    # Jeu de données avec des valeurs aberrantes
    data_outliers = pd.DataFrame({
        'A': [10, 12, 14, 11, 130, 15, 10],  # 130 est un outlier
        'B': [100, 102, 101, 99, 98, 300, 100]  # 300 est un outlier
    })
    outliers = detect_outliers_iqr(data_outliers)
    assert 130 in outliers['A'].values
    assert 300 in outliers['B'].values
    
    # Jeu de données avec une seule colonne
    data_single_column = pd.DataFrame({'A': [1, 2, 3, 4, 100]})  # 100 est un outlier
    outliers = detect_outliers_iqr(data_single_column)
    assert 100 in outliers['A'].values
    
    # Jeu de données contenant des NaN
    data_with_nan = pd.DataFrame({
        'A': [10, 12, np.nan, 11, 130, 15, 10],
        'B': [100, 102, 101, np.nan, 98, 300, 100]
    })
    outliers = detect_outliers_iqr(data_with_nan)
    assert 130 in outliers['A'].values
    assert 300 in outliers['B'].values


if __name__ == "__main__":
    unittest.main()
