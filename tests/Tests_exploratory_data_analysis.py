import unittest
import pandas as pd
import os

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

if __name__ == "__main__":
    unittest.main()
