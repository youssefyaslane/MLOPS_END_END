"""
Tests unitaires pour le module de preprocessing des données.
"""
import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import load_data, make_preprocessor, split_data


class TestDataPreprocessing:
    """Tests pour les fonctions de preprocessing."""
    
    @pytest.fixture
    def sample_data(self):
        """Crée un jeu de données de test."""
        data = {
            'customerID': ['C001', 'C002', 'C003', 'C004', 'C005'],
            'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'SeniorCitizen': [0, 1, 0, 0, 1],
            'Partner': ['Yes', 'No', 'Yes', 'No', 'Yes'],
            'Dependents': ['No', 'Yes', 'No', 'Yes', 'No'],
            'tenure': [12, 24, 6, 48, 36],
            'PhoneService': ['Yes', 'Yes', 'No', 'Yes', 'Yes'],
            'InternetService': ['Fiber optic', 'DSL', 'No', 'Fiber optic', 'DSL'],
            'Contract': ['Month-to-month', 'One year', 'Month-to-month', 'Two year', 'One year'],
            'MonthlyCharges': [50.0, 70.0, 30.0, 80.0, 60.0],
            'TotalCharges': [600.0, 1680.0, 180.0, 3840.0, 2160.0],
            'Churn': ['No', 'Yes', 'No', 'No', 'Yes']
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_csv_file(self, tmp_path, sample_data):
        """Crée un fichier CSV temporaire."""
        csv_file = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_file, index=False)
        return csv_file
    
    def test_load_data(self, sample_csv_file):
        """Test du chargement des données."""
        df = load_data(str(sample_csv_file))
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'Churn' in df.columns
        assert 'customerID' in df.columns
    
    def test_load_data_with_invalid_charges(self, tmp_path):
        """Test avec des TotalCharges invalides."""
        data = {
            'customerID': ['C001', 'C002'],
            'gender': ['Male', 'Female'],
            'SeniorCitizen': [0, 1],
            'Partner': ['Yes', 'No'],
            'Dependents': ['No', 'Yes'],
            'tenure': [12, 24],
            'PhoneService': ['Yes', 'Yes'],
            'InternetService': ['Fiber optic', 'DSL'],
            'Contract': ['Month-to-month', 'One year'],
            'MonthlyCharges': [50.0, 70.0],
            'TotalCharges': ['600.0', ' '],  # Une valeur invalide
            'Churn': ['No', 'Yes']
        }
        df = pd.DataFrame(data)
        csv_file = tmp_path / "invalid_data.csv"
        df.to_csv(csv_file, index=False)
        
        result_df = load_data(str(csv_file))
        
        # Vérifie que la ligne avec valeur invalide est supprimée
        assert len(result_df) == 1
        assert result_df['customerID'].iloc[0] == 'C001'
    
    def test_make_preprocessor(self, sample_data):
        """Test de la création du preprocessor."""
        X, y, preprocessor = make_preprocessor(sample_data)
        
        # Vérifications de X
        assert isinstance(X, pd.DataFrame)
        assert 'customerID' not in X.columns
        assert 'Churn' not in X.columns
        assert len(X) == len(sample_data)
        
        # Vérifications de y
        assert isinstance(y, pd.Series)
        assert len(y) == len(sample_data)
        assert set(y.unique()).issubset({0, 1})
        
        # Vérifications du preprocessor
        assert preprocessor is not None
        assert hasattr(preprocessor, 'fit_transform')
    
    def test_preprocessor_transform(self, sample_data):
        """Test de la transformation des données."""
        X, y, preprocessor = make_preprocessor(sample_data)
        
        # Fit et transform
        X_transformed = preprocessor.fit_transform(X)
        
        # Vérifications
        assert X_transformed.shape[0] == len(X)
        assert X_transformed.shape[1] > len(X.columns)  # One-hot encoding augmente les colonnes
        assert not np.isnan(X_transformed).any()  # Pas de valeurs manquantes
    
    def test_split_data(self, sample_data):
        """Test du split des données."""
        X, y, _ = make_preprocessor(sample_data)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
        
        # Vérifications des tailles
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        
        # Vérifications du ratio
        test_ratio = len(X_test) / len(X)
        assert 0.15 <= test_ratio <= 0.25  # Approximativement 20%
        
        # Vérifications de la stratification
        train_churn_rate = y_train.sum() / len(y_train)
        test_churn_rate = y_test.sum() / len(y_test)
        overall_churn_rate = y.sum() / len(y)
        
        # Les taux de churn doivent être similaires
        assert abs(train_churn_rate - overall_churn_rate) < 0.2
        assert abs(test_churn_rate - overall_churn_rate) < 0.2
    
    def test_split_data_reproducibility(self, sample_data):
        """Test de la reproductibilité du split."""
        X, y, _ = make_preprocessor(sample_data)
        
        # Premier split
        X_train1, X_test1, y_train1, y_test1 = split_data(X, y, random_state=42)
        
        # Deuxième split avec même seed
        X_train2, X_test2, y_train2, y_test2 = split_data(X, y, random_state=42)
        
        # Les résultats doivent être identiques
        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_frame_equal(X_test1, X_test2)
        pd.testing.assert_series_equal(y_train1, y_train2)
        pd.testing.assert_series_equal(y_test1, y_test2)


class TestDataValidation:
    """Tests de validation des données."""
    
    def test_missing_target_column(self):
        """Test avec colonne cible manquante."""
        data = {
            'customerID': ['C001', 'C002'],
            'gender': ['Male', 'Female'],
            'tenure': [12, 24],
        }
        df = pd.DataFrame(data)
        
        with pytest.raises(KeyError):
            make_preprocessor(df)
    
    def test_empty_dataframe(self):
        """Test avec DataFrame vide."""
        df = pd.DataFrame()
        
        with pytest.raises(Exception):
            make_preprocessor(df)
    
    def test_single_row(self):
        """Test avec une seule ligne."""
        data = {
            'customerID': ['C001'],
            'gender': ['Male'],
            'SeniorCitizen': [0],
            'Partner': ['Yes'],
            'Dependents': ['No'],
            'tenure': [12],
            'PhoneService': ['Yes'],
            'InternetService': ['Fiber optic'],
            'Contract': ['Month-to-month'],
            'MonthlyCharges': [50.0],
            'TotalCharges': [600.0],
            'Churn': ['No']
        }
        df = pd.DataFrame(data)
        
        X, y, preprocessor = make_preprocessor(df)
        
        assert len(X) == 1
        assert len(y) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])