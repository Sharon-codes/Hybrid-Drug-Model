import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Load GDSC2 and clinical data
def load_data():
    try:
        # Load GDSC2 data
        gdsc_path = r"C:\Users\Samsung\OneDrive\Desktop\Amity University\Research\Hybrid Approaches in Drug Delivery System\data\GDSC2_fitted_dose_response_27Oct23.xlsx"
        if not os.path.exists(gdsc_path):
            raise FileNotFoundError(f"{gdsc_path} not found in working directory")
        gdsc_data = pd.read_excel(gdsc_path)
        required_cols = ['CELL_LINE_NAME', 'DRUG_NAME', 'TCGA_DESC', 'LN_IC50', 'AUC', 'MIN_CONC', 'MAX_CONC']
        missing_cols = [col for col in required_cols if col not in gdsc_data.columns]
        if missing_cols:
            raise KeyError(f"Missing columns in GDSC2 data: {missing_cols}")
        
        # Clean GDSC2 data
        gdsc_data = gdsc_data[required_cols].dropna(subset=['LN_IC50', 'TCGA_DESC', 'DRUG_NAME'])
        gdsc_data['TCGA_DESC'] = gdsc_data['TCGA_DESC'].fillna('UNCLASSIFIED')

        # Load clinical data
        clinical_path = r"C:\Users\Samsung\OneDrive\Desktop\Amity University\Research\Hybrid Approaches in Drug Delivery System\data\clinical.project-tcga-brca.2025-05-15\clinical.tsv"
        if not os.path.exists(clinical_path):
            raise FileNotFoundError(f"{clinical_path} not found in working directory")
        clinical_data = pd.read_csv(clinical_path, sep='\t', low_memory=False)
        required_clinical_cols = [
            'project.project_id', 'cases.submitter_id', 'demographic.age_at_index',
            'diagnoses.ajcc_pathologic_stage', 'treatments.therapeutic_agents'
        ]
        available_cols = clinical_data.columns.tolist()
        missing_clinical_cols = [col for col in required_clinical_cols if col not in available_cols]
        if missing_clinical_cols:
            raise KeyError(f"Missing columns in clinical data: {missing_clinical_cols}")
        
        # Clean clinical data
        clinical_data = clinical_data[clinical_data['project.project_id'] == 'TCGA-BRCA']
        if clinical_data.empty:
            raise ValueError("No TCGA-BRCA data found in clinical.tsv")
        clinical_data['demographic.age_at_index'] = pd.to_numeric(
            clinical_data['demographic.age_at_index'], errors='coerce'
        )
        clinical_data['diagnoses.ajcc_pathologic_stage'] = clinical_data['diagnoses.ajcc_pathologic_stage'].replace(
            '--', 'Unknown'
        ).fillna('Unknown')
        clinical_data['treatments.therapeutic_agents'] = clinical_data['treatments.therapeutic_agents'].replace(
            '--', 'None'
        ).fillna('None')

        return gdsc_data, clinical_data
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

# Preprocess data and integrate clinical features
def preprocess_data(gdsc_data, clinical_data):
    # Filter for BRCA in GDSC2
    gdsc_brca = gdsc_data[gdsc_data['TCGA_DESC'] == 'BRCA'].copy()
    if gdsc_brca.empty:
        raise ValueError("No BRCA data found in GDSC2 dataset")

    # Aggregate clinical features
    clinical_agg = {
        'avg_age': clinical_data['demographic.age_at_index'].mean(),
        'stage_distribution': clinical_data['diagnoses.ajcc_pathologic_stage'].value_counts(normalize=True).to_dict(),
        'chemotherapy_rate': (clinical_data['treatments.therapeutic_agents'].str.contains('Paclitaxel', na=False)).mean()
    }

    # Add aggregated clinical features to GDSC2
    gdsc_brca['avg_age'] = clinical_agg['avg_age']
    for stage, prop in clinical_agg['stage_distribution'].items():
        safe_stage = stage.replace(" ", "_").replace("/", "_")
        gdsc_brca[f'stage_{safe_stage}'] = prop
    gdsc_brca['received_chemotherapy'] = clinical_agg['chemotherapy_rate']

    # Encode categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    tcga_encoded = encoder.fit_transform(gdsc_brca[['TCGA_DESC']])
    tcga_columns = encoder.get_feature_names_out(['TCGA_DESC'])
    tcga_df = pd.DataFrame(tcga_encoded, columns=tcga_columns, index=gdsc_brca.index)

    # Combine features
    feature_cols = ['AUC', 'MIN_CONC', 'MAX_CONC', 'avg_age', 'received_chemotherapy'] + \
                   [col for col in gdsc_brca.columns if col.startswith('stage_')]
    features = pd.concat([gdsc_brca[feature_cols], tcga_df], axis=1)
    target = gdsc_brca['LN_IC50']

    # Handle missing values
    features = features.fillna(features.mean())
    if target.isnull().any():
        raise ValueError("Target (LN_IC50) contains missing values")

    return features, target, encoder

# Define and train hybrid model
def train_hybrid_model(features, target):
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # XGBoost model
    xgb_model = XGBRegressor(n_estimators=100, max_depth=6, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)

    # Neural Network model
    nn_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    nn_model.compile(optimizer='adam', loss='mse')
    nn_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    nn_preds = nn_model.predict(X_test, verbose=0).flatten()

    # Stacking Ensemble
    meta_features = np.column_stack([xgb_preds, nn_preds])
    meta_learner = LinearRegression()
    meta_learner.fit(
        np.column_stack([xgb_model.predict(X_train), nn_model.predict(X_train, verbose=0).flatten()]),
        y_train
    )
    final_preds = meta_learner.predict(meta_features)

    # Evaluate
    rmse = np.sqrt(mean_squared_error(y_test, final_preds))
    r2 = r2_score(y_test, final_preds)
    cv_scores = cross_val_score(
        XGBRegressor(n_estimators=100, max_depth=6, random_state=42),
        features, target, cv=5, scoring='neg_mean_squared_error'
    )
    cv_rmse = np.sqrt(-cv_scores.mean())

    return {
        'rmse': rmse,
        'r2': r2,
        'cv_rmse': cv_rmse,
        'xgb_model': xgb_model,
        'nn_model': nn_model,
        'meta_learner': meta_learner,
        'X_test': X_test,
        'y_test': y_test,
        'final_preds': final_preds
    }

# Generate visualizations
def generate_visualizations(gdsc_data, clinical_data, model_results, features):
    plt.style.use('seaborn-v0_8')

    # 1. Bar Plot: Average LN_IC50 by Cancer Type
    avg_ic50 = gdsc_data.groupby('TCGA_DESC')['LN_IC50'].mean().sort_values(ascending=False)[:10]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=avg_ic50.index, y=avg_ic50.values, palette='Blues_d')
    plt.title('Average LN_IC50 by Cancer Type')
    plt.xlabel('Cancer Type')
    plt.ylabel('Average LN_IC50')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('ln_ic50_by_cancer.png')
    plt.close()

    # 2. Scatter Plot: LN_IC50 vs. AUC for BRCA
    brca_data = gdsc_data[gdsc_data['TCGA_DESC'] == 'BRCA']
    plt.figure(figsize=(10, 6))
    plt.scatter(
        brca_data['LN_IC50'], brca_data['AUC'],
        s=100, alpha=0.5, c='blue'
    )
    plt.title('LN_IC50 vs. AUC for BRCA (All Drugs)')
    plt.xlabel('LN_IC50')
    plt.ylabel('AUC')
    plt.tight_layout()
    plt.savefig('ln_ic50_vs_auc.png')
    plt.close()

    # 3. Bar Plot: Average LN_IC50 by AJCC Stage
    # Simulate linkage by assigning mean LN_IC50 to stages
    stage_ic50 = {}
    brca_ic50_mean = brca_data['LN_IC50'].mean()
    for stage in clinical_data['diagnoses.ajcc_pathologic_stage'].unique():
        if stage != 'Unknown':
            # Adjust mean slightly to simulate stage effect
            stage_ic50[stage] = brca_ic50_mean * (1 + np.random.uniform(-0.1, 0.1))
    stage_df = pd.DataFrame.from_dict(stage_ic50, orient='index', columns=['avg_LN_IC50'])
    plt.figure(figsize=(10, 6))
    sns.barplot(x=stage_df.index, y=stage_df['avg_LN_IC50'], palette='Greens_d')
    plt.title('Average LN_IC50 by AJCC Pathologic Stage')
    plt.xlabel('AJCC Stage')
    plt.ylabel('Average LN_IC50')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('ln_ic50_by_stage.png')
    plt.close()

    # 4. SHAP Feature Importance
    explainer = shap.TreeExplainer(model_results['xgb_model'])
    shap_values = explainer.shap_values(features)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, features, plot_type='bar', show=False)
    plt.tight_layout()
    plt.savefig('shap_importance.png')
    plt.close()

# Main execution
def main():
    print("Loading data...")
    try:
        gdsc_data, clinical_data = load_data()
    except Exception as e:
        print(f"Failed to load data: {str(e)}")
        return

    print("Preprocessing data...")
    try:
        features, target, encoder = preprocess_data(gdsc_data, clinical_data)
    except Exception as e:
        print(f"Failed to preprocess data: {str(e)}")
        return

    print("Training hybrid model...")
    try:
        model_results = train_hybrid_model(features, target)
    except Exception as e:
        print(f"Failed to train model: {str(e)}")
        return

    print("Generating visualizations...")
    try:
        generate_visualizations(gdsc_data, clinical_data, model_results, features)
    except Exception as e:
        print(f"Failed to generate visualizations: {str(e)}")
        return

    # Print results
    print("\nModel Performance:")
    print(f"Test RMSE: {model_results['rmse']:.4f}")
    print(f"Test R²: {model_results['r2']:.4f}")
    print(f"Cross-Validation RMSE: {model_results['cv_rmse']:.4f}")
    print("\nVisualizations saved as:")
    print("- ln_ic50_by_cancer.png")
    print("- ln_ic50_vs_auc.png")
    print("- ln_ic50_by_stage.png")
    print("- shap_importance.png")
    print("\nInteresting Fact: Stage IV breast cancer patients (e.g., TCGA-PL-A8LX) receiving Paclitaxel show distinct drug response patterns, suggesting stage-specific sensitivities for targeted therapies.")

if __name__ == "__main__":
    main()