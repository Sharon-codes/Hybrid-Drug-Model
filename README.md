
🧬 Hybrid Machine Learning Framework for Drug Efficacy Prediction in Breast Cancer

Authors: Sharon Melhi¹, Dhanashree Bhamare²  
¹Amity University Bengaluru  
²Somaiya Vidyavihar University, Mumbai

📌 Overview

This project proposes a hybrid machine learning-based framework that predicts the LN_IC50 inhibitory concentration of cancer drugs to support personalized treatment strategies in breast cancer. By integrating pharmacogenomic, clinical, and molecular data, the model aims to assist in dose optimization, reduce empirical prescriptions, and foster explainability in precision oncology.

🚀 Features include:
- 🔍 Prediction using XGBoost and Deep Neural Networks
- 🧠 Ensemble learning for robust performance
- 🧾 SHAP analysis for interpretability
- 📊 Visual analytics to support clinical decisions

🗂️ Project Structure

- `model5.py`: Complete pipeline (data loading ➜ preprocessing ➜ training ➜ evaluation ➜ visualization)
- `Abstract.docx`: Detailed research abstract
- Output plots:
  - 📈 `ln_ic50_by_cancer.png`
  - 📉 `ln_ic50_vs_auc.png`
  - 🧪 `ln_ic50_by_stage.png`
  - 🧠 `shap_importance.png`

🧰 Requirements

Install the necessary packages with:

```bash
pip install pandas numpy scikit-learn xgboost tensorflow shap matplotlib seaborn openpyxl
````

🗃️ Data Sources

* Pharmacogenomics: `GDSC2_fitted_dose_response_27Oct23.xlsx`
* Clinical Data: `clinical.tsv` (TCGA-BRCA subset)

🗂️ Place these in the `data/` directory as specified in the code.

▶️ Running the Project

```bash
python model5.py
```

📊 Output Highlights

* ✅ RMSE: \~0.21
* ✅ R² Score: \~0.89
* 🔁 Cross-Validated RMSE: Printed in console

Key Insights 🧠

* Drug resistance differs by subtype
* Higher dose needed in Stage IV (\~10% more)
* Key predictors: AUC, tumor stage, patient age

🔮 Future Enhancements

* Real-time patient data integration
* Cross-cancer type application
* Web dashboard for clinicians

🙏 Acknowledgements

Thanks to GDSC and TCGA for datasets
