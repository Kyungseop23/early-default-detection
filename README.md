# Early Credit Default Detection from Application Stage Data
## Balancing Customer Experience and Financial Risk

## Key Results
| Model | Precision | Recall | F1 Score |
| ----- | ----- | ----- | ----- |
| Baseline | 24.07% | 40.66% | 30.24% |
| Merged + Adjusted | 24.32% | 45.90% | 31.80% |
- **12.89% improvement in recall** with no loss in both precision and F1 score.
- Improved early default detection capability as a first stage filter while maintaining overall model stability.

## Project Overview
### Real World Interest: Balancing Frictionless Customer Experience with Financial Security
The rapidly growing online lending industry led by Financial Technology (FinTech) companies faces a fundamental trade off between frictionless customer experience and financial risk management.
- **Customer Expectation:** Clients expect quick credit decisions with minimal friction, avoiding manual document submission or additional verification steps.
- **Financial Risk:** Faster and simplified processes often increase the risk of default, leading to potential financial losses and reduced trust.

### Key Research Question
How can machine learning be used to effectively detect potential credit defaults using only client application data at the first stage filter, enhancing client experience while maintaining low financial risk?

### Goal
To build a fast and cost efficient prescreening modle that identifies high risk clients early before any further documentation request, manual review, or external bureau checks.


## Dataset Sources
[Kaggle: Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
| Dataset | Description | Key Usage |
| ----- | ----- | ------ |
| Recent Applications | Client data from the current loan application | Used for initial model training |
| Past Applications | Past records of previous loan application | Used for feature enrichment and identifying risk patterns more precisely |


## Tech Stack
- **Data Preprocessing:** Panday, Numpy
- **Statistical Analysis:** SciPy (Chi-square test)
- **Modeling & Evaluation:** Scikit-learn, RandomForest, LightGBM, Imbalanced-learn
- **Visualization:** Matplotlib, Seaborn
- **Development Environmen:** Google Colab

## Full Analysis Notebook
[Early Default Application](./notebooks/early_default_application.ipynb) - Main Notebook
[Model Selection and Baseline Results](./notebooks/[early_default_application]_model_selection_and_baseline_results.ipynb) - Model Selection & Baseline Results

## Analysis Workflow
### 1. Recent Application: Preprocessing & EDA
- Checked target imbalance, misssingness patterns.
- Conducted chi-square test to analyze binary features.
- Handled missing values, outliers, and highly correlated features.
- Created business meaningful features and dropped less relevant features using feature importance.

### 2. Model & Method Selection
- Compared **Random Forest** and **LightGBM** with various imbalance handling methods (class weight, is unbalance, SMOTE, RandomOverSampler, RandomUnderSampler).
- Selected **LightGBM with RandomOverSampler (sampling_strategy = 0.5)** as the baseline model
    - LightGBM was faster and achieved higher recall and F1 than Random Forest, consistent with the project's goal of early default detection.

### 3. Recent Application Model Training & Evaluation
- Used Stratified K-Fold (n=5) cross validation.
- Performed feature selection using feature importance threshold.
- Evaluated performance using precision, recall, and F1 score.

### 4. Merged Application Data (Recent + Past Application Data)
- Processed past application data with the same rules used for recent application data.
- Aggregated past records by client and merged with recent application data.

### 5.Merged Application Model Training & Evaluation
- Trained the **same LightGBM + RandomOversampler pipeline** for consistency.
- Fine tuned thresholds to optimize recall while maintaining F1 score.


## Key Insights & Discussion
### Insights
- Past application data captured meaningful behavioral patterns, indicating that it can enhance early default detection even without relying on external bureau data.
- Balanced oversampling (0.5 ratio) improved recall while keeping F1 stable, showing that proper resampling enhances default detection without overfitting.
- The model achieved resaonably strong recall using only application data, enabling real time prescreening to reduce manual reviews or external bureau data check while keeping a frictionless customer experience.

### Challenges
#### Data Quality and Reliability
- Application data includes many missing or anonymized fields (due to privacy constraints), which can reduce modeling reliability.
- Some information is also self reported by clients, so it may not always be consistent or accurate.

#### Balancing Evaluation Metrics
- Depending on how evaluation priorities are defined, the overall process, including model selection and thresholding, can vary.

#### Biases and Ethical Concerns
- If the data reflects social or historical bias, the model may unintentially reproduce or amplify it.


## Future Improvements
### Upgrade Application Form Design
- Use modeling insights to refine application questions and include the most predictive fields.
### Real Costs & Business Impact
- Understanding the financial cost of false negative (missed default) and false positives (rejected good clients) will enable more precise optimization of the precision recall trade off.
### Continuous Monitoring
- Regularly monitor model fairness to ensure long term reliability.
- update and retrain the model periodically to identify and mitigate emerging biases over time.
