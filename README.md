# Loan Default Risk Analysis

A end-to-end project analyzing loan default risk using Python for predictive modeling and Power BI for interactive reporting - combining descriptive dashboarding with a machine learning model to identify at-risk borrowers.

## Business Objective

Lenders need to identify high-risk borrowers before issuing loans to minimize default losses. This project analyzes historical loan data to answer:

- What factors most strongly predict loan default?
- Can we build a model to flag high-risk applicants before approval?
- How is risk distributed across the current loan portfolio?

## Dataset

- **Source:** [Loan Default Prediction Dataset](https://www.kaggle.com/datasets/nikhil1e9/loan-default/data) (255,347 records, 18 features)
- **Target variable:** `Default` (0 = repaid, 1 = defaulted)
- **Features:** Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines, InterestRate, LoanTerm, DTIRatio, Education, EmploymentType, MaritalStatus, HasMortgage, HasDependents, LoanPurpose, HasCoSigner

## Methodology

### 1. Data Preparation
- Verified no missing values across all 255,347 rows
- Dropped non-predictive identifiers (`LoanID`, `Loan Date`) from the modeling set (retained `LoanID` for joining predictions back to source data)
- One-hot encoded categorical features (Education, EmploymentType, MaritalStatus, HasMortgage, HasDependents, LoanPurpose, HasCoSigner)
- Engineered two additional features: `Loan_to_Income` (loan amount relative to income) and `Credit_per_Year_Employed` (credit score relative to employment tenure)

### 2. Modeling
Given the significant class imbalance (11.6% defaults), model selection prioritized **recall on the default class** - in a lending context, failing to catch an actual defaulter is costlier than a false alarm on a safe borrower.

Four algorithms were benchmarked, along with two optimization techniques:

| Approach | Recall (Default) | Precision (Default) | ROC-AUC |
|---|---|---|---|
| **Logistic Regression + feature engineering (final model)** | **0.70** | **0.23** | **0.76** |
| LightGBM | 0.68 | 0.23 | 0.76 |
| XGBoost | 0.63 | 0.23 | 0.74 |
| Random Forest | 0.01 | 0.66 | 0.73 |

Random Forest achieved the highest raw accuracy (88%) but caught almost none of the actual defaulters - a classic pitfall of optimizing for accuracy on imbalanced data. GridSearchCV hyperparameter tuning and SMOTE resampling were also tested but did not outperform the class-weighted Logistic Regression baseline.

### 3. Key Risk Drivers (Feature Importance)
Based on model coefficients, the strongest predictors of default were:
- **Age** (older borrowers = lower risk)
- **Interest Rate** (higher rate = higher risk)
- **Employment status** (unemployed/part-time = higher risk)
- **Months Employed** and **Income** (both inversely related to risk)

Notably, `CreditScore` and `DTIRatio` - often assumed to be dominant factors - did not rank in the top predictors, suggesting employment stability and demographics carry more independent signal in this dataset than traditional credit metrics alone.

### 4. Dashboard (Power BI)
A 4-page interactive dashboard was built:
1. **Business Overview** - portfolio-level KPIs and loan volume trends
2. **Applicant Demographics & Financial Profile** - median loan amount by credit score category, average loan amount by age group and marital status, loan volume by credit score bins, mortgage/dependents breakdown, and loan distribution by education type
3. **Financial Risk Metrics** - YOY loan amount and default trends (2013–2018), loan amount flow across credit score bins and marital status, and loan amount breakdown by income bracket and employment type
4. **Default Risk Model** - model performance (recall, ROC-AUC), risk score distribution, feature importance, and a high-risk loans watchlist for review

## Tools Used
Python (Pandas, Scikit-learn, XGBoost, LightGBM), Power BI, Jupyter Notebook

## Key Takeaways
- A well-configured simple model (Logistic Regression) outperformed more complex tree-based algorithms on this dataset - a reminder that model complexity doesn't guarantee better results, especially under class imbalance.
- Feature engineering (loan-to-income ratio) provided a small but genuine improvement where hyperparameter tuning and resampling did not.
- The chosen model prioritizes recall by design, meaning it will flag more false positives than a precision-optimized model would - a deliberate trade-off aligned with the business cost of missed defaults.

## Future Improvements
- Test additional engineered features and interaction terms
- Explore cost-sensitive threshold tuning to formally optimize for a business-defined cost matrix (cost of false negative vs. false positive)
- Deploy the model as a scoring API for real-time risk assessment