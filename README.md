# Florida Residential Market Intelligence

Machine learning project for analyzing and modeling the Florida residential real estate market.

This project uses monthly county-level housing market data from Redfin. The goal is to study market conditions across Florida and gradually build a machine learning system that can later be presented through an interactive dashboard.

## Course Information

**Group 3**

**Courses:**  
- CAI 4505C — Artificial Intelligence  
- CAI 4510C — Machine Intelligence  

**Professor:** Martin Idahosa  
**Institution:** EnTec — Miami Dade College

### Team Members

- Deemond Allbritton
- Diana Gomez
- Ricardo Molina Gonzalez
- Rodrigo Betancourt

## Project Roadmap

### Part 1 — Proposal and Dataset Exploration ✅

The first part focused on understanding the dataset and creating a regression baseline.

Main work completed:

- Explored Florida residential real estate data
- Analyzed all 67 Florida counties
- Studied four property types
- Reviewed missing values, duplicates, and extreme values
- Created exploratory data analysis visualizations
- Used median sale price as the regression target
- Built Linear Regression baseline models
- Compared models with and without geographic information

The dataset contains monthly observations from January 2023 through August 2026. Each row represents one county, one property type, and one month.

### Part 2 — Classification Pipeline ✅

The second part extended the project into a supervised classification problem.

The goal was to predict whether the median sale price would increase the following month.

Work completed:

- Created a next-month binary classification target
- Used chronological validation to avoid future-data leakage
- Trained Logistic Regression and Decision Tree baseline models
- Added HistGradientBoosting as a stronger nonlinear model
- Evaluated Accuracy, Precision, Recall, F1-score, and ROC-AUC
- Used time-aware cross-validation
- Created confusion matrices and ROC curve visualizations
- Compared all three models on the same test period

Best test results came from HistGradientBoosting:

- Accuracy: 0.619
- F1-score: 0.615
- ROC-AUC: 0.670

The results show some predictive signal, but performance is still too modest for real property-level investment decisions.

Next, the project will explore 3-, 6-, and 12-month prediction horizons and more granular market and property-level data.

### Part 3 — Neural Models Integration

Planned extension using neural network models to explore more complex relationships in the housing market data.

### Part 4 — Time-Based Models

Planned extension focused on the temporal structure of the dataset and changes in market conditions over time.

### Part 5 — Integrated Market Intelligence Dashboard

Final goal:

- Combine results from the different project stages
- Present market trends and model outputs
- Create a clean and interactive dashboard for exploring Florida residential markets

## Dataset

Source: Redfin Housing Market Tracker

Current project scope:

- Florida only
- 67 counties
- January 2023 to August 2026
- Monthly data
- Condo/Co-op
- Townhouse
- Single Family Residential
- Multi-Family (2-4 Units)

## Repository Structure

```text
data/        Raw project dataset
src/         Python analysis and modeling scripts
outputs/     Generated figures and model results
reports/     Project reports
docs/        Project documentation and diagrams
