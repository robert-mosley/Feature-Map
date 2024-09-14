# Feature Analysis Tool

This Python script provides feature analysis using three methods:
- Feature Importances (ExtraTreesClassifier)
- Univariate Feature Selection (SelectKBest with chi-square)
- Heatmap of feature correlations

## Requirements

Install the required packages by running:
```bash
pip install -r requirements.txt

How to Run
To run the script, pass a CSV file as an argument containing your dataset. The script will guide you to analyze features using three different methods.

Example: python feature_analysis.py data.csv
