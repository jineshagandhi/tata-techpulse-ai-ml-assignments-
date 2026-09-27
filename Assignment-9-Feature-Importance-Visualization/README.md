# Assignment 9: Feature Importance Visualization

This project visualizes and interprets which features drive predictions on two automotive datasets.  
Four methods are compared: impurity based importance, permutation importance, standardised linear coefficients, and partial dependence plots.

## Datasets

Both datasets are included in this folder.

- `automobile.csv` – UCI Automobile dataset (vehicle price, regression), also used in Assignment 4
- `ai4i2020.csv` – AI4I 2020 Predictive Maintenance dataset (machine failure, classification), also used in Assignment 5

## Key findings

**Vehicle price** (Random Forest, R² 0.93)

- Curb weight and engine size dominate. Together they account for over 70 percent of the impurity importance, and both methods agree on the top three features (Spearman rank correlation 0.86).
- Size features such as engine size, curb weight, horsepower and width are strongly correlated with each other, so the forest shares importance between them. A single importance score should not be read as the effect of that feature alone.
- The linear coefficients add the direction of the effect: engine size and horsepower raise the price, while fuel economy (city mpg) lowers it.

**Machine failure** (Random Forest, class weighted)

- Torque and rotational speed are the most important sensors, followed by air temperature and tool wear.
- Partial dependence plots show the failure risk sitting at the extremes of speed and torque, which points to mechanical power rather than either sensor alone.
- Adding physics based features (power, temperature difference, tool strain) raised F1 from 0.65 to 0.84 and recall from 0.50 to 0.75. These features are reused in the Assignment 10 MLOps pipeline.
