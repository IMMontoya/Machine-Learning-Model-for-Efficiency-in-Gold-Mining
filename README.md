# Machine Learning Model for Efficiency in Gold Mining

## Project Overview

### Introduction

The objective of this project is to develop a machine learning model to predict the amount of gold recovered from gold ore. This model aims to optimize the production process and eliminate unprofitable parameters for Zyfra, a company specializing in efficiency solutions for heavy industry. The dataset includes various parameters from the extraction and purification processes of gold ore.

### Data Description

The dataset contains information on different stages of the gold recovery process:

- **Rougher feed**: Raw material for flotation
- **Rougher additions**: Flotation reagents such as Xanthate, Sulphate, and Depressant
- **Rougher process**: The flotation process itself
- **Rougher tails**: Residues with low concentration of valuable metals
- **Cleaner process**: Two-stage purification resulting in final concentrate and new tails

- **Target Values**: `rougher.output.recovery` and `final.output.recovery`

The data is indexed by date and time, and some features in the training set are absent from the test set due to the timing of measurements or calculations.

## Data Preprocessing

### Validate rougher.output.recovery is calculated correctly

The function for calculating recover is defined as:  
$Recovery = C × (F − T) / F × (C − T)$  
where:

- C is the share of gold in the concentrate
- F is the share of gold in the feed
- T is the share of gold in the tails

The mean absolute error between the `rougher.output.recovery` column and caclulated recovery is calculated as 9.210911277458828e-15. The value is exceedingly small and we conclude we can trust the values in this column for training the models.

### Analyze features not available in the test set and handle them appropriately

In addition to the expected missing target values (`rougher.output.recovery` and `final.output.recovery`), the test data set is missing 32 other columns. These values are not known at the time of prediction, and should therefor be dropped from the training data set.

Columns unknown at the time of prediction:

<small>

| Column Name | Type |
|-------------------------------------------|------|
| final.output.concentrate_ag               | float64 |
| final.output.concentrate_au               | float64 |
| final.output.concentrate_pb               | float64 |
| final.output.concentrate_sol              | float64 |
| final.output.recovery                     | float64 |
| final.output.tail_ag                      | float64 |
| final.output.tail_au                      | float64 |
| final.output.tail_pb                      | float64 |
| final.output.tail_sol                     | float64 |
| primary_cleaner.output.concentrate_ag     | float64 |
| primary_cleaner.output.concentrate_au     | float64 |
| primary_cleaner.output.concentrate_pb     | float64 |
| primary_cleaner.output.concentrate_sol    | float64 |
| primary_cleaner.output.tail_ag            | float64 |
| primary_cleaner.output.tail_au            | float64 |
| primary_cleaner.output.tail_pb            | float64 |
| primary_cleaner.output.tail_sol           | float64 |
| rougher.calculation.au_pb_ratio           | float64 |
| rougher.calculation.floatbank10_sulfate_to_au_feed | float64 |
| rougher.calculation.floatbank11_sulfate_to_au_feed | float64 |
| rougher.calculation.sulfate_to_au_concentrate | float64 |
| rougher.output.concentrate_ag             | float64 |
| rougher.output.concentrate_au             | float64 |
| rougher.output.concentrate_pb             | float64 |
| rougher.output.concentrate_sol            | float64 |
| rougher.output.recovery                   | float64 |
| rougher.output.tail_ag                    | float64 |
| rougher.output.tail_au                    | float64 |
| rougher.output.tail_pb                    | float64 |
| rougher.output.tail_sol                   | float64 |
| secondary_cleaner.output.tail_ag          | float64 |
| secondary_cleaner.output.tail_au          | float64 |
| secondary_cleaner.output.tail_pb          | float64 |
| secondary_cleaner.output.tail_sol         | float64 |

</small>

### Handle missing values

There are significant percentages of missing values across both axises. Dropping nearly 35% of rows in the training data is clearly not a viable option. Dropping columns is also not an ideal solution, since we want as many features as possible for prediction (and we would still have plenty of rows with missing values). Thankfully, we know from the project description that "Parameters that are next to each other in terms of time are often similar.", therefor we can forward fill missing feature values. We will, however, have to drop rows with missing values in either target, which accounts for about 14% of the total data (we want to train our model only on actual results).

### Prepare the data for training

Based on the discoveries from exploring the data, the following steps are taken to preprocess the data for the task at hand:

1. Drop rows where either target is missing from full dataset.
2. Be sure full dataset is organized by date, then forward fill missing values (because "Parameters that are next to each other in terms of time are often similar."). Check no missing values.
3. Redefine the test df and train df as inner merge on date with full data set, drop necessary columns

After preprocessing the data, the share of train data to test data is within 2% of the original share and remains an appropriate split.

## Analyze the Data

### Explore concentrations of metals per purification stage

![AU Concentration per Stage](/images/au_conc.png)

![AG Concentration per Stage](/images/ag_conc.png)

![PB Concentration per Stage](/images/pb_conc.png)

#### Findings

- Gold (AU): As the refining process progress, the concentration of gold increases steadily. This is to be expected, as gold is the product in this process.
- Silver (AG): Silver concentration moves in the opposite direction of gold concentration, decreasing at every step of the process.
- Lead (PB): Generally remains the same throughout the process, increasing slightly from the rougher to primary stage, then remaining around the same in the final stage with more concentration around the mean.

### Compare feed particle size between the training and test set

![rougher.input.feed_size Distribution Comparison](/images/rougher_input_feed_dist.png)

![primary.input.feed_size Distribution](/images/primary_input_feed_dist.png)

#### Findings

Upon visual inspection of both particle feed size columns (`primary_cleaner.input.feed_size` and `rougher.input.feed_size`), it can be determined that the distributions between the train and test set do not differ significantly. This is most clearly illustrated by plotting the empirical cumulative distribution. Additional steps to balance these features is not necessary. 

### Explore the total substance concentration at each stage

![Total Substance Concentration Distribution](/images/total_conc_per_stage.png)

#### Findings

As the process progresses, the total concentration of all substances measured increases and condenses toward the mean value. This is to be expected, and generally reflects the pattern of the target substance (gold).

### Symetric Mean Absolute Percentage Error Functions

#### Define function to calculate the sMAPE value

Similar to MAE, but is expressed in relative values instead of absolute ones. It equally takes into account the scale of both the target and the prediction.

The calculation is represented as:

$ sMAPE = (1 / N) * Σ(|target_i - prediction_i| / ((target_i + prediction_i) / 2) * 100)$

where:

- $target_𝑖$ is the actual value for the i-th observation
- $prediction_𝑖$ is the predicted value for the i-th observation
- $𝑁$ is the total number of observations

#### Define function to calculate the final sMAPE value

The final function places 75% weight on the `final.output.recovery` target as is represented as:

$ Final\ sMAPE = 25\% * sMAPE(rougher) + 75\% * sMAPE(final) $

### Develop and Evaluate Models

Three models are trained, cross validated, and evaluated according to their final sMAPE scores.  

The Linear Regressor achieves a SMAPE Of 9.315

The Decision Tree Regressor is tested on a range of `max_depth = [None] + list(range(1, 21))` and achieves sMAPE: 8.1695 with max depth of 16

The Random Forrest Regressor is teston on a range of `max_depth = range(16, 27, 2)` and a range of `estimators = [100, 200]`. Random Forest Final sMAPE: 6.7610 with max depth of 26 and 200 estimators

### Test the Best Model

The best model was determined to be the Random Forest with max depth of 26 and 200 estimators. The model is then retrained on the entire training set and evaluated on its predictions for the test set. The final sMAPE score on the test set was 10.0919. However, a dummy regressor was also developed using the strategy of always predicting the mean values for the targets and scored similarly with a sMAPE of 10.2896.

The model predictions and mean predictions are plotted for each target value.

![Rougher Predictions](/images/rougher_preds.png)

![Final Predictions](/images/final_preds.png)

#### Findings

The best model with the best parameters performs only marginally better (sMAPE = 10.0919) compared to a baseline regression model predicting the mean values for each target (sMAPE = 10.2896). Plotting the model predictions and the dummy predictions illuminates this further. The model does a significantly better job of predicting the rougher.output.recovery target than the using the mean value prediction, we can see a more general adherence to the line of perfect prediction. However, when it comes to predicting the final.output.recovery target, the model's predictions cluster around the mean value much more so than the line of perfect prediction. Considering the final sMAPE calculation heavily favors the sMAPE score for the final.output.recovery target, this is explains the near match in final sMAPE scores between the best model and dummy model predictions.

## Conclusion

In this project, we developed a machine learning model to predict the amount of gold recovered from gold ore, aiming to optimize the production process for Zyfra. The process involved extensive data preprocessing and the evaluation of multiple models (Linear Regression, Decision Tree Regressor, Random Forest Regressor, and a baseline Dummy Regressor). The best model was a Random Forrest Regressor with a max depth of 26 and 200 estimators. This model achieved only marginal improvement over the baseline, highlighting areas for further enhancement.

### Recommendation

Based on the model’s performance, I recommend exploring more computationally expensive hyperparameters and other advanced machine learning models to capture complex relationships in the data. Additionally, incorporating domain-specific features and conducting detailed exploratory data analysis will further improve model accuracy and efficiency.