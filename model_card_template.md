# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

For this project, I used RandomForestClassifer to train the model. 

## Intended Use

The intended use of this project and model is to predict if an individual earns more than 50K per year salary.

## Training Data

The data used for this project and its details can be found at https://archive.ics.uci.edu/dataset/20/census+income

The data includes the following columns of information on individuals:
    - age
    - workclass
    - education
    - marital-status
    - occupation
    - relationship
    - race
    - sex
    - capital-gain
    - capital-loss
    - hours-per-week
    - native-country
    - salary

## Evaluation Data

Data is evaluated by splitting the data into training and testing portions. I split the data with train_test_split and set a test size for 0.2 making the testing set 20% of the data.

## Metrics

Model Performance:

Precision: 0.7442 | Recall: 0.6334 | F1: 0.6843

_Please include the metrics used and your model's performance on those metrics._

## Ethical Considerations

The data in this project is public data and does not contain personally identifiable information(PII).

## Caveats and Recommendations

No recommendations to be made. This project is for learning purposes only. If this were applied in a real-life scenario and PII were present, I would recommend not showing that portion of the data in order to create integrity in the project.

This is an extra sentence made so I can do a pull request on VS Code where I am doing this project. This will test my GitHub Action.
