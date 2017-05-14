2017DMC
==============================

2017 Data Mining Cup Challenge - Revenue Forecast as a foundation for dynamic pricing

Machine Learning Pipeline 
------------
*Data Preparation and Model Training framework Set-up*
- Data Cleaning: run `1.0_clean.R`
- Feature Engineering: run `2.0_nolabel_features.R` -> `3.0_label_features.R` -> `4.0_merge_features`

*Training on the 1-63 days data and tuning on 63-77 days data*

- 1st-level models to predict order probability: run `5.1_h2o_gbm_1stLevel.R`, `5.2_h2o_glm_1stLevel.R`, `5.3_h2o_neural_network_1stLevel.R`, `5.4_h2o_rf_1stLevel.R` and `5.5_xgboost_1stLevel.R` separately. 
- 2nd-level models to predict the revenue: run `6.0_combine_1stLevelPreds.R` -> `6.1_h2o_glm_2ndLevel.R`(similar modeling script structure as last step)...

*Training on the hold-out set - last 15 days data*

- 3rd level model(blending): Combined the predictions from 2nd level models to fit a linear model on the `end92d_test.feather` to decide the weights for ensembling final models

*Retraining on all 92 days data and generate final predictions on the test set*

- Use the pre-configured 1st and 2nd model settings to retrain on the `end92d_train.feather` and then predict on the `end92d_test.feather` -> Save the final predictions from 2nd-level models
- Apply the 3rd level blending model on the final predictions

Notes: 
1. `3.1_ranef_features.R` takes long time to run so it can run independently to save the output files
2. `3.4_likelihood_features.R` includes the helper functions for generate likihood features used in `3.0_label_features.R`
3. To run the scripts properly, please make sure to set up the folder structure correctly as showed in the following section, especially for the `data` and `src` folders.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    |   ├── merge          <- Merge with other features
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
