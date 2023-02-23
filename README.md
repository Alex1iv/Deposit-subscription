# Marketing campaign

## Content

* [Summary](README.md#Summary)  
* [Project description](README.md#Project-description)  
* [Data and methods](README.md#Data-and-methods)                                
* [Project structure](README.md#Project-structure)                   


---

## Summary
It was predicted the outcome of the marketing campaign using the the gradient boosting model. It was identified 15 most significant featurs, affecting the campaign outcome.

## Project description
A bank initiated an one-year marketing campaign among its clients to deposit money. After it was finished, the management decided to identfy success factors of the campaign using machine learning algorythm to increase its efficiency and decrease costs.

## Data and methods
:arrow_up:[ to content](README.md#Content)

It was studied several models such as linear regression, random forest, decision tree, gradient boosting and stacking
Most significant featurs are: 'poutcome_success', 'balance', 'pdays', and 13 other.

The model shows good predictive capability.

<p align="center"> 
<img src="/logs_and_figures/fig_14.PNG" width="450" height="350"> </p>

## Project structure

<details>
  <summary>display project structure </summary>

```Python
gesture_classification
├── .gitignore  
├── config
│   └── config.json           # configuration setings
├── data                      # data archive
│   └── campaign.zip
├── figures
│   ├── fig_1.png
.....
│   └── fig_streamlit.PNG
├── main.py
├── models                    # models and weights
│   ├── models_collection.py
│   ├── model_rf_opt.pkl
│   └── __ init __.py
├── notebooks                 # notebooks
│   └── Project_en.ipynb
├── project tree.ipynb
├── README.md                 # readme in English
└── utils                     # functions, variables, and data loaders
    ├── application.py
    ├── functions.py
    ├── reader_config.py
    ├── __ init __.py
    └── __pycache__
```
</details>


:arrow_up:[ to content](README.md#Content)