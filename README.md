# Marketing campaign

## Content

* [Summary](README.md#Summary)  
* [Project description](README.md#Project-description)  
* [Data and methods](README.md#Data-and-methods)                                
* [Project structure](README.md#Project-structure)                   


---

## Summary
It was predicted results of the direct marketing campaign: whether a client will subscribe for a term depsit. It was identified  most significant factors of positive campaign outcome. The customer may get a forecast of contact while marketing campaign by a given data using the [application on a web server](https://alex1iv-marketing-campaign-application-n8fexu.streamlit.app/). The sample page shown below.


<div align="center">
<img src="./figures/fig_streamlit.PNG" width="500"/></div>

<p align="center"> Microservice application page</p>

## Project description
A bank initiated an one-year marketing campaign among its clients to deposit money. After it was finished, the management decided to identfy success factors of the campaign using machine learning algorythm to increase its efficiency and decrease costs.

## Data and methods
:arrow_up:[ to content](README.md#Content)

It might be suggested to contact customers either in December or in summer, because the probability of success depends on month:

<div align="center">
<img src="./figures/fig_6.png" width="400" height="300"/> 
<img src="./figures/fig_7.png" width="400" height="300">  </div>


&emsp; &emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;**Probability of subscription by months** &emsp; &emsp;&emsp;&emsp;&emsp; &emsp; &emsp;&emsp;&emsp;&emsp; **Deposit outcome by age groups**


It was studied several models such as linear regression, random forest, decision tree, gradient boosting and stacking

The model shows good predictive capability.

<p align="center"> 
<img src="./figures/fig_15.png" width="400"> </p>
<p align="center"> ROC curve </p>

Most significant featurs are: poutcome_success, balance, age, pdays, contact by cellular phone, and others. It implies that elder people as well as young have propensity to subscribe for term depostit after summer. Most people in these groups have secondary education and married. Meanwhile single people with tertiary education have higher rate to subscribe than that of other social group.   

<p align="center"> 
<img src="./figures/fig_14.png" width="400" > </p>


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