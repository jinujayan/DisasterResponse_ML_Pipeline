# Disaster Response Pipeline Project
Classify request for help messages recevied during or after a disaster

## Table of contents
1. [Installation](#installation)
2. [Input](#input)
3. [Execute](#execute)
4. [Results](#results)

## Installation
With Python 3.6 installed, ensure the packages in the requirements.txt are available.<br>
__Note:__ Once spacy is successfully installed , please download the best-matching version of specific model for that spaCy installation by running the below command <br><br>
python -m spacy download en_core_web_sm

## Input
The input data is in the form of csv files, they are used by the preprocessing script process_data.py to generate a formatted data and stored as a database file.
This database is then used by the train_classifier to generate a model to be later used by webapp for performing predictions.

## Execute
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

__Note :__ If there is an error of helper_class module not found, then update the system path with current path to models directory in the file run.py<br>
sys.path.append('/home/workspace/models')

## Results
__Home Page__
<br>

![](https://github.com/jinujayan/DisasterResponse_ML_Pipeline/blob/master/images/HomePage_top.png)

__Category class distribution__
<br>
![](https://github.com/jinujayan/DisasterResponse_ML_Pipeline/blob/master/images/Barplot_group.png)
