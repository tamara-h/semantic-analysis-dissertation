# Sentiment Analysis

The postman layout for all the HTTP requests, with example responses can be found here: https://documenter.getpostman.com/view/5563343/S1EJXgL9

Gitlab repo: https://git-teaching.cs.bham.ac.uk/mod-ug-proj-2018/tlh537
VERSIONS USED:
Python 3.6
Node  8.10

# Folder Structure
## Diss
Directory contains dissertation files

## UI 
Contains UI for running the web application.  To run, node.js and Angular.js will need to be installed.
To start, navigate into directory and run "npm start", this will automatically install dependencies as well.
This runs on port 8000.

UI Design files are in the app directory.

## Sentiment Analysis

The final files for carrying out the investigations are in the main directory:

dataProcess.py - Formats each datasets with correct labels and for putting the variables into discrete categories
preProcessingandModel.py - Investigating N-Gram / Number of features / Model
overUnderTest.py - Investigating sampling methods
finalModel.py - Contains a copy of the final model
adjustModel.py  - Contains the adjustment model for dealing with the dependencies between the variables.
bagOfWords.py - Runs the lexicon analysis investigation

./data -  contains the datasets and output statistics for the hypothesis tests
./FinalGraphs -  contains the output graphs from the investigations
./oldFiles - contains initial trials into setting up the investigations.

venv folder has been removed so the project can be uploaded to canvas / git. You will need to install dependencies yourself.


## Sentiment Python Server

Contains two files
* train.py
*  server.py

train.py contains an instance of the final model as set out by the Sentiment Analysis directory finalModel.py and  hosts it on a very simple python server so that it can be accessed by the web app. 
venv folder has been removed so the project can be uploaded to canvas / git. You will need to install dependencies yourself.
To run the server, navigate into the directory and run "python server.py". This runs on port 8090

## Spotify Data

To start, navigate into directory and run "npm start", this will automatically install dependencies as well.  This will only run if the secrets.js file is accessible. Runs on port 8080.

## Statistical Analysis

Contains the R files for running the hypothesis tests set out in the report 
