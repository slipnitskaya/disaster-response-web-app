# Disaster Response Web App
## Flask web app for classification of disaster messages

### Summary
Global losses due to disasters has been skyrocketing nowadays, plunging the economy into an ever deeper contraction. 
As a result, some fleeing danger might not have enough money for proper supplies. 
In order to effectively respond to these inevitable situations, there is a need for a software which allows to spot signs of potential danger, especially in crisis situations. 

Social media platforms such as Facebook and Twitter provide an exceptional opportunity to mine valuable information for aiding disaster response during emergency events.
Real-time categorization and classification of social media data have been intensively [utilized](https://onlinelibrary.wiley.com/doi/abs/10.1002/widm.1366) in disaster management for establishment of basic needs of the people affected by emerged incidences. 

One of the examples is the disaster response [data](https://www.figure-eight.com/dataset/combined-disaster-response-data/) from Figure Eight. 
The dataset contains 30k messages with the corresponding sentiment labels, and contains information about different natural disasters in Haiti, Chile, Pakistan and United States (e.g., a series of flood and earthquakes in 2010, a tornado in 2011, a hurricane Sandy in 2012). These messages are encoded into 36 different categories related to disaster response.

This project demonstrates how to build a neural language prerocessing (NLP) pipeline for an API that classifies disaster messages into relevant categories, that would potentially enable better emergency services responses. 

### Structure
Here you can get familiarized with the content more properly:

1. **ETL Pipeline:** 
`data/process_data.py` file provides the script to create an ETL pipeline that: 
 
    - Loads and combines the `messages` and `categories` datasets
    - Cleans the data
    - Exports it into an SQLite database 
    
2. **ML Pipeline:** 
`model/train_classifier.py` file provides the script to create an ML pipeline that: 

    - Splits the dataset into training and test sets
    - Builds text processing and machine learning pipelines 
    - Trains and tunes hyper-parameters of a multi-label classifier 
    - Displays performance metrics on the test set 
    - Exports the final model as a pickle file 
        
3. **Web Application:** 
`app/run.py` file provides the script to run a Flask web app that: 

    - Enables the user to enter a disaster message
    - Views the categories of the message
    - Visualizes the data-driven statistics 

**Libraries:** `requirements.txt` file provides libraries required for the successful start of the code.


### Instructions
In order to run the NLP pipeline and to start the Web App follow the instructions:

1. Run the following commands in the project's root directory to set up your database and model:

    - To run the ETL pipeline that cleans data and store in database
        `python data/process_data.py< -m data/disaster_messages.csv>< -c data/disaster_categories.csv>< -d data/disaster_responses.db>`
    - To run the ML pipeline that trains classifier and saves
        `python model/train_classifier.py< -d data/disaster_responses.db>< -M model/classifier.pkl>`

2. Run the following command in the app directory to start the web app:
    `python run.py< -d ../data/disaster_responses.db>< -M ../model/classifier.pkl>`

3. Go to http://0.0.0.0:3001/
