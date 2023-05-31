
# Student Score Prediction

This is an end to end machine learning project to predict the score of the students based on parental level of education, ethnicity group and other several factors.



## Repository Structure
1.  Artifacts: In this folder preprocessed model pkl file, trained mode pkl file, raw data csv file, train data csv file, test data csv file are saved.
2.  Logs: These folder is dedicated for saving logging information.
3.  Notebook: In this folder basic eda and model training are perfomed.
4.  src: This is the source foldr which contains the following sub folders.
    *   Components: In this folder data collection, data transformation, and model training files are stored.
    *   Pipeline: In this folder predict pipeline and train pipeline files are stored.
    *   Exception.py: This file is responsible for exception handling.
    *   logger.py: This file is responsible for logging the necessary information.
    *   utils.py: This file is responsible for reuse of the code for repeatable tasks.
5.  Templates: In this folder we design the home page and prediction page.
6.  app.yaml: for web deployment using google cloud platform.
7.  main.py: to create web application using flask framework.
8.  requirements.txt: list of necessary dependencies to run the application.
9.  setup.py: code to install libraries listed in requirements.txt

## Installation
1.  Clone the github repository by using the following link.

https://github.com/SaiSrinivas1997/Student_Score_Prediction.git

2. Install the necessary dependencies usung the following command.
```bash
pip install -r requirements.txt
```
3. To run the application locally using flask execute the following command in the terminal and navigate to 127.0.0.1 in your browser.
```bash
python main.py
```
4. To deploy the application in google cloud platform edit your python version in app.yaml file and follow below procedure.
* Execute the below command and selct the configuration, mail and project.
    ```bash
    gcloud init
    ```
* Now deploy the code using the following command and select the region.
    ```bash
    gcloud app deploy app.yaml --project selected_project_name
    ```
* Navigate to the website by executing the following command.
    ```bash
    gcloud app browse
    ```
## Usage
when you navigated to the website add "/predictdata" at the end of the url to route to prediction page and select the one of the following options for the fields and click predict button to predict the score.

*  Gender : Male/Female.
*  Race or Ethnicity : Group A to E.
*  Parental Level of Education : associate degree, bachelor degree, master degree, some college, some high school.
*  Lunch type : standard, free/reduced.
*  Test preparation score : none, completed.
*  writing score : 0 to 100
*  Reading score : 0 to 100

![Screenshot](https://github.com/SaiSrinivas1997/Student_Score_Prediction/blob/4695c0aaa5b8bd982ad27ec19557524bf1f3e0c9/img/Screenshot%20(14).png)





