# Coding week_group 7 : Medical Decision Support Application
# PROJECT 1 : predicting heart failure risk with explainable ML SHAP
  This project aims to enhance clinical decision-making by providing an **explainable machine learning model** for predicting heart failure risk. By leveraging **SHAP (Shapley Additive Explanations)**, our approach ensures **transparency and interpretability**, allowing physicians to understand the reasoning behind each prediction. The system is designed to be **efficient, user-friendly, and professionally deployed**, featuring a **Streamlit or Flask interface** for seamless interaction. Additionally, best software development practices, including **CI/CD automation**, are incorporated to ensure a robust and scalable solution.
## Objectives  🚀:
- Develop a robust and accurate machine learning model to predict heart failure risk.
- Ensure model interpretability using SHAP (Shapley Additive Explanations) for transparent decision-making.
- Design an intuitive user interface with Streamlit or Flask for seamless interaction.
- Optimize memory usage to improve efficiency and scalability.
- Implement a CI/CD pipeline with GitHub Actions for automated testing and deployment.
- Apply prompt engineering techniques to enhance AI-driven workflows and documentation.
## Installation and Run Project
If you would like to run or contribute to this project, follow the steps below to set it up on your local machine.

### Fork or Clone the Repository
- Fork the repository: If you want to contribute to this project, you should fork it first. This will create a copy of the repository in your own GitHub account where you can make changes without affecting the original project.

- Go to the project repository.
- Click the Fork button at the top right of the page.
- Clone the repository: After forking the repository (or if you just want to run it without contributing), clone it to your local machine. You can do this with the following steps:

- Navigate to the repository page (after forking or directly cloning).

- Copy the repository URL: https://github.com/username/Coding-week_gr7_PREDICTING-HEART-FAILURE-RISK-WITH-EXPLAINABLE-ML-SHAP.git (replace username with your GitHub username).

- Open your terminal and navigate to the folder where you want to store the project.

- Run the following command to clone the repository:

```git clone https://github.com/username/Coding-week_gr7_PREDICTING-HEART-FAILURE-RISK-WITH-EXPLAINABLE-ML-SHAP.git```

### Set Up the Environment
It’s a good practice to set up a virtual environment to avoid conflicts with other Python packages you have installed. Follow these steps to create a virtual environment:

### Install Virtualenv (if not installed): If you don't have virtualenv installed, install it via pip:

```pip install virtualenv```

### Create a virtual environment:

In your terminal, navigate to the project directory (where requirements.txt is located) and create a virtual environment:

```virtualenv venv```

### Activate the virtual environment:

#### On macOS/Linux:

```source venv/bin/activate```

#### On Windows:

```venv\Scripts\activate```

### Install Project Dependencies
Now that your virtual environment is set up, install the required Python libraries and dependencies using the requirements.txt file.

Ensure you're in the project directory where the requirements.txt file is located.

Run the following command to install all dependencies:

```pip install -r requirements.txt```

### Prepare the Data
For this project, you will need to have the dataset used in the project. The dataset is heart_failure_clinical_records_dataset.csv.

Download the dataset: You can find the dataset by following the link below:

https://archive.ics.uci.edu/dataset/519/heart%2Bfailure%2Bclinical%2Brecords

Place the dataset: Place the downloaded CSV file in the project directory or the folder specified in the code.

### Running the Project
Once everything is set up, you can run the project.

Running the Python Script
You can execute it by running:

```python main.py```

### Troubleshooting
If you encounter issues during the setup or execution of the project, here are a few steps to resolve common problems:

Missing dependencies: Ensure all libraries are installed. You can check this by running:

```pip list```

This will list all installed packages. If you notice a missing package, you can install it using pip.

Dataset Issues: Ensure the dataset is properly downloaded and placed in the expected location. If the code references a specific path, make sure the dataset is placed there.


### Summary of Key Commands
To summarize, here are the key commands you need to follow to set up and run the project:

#### Clone the repository:

```git clone https://github.com/username/Coding-week_gr7_PREDICTING-HEART-FAILURE-RISK-WITH-EXPLAINABLE-ML-SHAP.git```

#### Create a virtual environment:

```virtualenv venv

source venv/bin/activate  # macOS/Linux

venv\Scripts\activate  # Windows 
```
#### Install dependencies:

```pip install -r requirements.txt ```

#### Run the Python script:

```python main.py```



