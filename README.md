# HR Salary Decider

Salary Decider is a Streamlit app that predicts a recommended salary for a single employee profile using a trained regression model. It is designed for salary estimation, compensation planning, and demoing an end-to-end machine learning deployment workflow.

## Use Case

The project helps estimate fair compensation based on common employee and job attributes such as role, experience level, location, employment type, work setting, and company size. It is useful for:

- Quick salary estimation for a candidate or employee profile
- Supporting internal compensation reviews
- Comparing salary recommendations across roles and locations
- Demonstrating machine learning inference in a user-facing app

## What the App Does

The app lets a user enter a profile and then:

- Loads a saved regression pipeline
- Generates a salary recommendation
- Shows the prediction in a simple dashboard
- Displays an input snapshot for review
- Exports the entered profile and prediction metadata

## How It Works

The model is trained on a salary dataset using PyCaret. During training, the data is cleaned, outliers are filtered, and the best regression model is selected and saved for later use.

When the app runs, it reads the saved model, accepts user input from the dashboard, and returns a salary estimate. The displayed result is converted to USD for presentation.

## Main Features

- Interactive Streamlit interface
- Single-profile salary prediction
- Cached model and dataset loading
- Downloadable CSV input snapshot
- Downloadable JSON metadata
- Docker-ready application structure
- GitHub Actions CI/CD for automated build and deployment

## Deployment

The repository is set up for container-based deployment. A GitHub Actions workflow builds the Docker image, pushes it to Azure Container Registry, and redeploys the application on an Azure VM when changes are pushed to the main branch.

## Steps to Use

1. Create and activate your Python environment.
2. Install dependencies:
	`pip install -r requirements.txt`
3. Train the regression model:
	`python scripts/modeltraining.py`
4. Start the Streamlit app:
	`streamlit run main.py`
5. Open the app in your browser (Streamlit prints the local URL in the terminal).
6. Fill in the employee profile fields and click **Get Recommended Salary**.
7. Review the salary recommendation and input snapshot.
8. Download the generated CSV/JSON exports if you need a record of the prediction.

## Notes

- A trained model must exist before the app can generate predictions.
- The app is intended for estimation and demonstration, not final compensation decisions.

## Live-Project-url
`https://www.project.pankaj-kumar-tech.me `
[click here](https://www.project.pankaj-kumar-tech.me)
