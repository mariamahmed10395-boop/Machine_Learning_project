# Finding Donors for CharityML

## Overview

This project builds a supervised machine learning pipeline to predict whether an individual earns more than $50,000 per year, based on U.S. census data. The goal is to help CharityML identify the most likely donors and optimize their outreach efforts.

The project goes beyond the base Udacity template by integrating MLflow for experiment tracking and a Tkinter GUI for live predictions.

## Installation

This project requires Python 3.x and the following libraries:

- NumPy
- Pandas
- Matplotlib
- scikit-learn
- Tkinter (built into Python standard library)

Install dependencies with:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Project Structure

```
finding_donors/
│
├── finding_donors.ipynb       # Main analysis and modeling notebook
├── finding_donors_gui.py      # Main GUI launcher
├── app_gui.py                 # GUI application logic
├── models.py                  # Model definitions and training logic
├── data_utils.py              # Data loading and preprocessing utilities
├── constants.py               # Project-wide constants and configuration
├── visuals.py                 # Visualization helper functions
├── census.csv                 # Dataset
├── project_description.md     # Project description and notes
└── README.md
```

## How to Run

Run the notebook:

```bash
jupyter notebook finding_donors.ipynb
```

Launch the GUI:

```bash
python finding_donors_gui.py
```

## Dataset

The census dataset contains approximately 32,000 data points with 13 features each. It is a modified version of the dataset from the paper "Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid" by Ron Kohavi, originally hosted on UCI Machine Learning Repository.

**Target Variable:** `income` — whether an individual earns <=50K or >50K per year.

**Features include:** age, workclass, education level, marital status, occupation, relationship, race, sex, capital gain, capital loss, hours per week, and native country.

## Models Trained

Three supervised learning algorithms were trained and compared:

- Logistic Regression
- Random Forest
- Gradient Boosting

Models were evaluated on accuracy, F-beta score, and training time. The best performing model was further optimized using GridSearchCV.

## GUI Demo

### GUI

<p align="center">
  <img src="C:\Users\Test\Desktop\project_2ML\Machine_Learning_project\P2\images\GUI.PNG" width="900">
</p>

### Video Demo

## Demo

<p align="center">
  ▶️ <a href="C:\Users\Test\Desktop\project_2ML\Machine_Learning_project\P2\video\GUIDonor.mp4">Watch the GUI demo</a>
</p>

## Enhancements Beyond the Base Project

- Modular codebase split across dedicated files: `models.py`, `data_utils.py`, `constants.py`
- Compared three classification algorithms with detailed performance analysis
- Integrated MLflow for experiment tracking, metric logging, and model versioning
- Built a Tkinter desktop GUI (`app_gui.py` + `finding_donors_gui.py`) for real-time predictions on new donor profiles

## Results

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 71.87%   |
| Random Forest       | 64.40%   |
| Gradient Boosting   | 72.01%   |

Replace the placeholders above with your actual results from the notebook.

## Acknowledgements

- Dataset: UCI Machine Learning Repository
- Original project template: Udacity Machine Learning Engineer Nanodegree

```

```
