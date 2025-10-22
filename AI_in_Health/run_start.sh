%cd /content
!git clone https://github.com/forpetesaken/Individual_project.git
%cd /content/Individual_project
!source .venv/bin/activate

## Exploratory Data Analysis
!/usr/local/bin/python "/content/Individual_project/AI_in_Health/eda.py" --csv "/content/Individual_project/AI_in_Health/training.validation_data.csv"

## Running model
!/usr/local/bin/python "/content/Individual_project/AI_in_Health/compression_onset.py" --csv "/content/Individual_project/AI_in_Health/training.validation_data.csv"

## Testing model
!/usr/local/bin/python "/content/Individual_project/AI_in_Health/compression_onset_test.py" --bundle "/content/Individual_project/model_bundle.joblib" --test_csv "/content/Individual_project/AI_in_Health/testing_data.csv"
