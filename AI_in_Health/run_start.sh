#!/bin/bash
git clone https://github.com/forpetesaken/Individual_project.git
cd Individual_project || exit 1

## Exploratory Data Analysis
/usr/local/bin/python "./AI_in_Health/eda.py" --csv "./AI_in_Health/training.validation_data.csv"

## Running model
/usr/local/bin/python "./AI_in_Health/compression_onset.py" --csv "./AI_in_Health/training.validation_data.csv" --out "./AI_in_Health/output_model_bundle.joblib"

## Testing model
/usr/local/bin/python "./AI_in_Health/compression_onset_test.py" --bundle "./AI_in_Health/output_model_bundle.joblib" --test_csv "./AI_in_Health/testing_data.csv"
