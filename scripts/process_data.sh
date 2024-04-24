#!/bin/bash

# Define the cutoff date
# Beta Testing Round 2 started on 6th Nov 2023
# Launch 22nd Nov 2023
CUTOFF_DATE=$"2023-11-22"

# Run the initial data cleaning scripts
python3 -m src.data_processing.process_survey --cutoff_date $CUTOFF_DATE
python3 -m src.data_processing.process_conversations --cutoff_date $CUTOFF_DATE