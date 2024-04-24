#!/bin/bash

# Run the data collection scripts (survey and conversations are different interfaces)
python3 -m src.data_collection.pull_survey
python3 -m src.data_collection.pull_conversations