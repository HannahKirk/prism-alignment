#!/bin/bash
# This script prepares the data for public release
# For crowdworker data, we used the merged version with approved workers
python3 -m src.data_processing.prepare_survey --merged_version
python3 -m src.data_processing.prepare_conversations

