#!/bin/bash

# Check worker and study data for approvals and quality checks
python3 -m src.data_collection.pull_crowdworkers
python3 -m src.quality_control.run_worker_checks
python3 -m src.quality_control.run_study_checks
python3 -m src.data_processing.merge_crowd_details