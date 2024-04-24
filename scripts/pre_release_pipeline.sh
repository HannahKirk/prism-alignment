#!/bin/bash
# This script runs all the steps for processing and cleaning the data prior to public release

./collect_data.sh

./process_data.sh

./check_crowd.sh

./prepare_data_for_release.sh

./prepare_metadata.sh