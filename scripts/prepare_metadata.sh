#!/bin/bash

# First get text instances for processing
python3 -m src.data_processing.get_text_instances

# Now process the text instances to get metadata
python3 -m src.data_processing.add_metadata


