#!/bin/bash

# define out dir
OUT=${1:-"faith/library/temporal_annotator/sutime_service.out"}

# start server
nohup python -u faith/library/temporal_annotator/sutime_date_annotator_service.py > $OUT 2>&1 &