#!/bin/bash
gunicorn -w 1 --thread 3 -t 60 --max-request 8 --pid process.pid --bind 0.0.0.0:5000 --daemon run:application --access-logfile logs/access.log --error-logfile logs/error.log
echo "Service Started!"
