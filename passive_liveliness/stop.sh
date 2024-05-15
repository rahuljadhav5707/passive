#!/bin/bash
PROCESS_ID=$(cat process.pid)
kill -9 $PROCESS_ID
echo "Service Stopped!"
