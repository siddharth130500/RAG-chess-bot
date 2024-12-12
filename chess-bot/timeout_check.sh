#!/bin/bash
# timeout_check.sh

# Define inactivity timeout (in seconds)
INACTIVITY_TIMEOUT=120  # e.g., 2 minutes

# Get the timestamp of the last access
LAST_ACCESS=$(stat -c %Y /var/log/nginx/access.log)

# Compare with the current time
CURRENT_TIME=$(date +%s)
TIME_DIFF=$((CURRENT_TIME - LAST_ACCESS))

# Define the container name and cron job identifier
CONTAINER_NAME="siddharth130500/chess-bot-app"
CRON_JOB_ID="#StopDockerContainerCron"

# Stop the container if it exceeds the timeout
if [ "$TIME_DIFF" -ge "$INACTIVITY_TIMEOUT" ]; then
    docker stop $(docker ps -q --filter ancestor=$CONTAINER_NAME)
    
    # Remove the cron job associated with stopping the container
    crontab -l | grep -v "$CRON_JOB_ID" | crontab -
fi
