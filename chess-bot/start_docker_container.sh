#!/bin/bash

# Define your container name
CONTAINER_NAME="siddharth130500/chess-bot-app"

# Check if the container is already running
if [ "$(sudo docker ps -q -f name=${CONTAINER_NAME})" ]; then
    echo "Container is already running."
else
    # Start the Docker container
    echo "Starting the Docker container..."
    sudo docker run -d -p 5000:5000 -v /mnt/e/SDSC/env_var/.env:/app/.env ${CONTAINER_NAME}

    # Define the cron job to stop the container after inactivity
    CRON_JOB="*/3 * * * * /mnt/e/SDSC/Chess_bot/chess-bot/timeout_check.sh #StopDockerContainerCron"

    # Check if the cron job already exists
    (crontab -l | grep -q "$CRON_JOB") || (crontab -l; echo "$CRON_JOB") | crontab - 
fi
