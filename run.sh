#!/bin/bash

# Image Annotation System Runner
# Usage: ./run.sh [user_folder] [port]

USER_FOLDER=${1:-"example_user"}
PORT=${2:-7865}

echo "Starting Image Annotation System..."
echo "User folder: $USER_FOLDER"
echo "Port: $PORT"
echo ""

python app.py --user-folder "$USER_FOLDER" --port "$PORT"
