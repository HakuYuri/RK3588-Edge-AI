#!/bin/bash

# Define variables
TARGET_NAME="node_exporter"
REPO_URL="https://api.github.com/repos/prometheus/node_exporter/releases/latest"

echo "Checking for the latest node_exporter arm64 release..."

# 1. Get the download URL of the arm64 version using GitHub API
# Using grep and cut to extract the browser_download_url for linux-arm64
DOWNLOAD_URL=$(curl -s $REPO_URL | grep "browser_download_url" | grep "linux-arm64.tar.gz" | cut -d '"' -f 4)

if [ -z "$DOWNLOAD_URL" ]; then
    echo "Error: Could not find the arm64 download URL."
    exit 1
fi

echo "Downloading from: $DOWNLOAD_URL"

# 2. Download the package
curl -L -o node_exporter_latest.tar.gz "$DOWNLOAD_URL"

if [ $? -ne 0 ]; then
    echo "Error: Download failed."
    exit 1
fi

# 3. Create a temporary directory for extraction
mkdir -p ./tmp_extract

echo "Extracting files..."
tar -xzf node_exporter_latest.tar.gz -C ./tmp_extract

# 4. Locate the binary, move and rename it
# The tarball usually contains a subfolder like 'node_exporter-1.x.x.linux-arm64/'
EXTRACTED_DIR=$(ls ./tmp_extract)
mv "./tmp_extract/$EXTRACTED_DIR/node_exporter" "./$TARGET_NAME"

echo "Cleaning up temporary files..."
rm -rf node_exporter_latest.tar.gz ./tmp_extract

# 5. Set execution permissions
chmod +x "$TARGET_NAME"

echo "Success! The binary has been renamed to '$TARGET_NAME' and is ready to use."
./$TARGET_NAME --version