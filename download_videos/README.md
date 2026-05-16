# Downloading YouTube Videos through a script

## General

We use a python script to download bunch of YouTube videos from a list of URLs.

The list of URLs is provided inside a urls.txt file.

The script reads the URLs from the file and downloads each video using the `yt-dlp` library.

## Usage

```bash

# Create your own virual environment
python3 -m venv venv
# Source it
source venv/bin/activate

# Necessary dependencies
pip install yt-dlp
sudo apt update && sudo apt install -y ffmpeg

# download all the videos in the full_videos directory
python download_videos.py --out full_videos
```

> NOTE: Make sure you have the `urls.txt` file in the same directory as the script. The script may take some time to download all the videos.
