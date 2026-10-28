FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common gpg-agent ca-certificates \
 && add-apt-repository -y ppa:wfg/0ad && apt-get update \
 && apt-get install -y --no-install-recommends 0ad python3 xvfb xauth mesa-utils libgl1-mesa-dri ffmpeg curl \
 && rm -rf /var/lib/apt/lists/*
