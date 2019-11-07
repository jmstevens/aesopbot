FROM python:3.7-stretch
MAINTAINER Joel Stevens <joelstevens553@gmail.com>

# install build utilities
RUN apt-get update && \
	apt-get install -y gcc make apt-transport-https ca-certificates build-essential

# check our python environment
RUN python3 --version
RUN pip3 --version



# Installing python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# # Copy all the files from the project’s root to the working directory
COPY src/ /src/
COPY configs /src/configs
COPY data /src/data

RUN ls -la /src/*

WORKDIR  /src
