# Inherit from usbmd base image
FROM usbmd_base:latest

# Set working directory
WORKDIR /ultrasound-toolbox
COPY . /ultrasound-toolbox/

# Install usbmd (with headless opencv)
RUN pip install -e .

# Set entrypoint
ENTRYPOINT ["/bin/bash"]