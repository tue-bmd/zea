# Inherit from usbmd base image
FROM usbmd/all:latest

# Set working directory
WORKDIR /ultrasound-toolbox
COPY . /ultrasound-toolbox/

# Install usbmd (with headless opencv)
RUN pip install -e .

# Source working/installation directory and add motd (message of the day)
ENV INSTALL /usr/local/src
RUN echo '[ ! -z "$TERM" -a -r /etc/motd ] && cat /etc/motd' \
    >> /etc/bash.bashrc \
    ; echo "\
================================================================\n\
    UU   UU   SSSSS   BBBBB    MMM MMM   DDDD     \e[31mv$(pip show usbmd | grep Version | cut -d ' ' -f 2)\e[0m\n\
    UU   UU  SS       BB   BB  MMMMMMM   DD  D\n\
    UU   UU   SSSS    BBBBB    MM M MM   DD   D\n\
    UU   UU      SS   BB   BB  MM   MM   DD  D\n\
     UUUUU   SSSSS    BBBBB    MM   MM   DDDD     (c) \e[31mTU/e 2021\e[0m\n\
================================================================\n\
"\
    > /etc/motd

# Set entrypoint
ENTRYPOINT ["/bin/bash"]