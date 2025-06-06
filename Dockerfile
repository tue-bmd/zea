# Inherit from zea base image
FROM zeahub/all:latest

# Set working directory
WORKDIR /ultrasound-toolbox
COPY . /ultrasound-toolbox/

# Install zea
RUN pip install -e .

# Source working/installation directory and add motd (message of the day)
ENV INSTALL=/usr/local/src
RUN echo '[ ! -z "$TERM" -a -r /etc/motd ] && cat /etc/motd' \
    >> /etc/bash.bashrc \
    ; echo "\
=========================================\n\
  ZZZZZ   EEEEE   AAAAA     \e[31mv$(pip show zea | grep Version | cut -d ' ' -f 2)\e[0m\n\
     ZZ   EE     AA   AA\n\
    ZZ    EEEE   AAAAAAA\n\
   ZZ     EE     AA   AA\n\
  ZZZZZ   EEEEE  AA   AA    \e[31mTU/e 2021\e[0m\n\
=========================================\n\
"\
    > /etc/motd
