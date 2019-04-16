FROM datagovsg/python-spark:2.7-2.1
RUN printf "deb http://archive.debian.org/debian/ jessie main\ndeb-src http://archive.debian.org/debian/ jessie main\ndeb http://security.debian.org jessie/updates main\ndeb-src http://security.debian.org jessie/updates main" > /etc/apt/sources.list
RUN apt-get update && apt-get install -y nano && pip install numpy
COPY log4j.properties /opt/spark-2.1.2/conf/
CMD 'bin/bash'