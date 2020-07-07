# Run the following commands
# docker build --tag huya_ai:rosetta .
# docker run --rm -it -v $(PWD):/tmp/rosetta --name resetta huya_ai:rosetta bash
FROM python:3.6.8

ENV TERM=xterm-256color

WORKDIR /tmp/rosetta

RUN pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

CMD ["/bin/bash"]


