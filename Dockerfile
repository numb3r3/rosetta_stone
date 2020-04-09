# Run the following commands
# docker build --tag huya_ai:rosetta .
# docker run --rm -it -v $(PWD):/tmp/resetta --name resetta huya_ai:rosetta bash
FROM python:3.6.8

WORKDIR /tmp/resetta

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]


