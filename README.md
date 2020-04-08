Rosetta Stone
=============

**Resetta Stone** is a toolkit for Natural Lanauge Processing (NLP).

# Requirements
- Python >= 3.6
- Pytorch >= 1.10

# Setup with Docker

- build docker image

```bash
docker build --tag huya_ai:rosetta .
```

- run the docker container

```bash
docker run --rm -it -v $(PWD):/tmp/resetta --name resetta huya_ai:rosetta bash
```