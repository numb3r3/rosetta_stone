Rosetta Stone
=============

**Resetta Stone** is a toolkit for Natural Lanauge Processing (NLP).

- Unified design for various applications
- Pre-trained models
- State-of-the-art performance

## Requirements

- Python >= 3.6
- Pytorch >= 1.4.0

## Setup with Docker

- build docker image

```bash
docker build --tag huya_ai:rosetta .
```

- run the docker container

```bash
docker run --rm -it -v $(PWD):/tmp/rosetta --name rosetta huya_ai:rosetta bash
```

## References

- [homura](https://github.com/moskomule/homura): PyTorch utilities including trainer, reporter, etc.