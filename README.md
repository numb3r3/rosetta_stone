Rosetta Stone
=============

**Resetta Stone** is a toolkit that aims to reduce the cycle time in a deep learning experiment. It enables users to performe end-to-end experiment quickly and efficiently. In comparison with the other open source libraries, Rosetta is an alternate low-code toolkit that can be used to perform deep learning tasks with only few lines of code. Rosetta is essentially a wrapper aroud pytorch, apex, tensorboardX and many more. 

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

## Contribution Guide
You can contribute to this project by sending a merge request. After approval, the merge request will be merged by the reviewer.

Before making a contribution, please confirm that:
- Code quality stays consistent across the script, module or package.
- Code is covered by unit tests.
- API is maintainable.

## References

- [homura](https://github.com/moskomule/homura): PyTorch utilities including trainer, reporter, etc.
- [kotonoha](https://github.com/moskomule/kotonoha): NLP utilities for research