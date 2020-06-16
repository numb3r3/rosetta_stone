Rosetta Stone
=============

# 🏄 Introducton
make your deep learning life easier

**Rosetta Stone** is a toolkit that aims to make your deep learning life easier. It enables users to performe end-to-end experiment quickly and efficiently. In comparison with the other open source libraries, Rosetta is an alternate low-code toolkit that can be used to perform deep learning tasks with only few lines of code. Rosetta is essentially a wrapper aroud pytorch, apex, tensorboardX and many more. 

The key features are:

- yaml-styled model construction and hyper parameters setting
- best practice
- Unified design for various applications
- Pre-trained models
- State-of-the-art performance

# 👷‍ Installation

## Requirements

- Python >= 3.6
- Pytorch >= 1.4.0

## Setup with Docker

- build docker image

```bash
$ docker build --tag huya_ai:rosetta .
```

- run the docker container

```bash
$ docker run --rm -it -v $(PWD):/tmp/rosetta --name rosetta huya_ai:rosetta bash
```

# 🤖 To use `Rosetta`

- training from scratch

```bash
$ python app.py resnet56
```

- overrides parameters defined in yaml file

```bash
$ python app.py resnet56 --batch_size=125
```

- training using automatic mixture precision (amp)

```bash
$ python app.py resnet56 --use_amp
```



# Contribution Guide
You can contribute to this project by sending a merge request. After approval, the merge request will be merged by the reviewer.

Before making a contribution, please confirm that:
- Code quality stays consistent across the script, module or package.
- Code is covered by unit tests.
- API is maintainable.

# 👩‍💻 References

- [homura](https://github.com/moskomule/homura): PyTorch utilities including trainer, reporter, etc.
- [FARM](https://github.com/deepset-ai/FARM): Fast & easy transfer learning for NLP. Harvesting language models for the industry. 
- [kotonoha](https://github.com/moskomule/kotonoha): NLP utilities for research
- [Tips, tricks and gotchas in PyTorch](https://coolnesss.github.io/2019-02-05/pytorch-gotchas)
- [给训练踩踩油门 —— Pytorch 加速数据读取](https://zhuanlan.zhihu.com/p/80695364)