Rosetta Stone
=============

# 🏄 Introducton
make your deep learning life easier

**Rosetta Stone** is a toolkit that aims to make your deep learning life easier. It enables users to performe end-to-end experiment quickly and efficiently. In comparison with the other open source libraries, Rosetta is an alternate low-code toolkit that can be used to perform deep learning tasks with only few lines of code. Rosetta is essentially a wrapper aroud pytorch, apex, tensorboardX and many more. 

The key features are:

- yaml-styled model for elegantly configuring complex applications
- best practice
- Unified design for various applications
- Pre-trained models
- State-of-the-art performance

# 👷‍ Installation

## Requirements

- Python >= 3.6
- Pytorch >= 1.4.0

## Setup `rosetta-stone`

- **setup with `pip`**

```bash
$ pip install rosetta-stone
```

- **setup with `Docker`**

    1. build docker image

    ```bash
    $ docker build --tag huya_ai:rosetta .
    ```

    2. run the docker container

    ```bash
    $ docker run --rm -it -v $(PWD):/tmp/rosetta --name rosetta huya_ai:rosetta bash
    ```

# 🚀 Train model with `rosetta-stone`

- training from scratch

```bash
$ rosetta train resnet56 --yaml-path app.yaml
```

- overrides parameters defined in yaml file

```bash
# the paramer of `--yaml-path` has default value `app.yaml`
$ rosetta train resnet56 --batch_size=125
```

- training using automatic mixture precision (amp)

```bash
$ rosetta train resnet56 --yaml-path app.yaml --use-amp
```

- distributed training using `torch.distributed.launch` (recommended)

```bash
$ python -m torch.distributed.launch --module --nproc_per_node=#{GPU_NUM} rosetta.main train resnet56
```

- distributed training using `horovod`

```bash
$ rosetta train resnet56 --yaml-path app.yaml --use-horovod
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
- [padertorch](https://github.com/fgnt/padertorch): A collection of common functionality to simplify the design, training and evaluation of machine learning models based on pytorch with an emphasis on speech processing.
- [Tips, tricks and gotchas in PyTorch](https://coolnesss.github.io/2019-02-05/pytorch-gotchas)
- [PyTorch Parallel Training](https://zhuanlan.zhihu.com/p/145427849): PyTorch Parallel Training（单机多卡并行、混合精度、同步BN训练指南文档）
- [给训练踩踩油门 —— Pytorch 加速数据读取](https://zhuanlan.zhihu.com/p/80695364)
- [高性能PyTorch是如何炼成的？](https://mp.weixin.qq.com/s/x7u26Ok7O4xMOETmUYROJQ)
- [service-streamer](https://github.com/ShannonAI/service-streamer): Boosting your Web Services of Deep Learning Applications.