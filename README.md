<p align="center">
<a href="#">
    <img src=".github/github-banner.png?raw=true" height="200" alt="">
</a>
</p>

# Rosetta Stone <img src=".github/logo.png?raw=true" width="120" align="right" alt="">


`Rosetta Stone` is a lightweight `PyTorch` wrapper & toolkit that aims to make your deep learning life easier. It enables users to performe end-to-end experiment quickly and efficiently. In comparison with the other open source libraries, Rosetta is an alternate low-code toolkit that can be used to perform deep learning tasks with only few lines of code. It easy to use and make you focus on designing your models!

⚠️  The code is still under active development (with no frozen API), but is already usable in applications.

## Features

- yaml-styled model for elegantly configuring complex applications
- best practice
- Unified design for various applications
- Pre-trained models
- State-of-the-art performance

# 🚀 Installation

### Requirements

    - Python >= 3.6
    - Pytorch >= 1.4.0

### Setup

Install the **latest** version from source

```bash
# clone the project repository, and install via pip
$ git clone https://git.huya.com/wangfeng2/rosetta_stone.git \
    && cd rosetta_stone \
    && pip install -e .
```

or **released** stable version via `pip`:

```bash
$ pip install --upgrade rosetta-stone
```

For ease-of-use, you can also use rosetta with `Docker`:

```bash
# build docker image
$ docker build --tag huya_ai:rosetta .

# run the docker container
$ docker run --rm -it -v $(PWD):/rosetta --name rosetta huya_ai:rosetta bash
```

# 📖 Usage

In `rosetta` you don’t need to specify a training loop, just define the dataLoaders and the models. For `ResNet` example,

- **Step 1**: Create YAML Configuration

create a yaml file (usually named as `app.yaml`) within your repo as the example below.

    ```YAML
    resnet56: &resnet56
      model_module: examples.vision.resnet_model:ResNet
      dataio_module: examples.vision.cifar10:CIFAR10

      batch_size: 256
      num_classes: 10

      n_size: 9
    ```

- **Step 2**: Define Dataloader

- **Step 3**: Define Model



- **Step 4**: Start to train

    - training from scratch

        ```bash
        $ rosetta train resnet56 --yaml-path app.yaml
        ```

    - overrides parameters defined in yaml file

        ```bash
        # the cli paramer `--yaml-path` has default value `app.yaml`
        $ rosetta train resnet56 --batch_size=125
        ```

    - training using automatic mixture precision (amp)

        ```bash
        $ rosetta train resnet56 --yaml-path app.yaml --use-amp
        ```

    - distributed training using `torch.distributed.launch` (recommended)

        ```bash
        $ python -m torch.distributed.launch --module --nproc_per_node={GPU_NUM} rosetta.main train resnet56
        ```

    - distributed training using `horovod` (not recommended)

        ```bash
        $ rosetta train resnet56 --use-horovod
        ```

# 👋 Contribution Guide
You can contribute to this project by sending a merge request. After approval, the merge request will be merged by the reviewer.

Before making a contribution, please confirm that:
- Code quality stays consistent across the script, module or package.
- Code is covered by unit tests.
- API is maintainable.

# 👍 References

- [homura](https://github.com/moskomule/homura): PyTorch utilities including trainer, reporter, etc.
- [FARM](https://github.com/deepset-ai/FARM): Fast & easy transfer learning for NLP. Harvesting language models for the industry.
- [kotonoha](https://github.com/moskomule/kotonoha): NLP utilities for research
- [padertorch](https://github.com/fgnt/padertorch): A collection of common functionality to simplify the design, training and evaluation of machine learning models based on pytorch with an emphasis on speech processing.
- [Tips, tricks and gotchas in PyTorch](https://coolnesss.github.io/2019-02-05/pytorch-gotchas)
- [PyTorch Parallel Training](https://zhuanlan.zhihu.com/p/145427849): PyTorch Parallel Training（单机多卡并行、混合精度、同步BN训练指南文档）
- [给训练踩踩油门 —— Pytorch 加速数据读取](https://zhuanlan.zhihu.com/p/80695364)
- [高性能PyTorch是如何炼成的？](https://mp.weixin.qq.com/s/x7u26Ok7O4xMOETmUYROJQ)
- [service-streamer](https://github.com/ShannonAI/service-streamer): Boosting your Web Services of Deep Learning Applications.
- [Masked batchnorm in PyTorch](https://yangkky.github.io/2020/03/16/masked-batch-norm.html)
- [ru_transformers](https://github.com/mgrankin/ru_transformers): train GPT-2 on Google TPUs
