
# Release Note (`v1.1.16`)
> Release time: 2020-08-20 12:47:14


🙇 We'd like to thank all contributors for this new release! In particular,
 wangfeng,  🙇


### 🐞 Bug fixes
 - [[```aa0fef0f```](https://github.com/numb3r3/rosetta_stone/commit/aa0fef0f158df29d6ca3d258b231387dfdf39cae)] __-__ lr decay strategy in distributed mode (*wangfeng*)

### 🍹 Other Improvements
 - [[```3c92b149```](https://github.com/numb3r3/rosetta_stone/commit/3c92b149bead5b05b9246bbc34b7809858b4b41a)] __-__ fix amp under distributed env (*wangfeng*)
 - [[```cded3538```](https://github.com/numb3r3/rosetta_stone/commit/cded3538062c572802449883acaacd0bfa14da3c)] __-__ __changelog__: update change log to v1.1.15 (*wangfeng*)

# Release Note (`v1.1.15`)
> Release time: 2020-08-06 11:12:24


🙇 We'd like to thank all contributors for this new release! In particular,
 wangfeng,  🙇


### 🆕 New Features
 - [[```1d6f1f79```](https://github.com/numb3r3/rosetta_stone/commit/1d6f1f79451593193e656e727d727564e6806e9c)] __-__ load model&#39;s hparams from yaml (*wangfeng*)

### 🐞 Bug fixes
 - [[```8e875692```](https://github.com/numb3r3/rosetta_stone/commit/8e875692db02a12af54dc3f6678d0317ba3e1a04)] __-__ __trainer__: scale the learning rate by the number of workers (half of workers) (*wangfeng*)

### 🍹 Other Improvements
 - [[```0ff7f1ae```](https://github.com/numb3r3/rosetta_stone/commit/0ff7f1aec93d7b4f39b735093ee63c688f620922)] __-__ add isort.cfg (*wangfeng*)
 - [[```b1816616```](https://github.com/numb3r3/rosetta_stone/commit/b1816616172844c27090077bf70fe6197809444d)] __-__ __changelog__: update change log to v1.1.14 (*wangfeng*)

# Release Note (`v1.1.14`)
> Release time: 2020-07-16 14:49:06


🙇 We'd like to thank all contributors for this new release! In particular,
 wangfeng,  🙇


### 🐞 Bug fixes
 - [[```b7b426cb```](https://github.com/numb3r3/rosetta_stone/commit/b7b426cb43cc7078a0dec4f8b3b93e1a2e5488d3)] __-__ __trainer__: fix #issue [22049](https://github.com/pytorch/pytorch/issues/22049) Cannot update part of the parameters in DistributedDataParallel (*wangfeng*)

### 🍹 Other Improvements
 - [[```a552c464```](https://github.com/numb3r3/rosetta_stone/commit/a552c464509bd39492186a9bd270dedc3124cc4e)] __-__ format codes to pass pre-commit (*wangfeng*)
 - [[```146dc6f5```](https://github.com/numb3r3/rosetta_stone/commit/146dc6f55f8c1894a7acab5b5610e23eb34a387e)] __-__ __changelog__: update change log to v1.1.13 (*wangfeng*)

# Release Note (`v1.1.13`)
> Release time: 2020-07-16 11:39:10


🙇 We'd like to thank all contributors for this new release! In particular,
 wangfeng,  🙇


### 🍹 Other Improvements
 - [[```8a259140```](https://github.com/numb3r3/rosetta_stone/commit/8a25914004324abbe9dfaa97b5e4c004e8bc05ef)] __-__ remove unused requirement (*wangfeng*)
 - [[```2b3df569```](https://github.com/numb3r3/rosetta_stone/commit/2b3df5692668fed51c8950fa3553de7fc5b6c6b7)] __-__ __changelog__: update change log to v1.1.12 (*wangfeng*)

# Release Note (`v1.1.12`)
> Release time: 2020-07-16 11:29:58


🙇 We'd like to thank all contributors for this new release! In particular,
 wangfeng,  🙇


### 🆕 New Features
 - [[```f19551d7```](https://github.com/numb3r3/rosetta_stone/commit/f19551d7aa1369eb226c0d7cec25a3df23ae9c02)] __-__ __lr_decay__: enable decay learning rate by epoch (*wangfeng*)

### 🐞 Bug fixes
 - [[```294b02ca```](https://github.com/numb3r3/rosetta_stone/commit/294b02cad0155a071154b3cb09a863dca6265f54)] __-__ __trainer__: remove mistake logging (*wangfeng*)

### 📗 Documentation
 - [[```5caa1640```](https://github.com/numb3r3/rosetta_stone/commit/5caa1640d2c177a272e84be01c7683ff6af4cf16)] __-__ format readme (*wangfeng*)
 - [[```dabe739e```](https://github.com/numb3r3/rosetta_stone/commit/dabe739e67259dc4f95b83897209f061e6c0484a)] __-__ update readme (*wangfeng*)

### 🍹 Other Improvements
 - [[```88a75de5```](https://github.com/numb3r3/rosetta_stone/commit/88a75de5ceb42d15a0c1860864b51b364f2a0309)] __-__ format codes (*wangfeng*)
 - [[```240a15cc```](https://github.com/numb3r3/rosetta_stone/commit/240a15ccdcca8dede754e8be89be9f9605842061)] __-__ __changelog__: update change log to v1.1.11 (*wangfeng*)

# Release Note (`v1.1.11`)
> Release time: 2020-07-14 18:23:52


🙇 We'd like to thank all contributors for this new release! In particular,
 wangfeng,  🙇


### 🐞 Bug fixes
 - [[```f268af6a```](https://github.com/numb3r3/rosetta_stone/commit/f268af6a87e2516423a9f4ec473fb331e13c9b3d)] __-__ __docker__: fix docker dev container (*wangfeng*)
 - [[```cd4c8a1b```](https://github.com/numb3r3/rosetta_stone/commit/cd4c8a1bca22838c1ece7c80e56ac1625c3be2a3)] __-__ add missing release template (*wangfeng*)

### 🚧 Code Refactoring
 - [[```98794850```](https://github.com/numb3r3/rosetta_stone/commit/987948506e5e6e5181d0930f2ce35e0209b1b88a)] __-__ __model_api__: enable resume from pretrained model in func of create_model (*wangfeng*)

### 📗 Documentation
 - [[```32de8780```](https://github.com/numb3r3/rosetta_stone/commit/32de8780fe39925a63115247a30bf6af9e7d8fd8)] __-__ update README (*wangfeng*)
 - [[```4eb1c79e```](https://github.com/numb3r3/rosetta_stone/commit/4eb1c79e8e4ada7330b3e38b605834f7df7278de)] __-__ update readme (*wangfeng*)

### 🍹 Other Improvements
 - [[```e2ac2f85```](https://github.com/numb3r3/rosetta_stone/commit/e2ac2f8577ee431cb8fd32cf57c028f61cb69dff)] __-__ __changelog__: update change log to v1.1.10 (*wangfeng*)

# Release Note (`v1.1.10`)
> Release time: 2020-07-13 17:42:17


🙇 We'd like to thank all contributors for this new release! In particular,
 wangfeng,  🙇


### 🆕 New Features
 - [[```1f76af9c```](https://github.com/numb3r3/rosetta_stone/commit/1f76af9cd381d9b4a5e707e02cfbc5ec53df0965)] __-__ __evaluator__: add evaluate command, support standalone model only by now (*wangfeng*)

### 🐞 Bug fixes
 - [[```c9ca9514```](https://github.com/numb3r3/rosetta_stone/commit/c9ca9514498a643ef4ba5f256b8093d2f2661336)] __-__ __avg_meter__: overides keys() func (*wangfeng*)
 - [[```ea7e1c37```](https://github.com/numb3r3/rosetta_stone/commit/ea7e1c3756cf752c3768de1b26287d26191667a2)] __-__ __trainer__: revise logx printing (*wangfeng*)

### 📗 Documentation
 - [[```144625b7```](https://github.com/numb3r3/rosetta_stone/commit/144625b76308b676b9b82c4b60bbada52d7473dd)] __-__ minor revision (*wangfeng*)
 - [[```6dd28ca3```](https://github.com/numb3r3/rosetta_stone/commit/6dd28ca311ce0e50fd6ed72474649a2a78f4bcda)] __-__ update usage instruction (*wangfeng*)

### 🍹 Other Improvements
 - [[```92704b7e```](https://github.com/numb3r3/rosetta_stone/commit/92704b7e0826127c6544c9fd2addf62c3af0c762)] __-__ __changelog__: update change log to v1.1.9 (*wangfeng*)

# Release Note (`v1.1.9`)
> Release time: 2020-07-12 21:23:35


🙇 We'd like to thank all contributors for this new release! In particular,
 wangfeng,  🙇


### 🐞 Bug fixes
 - [[```0a63e115```](https://github.com/numb3r3/rosetta_stone/commit/0a63e115eef11cc00b712e055aadd3ca25e2eb49)] __-__ __core__: fix reporting error (*wangfeng*)

### 🍹 Other Improvements
 - [[```bd3fccef```](https://github.com/numb3r3/rosetta_stone/commit/bd3fccef9bd5e8d2bfde4e945334ebad3850e118)] __-__ __changelog__: update change log to v1.1.8 (*wangfeng*)

# Release Note (`v1.1.8`)
> Release time: 2020-07-12 20:59:22


🙇 We'd like to thank all contributors for this new release! In particular,
 wangfeng,  🙇


### 🐞 Bug fixes
 - [[```82407570```](https://github.com/numb3r3/rosetta_stone/commit/82407570d8ec25071b8d999318450ad2151bcca8)] __-__ __setup__: add missing package pytest (*wangfeng*)

### 🚧 Code Refactoring
 - [[```a9c3bf22```](https://github.com/numb3r3/rosetta_stone/commit/a9c3bf22bfeabdb5e10189f0338c72ed3a98ed62)] __-__ report all valid metrics after each epoch (*wangfeng*)

### 🍹 Other Improvements
 - [[```5a8cb688```](https://github.com/numb3r3/rosetta_stone/commit/5a8cb688412ed8097131d4d11080dc4da31d6698)] __-__ __changelog__: update change log to v1.1.7 (*wangfeng*)

# Release Note (`v1.1.7`)
> Release time: 2020-07-12 18:00:43


🙇 We'd like to thank all contributors for this new release! In particular,
 wangfeng,  🙇


### 🐞 Bug fixes
 - [[```334999a3```](https://github.com/numb3r3/rosetta_stone/commit/334999a3d5df6610a8d3f5ca3baf7a43e9fc54fd)] __-__ replace forked runx package instead to fix log_dir issue (*wangfeng*)
 - [[```d322052d```](https://github.com/numb3r3/rosetta_stone/commit/d322052dcf8ebdb7da90800c1fa7767e8675a99b)] __-__ merge chore-bumping locally (temporary option) (*wangfeng*)

### 🍹 Other Improvements
 - [[```af315d1d```](https://github.com/numb3r3/rosetta_stone/commit/af315d1defcd87a629ade5f05090a7870b55bfbc)] __-__ ... (*wangfeng*)
 - [[```4090317b```](https://github.com/numb3r3/rosetta_stone/commit/4090317beb0218bac61eafca71939f2f02def5b4)] __-__ fix release.sh (*wangfeng*)
 - [[```2d0ac832```](https://github.com/numb3r3/rosetta_stone/commit/2d0ac832c584941243d94c8b49ed46eb51de2b7d)] __-__ __changelog__: update change log to v1.1.6 (*wangfeng*)

# Release Note (`v1.1.6`)
> Release time: 2020-07-11 16:54:18


🙇 We'd like to thank all contributors for this new release! In particular,
 wangfeng,  🙇


### 🐞 Bug fixes
 - [[```ace08ea1```](https://github.com/numb3r3/rosetta_stone/commit/ace08ea1d9ef363eafb974c92875d394a38aa5fc)] __-__ enable release note (*wangfeng*)

### 🚧 Code Refactoring
 - [[```32b7113c```](https://github.com/numb3r3/rosetta_stone/commit/32b7113c73ae809d7c26e3537feab4de3bbba56e)] __-__ load_checkpoint returns state_dict and metas (*wangfeng*)
 - [[```1ca73b8b```](https://github.com/numb3r3/rosetta_stone/commit/1ca73b8bbff0b518a176bfc5a065194e31af1bd0)] __-__ change AverageMeter&#39;s average func (*wangfeng*)

### 📗 Documentation
 - [[```82805fa6```](https://github.com/numb3r3/rosetta_stone/commit/82805fa6f94eae1ddaa7070c815b7e1ae9704b43)] __-__ update readme (*wangfeng*)
