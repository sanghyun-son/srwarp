# SRWarp
![demo](src/record/demo.gif)

This repository contains an official implementation of the following CVPR 2021 paper:

Sanghyun Son and Kyoung Mu Lee, "SRWarp: Generalized Image Super-Resolution under Arbitrary Transformation."

## Requirements

This repository is tested under:
- Ubuntu 18.04
- CUDA 10.1 (Compute Capability 7.5)
- &gt;= PyTorch 1.6

We have found some issues with CUDA 10.0 version.
Please use a proper CUDA version for building this repository. 

Also, please install the following CUDA extension. More details are described in [this repository](https://github.com/sanghyun-son/pysrwarp).
```bash
$ git clone https://github.com/sanghyun-son/pysrwarp
$ cd pysrwarp
$ make
```

We also recommend using the attached `environment.yaml` file to easily set up the environment.
```bash
$ conda env create --file environment.yaml
$ conda activate srwarp
```

## Setup

Browse to the `pretrained` directory and download the pretrained checkpoints by following:
```bash
$ bash download.sh
```
Otherwise, download them manually. Please note that those files are ~3x larger than the original model due to the saved optimizers.

| SRWarp (MDSR) | SRWarp (MRDB) |
|:---:|:---:|
| [Download](https://cv.snu.ac.kr/research/srwarp/srwarp_mdsr-c66c4715.ckpt) (36MB) | [Download](https://cv.snu.ac.kr/research/srwarp/srwarp_mrdb-908878d3.ckpt) (212MB) |


## Demo

Browse to the `src` directory and run `interactive.py` to open an interactive session.
We note that this requires a large amount of GPU memory, especially with the MRDB backbone.
We will release a memory-friendly version soon.
```bash
$ cd src
# With the MDSR backbone
$ python interactive.py --pretrained ../pretrained/srwarp_mdsr-c66c4715.ckpt
# With the MRDB backbone
$ python interactive.py --pretrained ../pretrained/srwarp_mrdb-908878d3.ckpt --backbone mrdb

# Use --img [image_path] to test your own image.
```

You can freely drag blue dots around the image for real-time interaction.
Type 2 and 3 repeatedly to compare our results with the conventional interpolation-based warping algorithm.

## Training
We are currently reorganizing the code. The training script will be released soon!

## Reference
If you find our paper and repository useful in your research, please use the following BibTeX form:
```
@inproceedings{
  title={{SRW}arp: Generalized Image Super-Resolution under Arbitrary Transformation},
  author={Son, Sanghyun and Lee, Kyoung Mu},
  booktitle={CVPR},
  year={2021}
}
```