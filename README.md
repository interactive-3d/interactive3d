[comment]: <> (# Interactive3D: Create What You Want by Interactive 3D Generation)

<p align="center" width="100%">
<img src="assets/interactive3d.png"  width="30%" height="30%">
</p>

<p align="center">

  <h1 align="center">Interactive3D: Create What You Want by Interactive 3D Generation</h1>
  <p align="center">
    <a href='https://scholar.google.com/citations?user=Pee4FRsAAAAJ&hl=en'><strong>Shaocong Dong<sup>*</sup></strong></a>
    ·
    <a href='https://dinglihe.github.io/'><strong>Lihe Ding<sup>*</sup></strong></a>
    ·
    <a href='https://dblp.org/pid/135/5933.html'><strong>Zhanpeng Huang</strong></a>
    ·
    <a href='https://dblp.org/pid/54/8616.html'><strong>Zibin Wang</strong></a>
    ·
    <a href='https://tianfan.info/'><strong>Tianfan Xue</strong></a>
    ·
    <a href='https://www.danxurgb.net/'><strong>Dan Xu</strong></a>
  </p>

<!--
<div align="center">
    <a href='https://scholar.google.com/citations?user=Pee4FRsAAAAJ&hl=en' target='_blank'>Shaocong Dong<sup>1*</sup></a>&emsp;
    <a href='https://dinglihe.github.io/' target='_blank'>Lihe Ding<sup>2,4*</sup></a>&emsp;
    <a>Zhanpeng Huang<sup>3</sup></a>&emsp;
    <a>Zibin Wang<sup>3</sup></a>&emsp;
    </br>
    <a href='https://tianfan.info/'> Tianfan Xue<sup>2</sup></a>
    <a href='https://www.danxurgb.net/'> Dan Xu<sup>1,†</sup></a>&emsp;
</div>
<div>

<div align="center">
    <sup>1</sup>
    <a> The Hong Kong University of Science and Technology</a>&emsp;
    </br>
    <sup>2</sup> <a>The Chinese University of Hong Kong</a>
    </br>
    <sup>3</sup> <a> SenseTime </a>
    <sup>4</sup> <a> Shanghai AI Laboratory </a></br>
    <sup>*</sup> Equal Contribution&emsp;
    <sup>†</sup> Corresponding Author&emsp;
</div>
-->

[comment]: <> (  <h2 align="center">PAPER</h2>)
  <h3 align="center"><a href="https://arxiv.org/abs/2404.16510">Paper</a> | <a href="https://interactive-3d.github.io/">Project Page</a> | <a href="https://youtu.be/ZYSOonigv3s">Video</a></h3>
  <div align="center"></div>

<p align="left" width="100%">
<img src="assets/results.png"  width="100%" height="100%">
</p>

---

## NEWS
- 🔥 Interactive3D got accepted by CVPR24.
- release the refinement code based on threestudio.
- support Pixart-alpha as 2D guidance.
- we are developing a user-friendly interface and will release the stage-I implementation together.


## Abstract 
<p align="left" width="100%">
<img src="assets/arc.png"  width="100%" height="100%">
</p>
Interactive3D is an innovative framework for interactive 3D generation that grants users precise control over the generative process through extensive 3D interaction capabilities. Interactive3D is constructed in two cascading stages, utilizing distinct 3D representations. The first stage employs Gaussian Splatting for direct user interaction, allowing modifications and guidance of the generative direction at any intermediate step through (i) Adding and Removing components, (ii) Deformable and Rigid Dragging, (iii) Geometric Transformations, and (iv) Semantic Editing. Subsequently, the Gaussian splats are transformed into InstantNGP. We introduce a novel (v) Interactive Hash Refinement module to further add details and extract the geometry in the second stage.

## Installation
```sh
conda create -n inter python=3.9
conda activate inter

# Newer pip versions, e.g. pip-23.x, can be much faster than old versions, e.g. pip-20.x.
# For instance, it caches the wheels of git packages to avoid unnecessarily rebuilding them later.
python3 -m pip install --upgrade pip
```
- we have tested cuda 11.7 and torch 2.0.1, but other versions also work fine.
```sh
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```
- Install dependencies:
```sh
pip install -r requirements.txt
# install terminfo
sudo apt-get install libncurses5-dev libncursesw5-dev
```
- [Optional] Install gsgen, more information can be found [here](https://github.com/gsgen3d/gsgen)
```sh
git clone https://github.com/gsgen3d/gsgen.git
cd gsgen/gs
./build.sh
```
note: refer to [threestudio](https://github.com/threestudio-project/threestudio) for some installation issues.

## Quick Start
- Start from a given gaussian splatting 
result or generate one by:
```sh
# the checkpoint will be saved to gsgen folder
cd ../../gsgen
python main.py --config-name=base prompt.prompt="<prompt>"
```
- convert gaussian splatting to NeRF:
```sh
# feel free to adjust the hparams in the config
cd ..
python launch.py --config configs/fit_gs.yaml --train --gpu 0 system.prompt_processor.prompt="your prompt" trainer.max_steps=1800
```

- [Optional] refine the geometry:
```sh
# since we find that the text-to-img guidance sometimes changes the content, we use img-to-img guidance now
# feel free to adjust the parameters sunch as training steps
python launch.py --config configs/geo_refine.yaml --train --gpu 0 system.prompt_processor.prompt="your prompt" resume=path_to_your/ckpts/last.ckpt trainer.max_steps=20000 system.init_type='gsgen' system.only_super=True

# after geometry refinement, run the following script to get a standard nerf representation for further usage
python launch.py --config configs/post_geo_refine.yaml --train --gpu 1 system.prompt_processor.prompt="your prompt" trainer.max_steps=3000
```

- refine the interested region

note: export the terminfo path if you are using tmux
```sh
export TERMINFO=path_to/terminfo
```

```sh
# 1. save occ for interested region selection, the occ grid will be saved to debug_data
python launch.py --config configs/interested_refine.yaml --train --gpu 0 system.prompt_processor.prompt="your prompt" resume=path_to_your/ckpts/last.ckpt trainer.max_steps=20000 system.init_type='threestudio' system.only_super=True system.renderer.save_occ=True
# 2. use the utils we provided to select interested regions on your local machine, then upload the selected index npy to debug_data (more adavanced version including SAM will be released together with the interface)
python utils/region_select_tool.py
```

```sh
# 3. we provide a simple command controller to adjust camera now, which will be upgraded in the final interface. 
python launch.py --config configs/interested_refine.yaml --train --gpu 0 system.prompt_processor.prompt="your prompt" resume=path_to_your/ckpts/last.ckpt trainer.max_steps=20000 system.init_type='threestudio' system.only_super=True

# optional: you can also try pixart-alpha as guidance by:
python launch.py --config configs/interested_refine_pixart.yaml --train --gpu 0 system.prompt_processor.prompt="your prompt" resume=path_to_your/ckpts/last.ckpt trainer.max_steps=10000 system.init_type='threestudio' system.only_super=True
```
we will update more detailed illustrations soon.

## Acknowledgement
Interactive3D is built on many amazing research works, thanks a lot to all the authors for sharing! Thank [Yiyuan](https://github.com/invictus717) for the valuable discussion and paper refinement.
- [gsgen](https://github.com/gsgen3d/gsgen) and [instant-ngp](https://github.com/NVlabs/instant-ngp)
- [threestudio](https://github.com/threestudio-project/threestudio)


## Citation
If the paper and the code are helpful for your research, please kindly cite:
```
@article{dong2024interactive3d,
  title={Interactive3D: Create What You Want by Interactive 3D Generation},
  author={Dong, Shaocong and Ding, Lihe and Huang, Zhanpeng and Wang, Zibin and Xue, Tianfan and Xu, Dan},
  journal={arXiv preprint arXiv:2404.16510},
  year={2024}
}
```
