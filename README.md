<p align="center">
<h1 align="center"<strong>PhysGraph: Physically-Grounded Graph-Transformer Policies for Bimanual Dexterous Hand–Tool–Object Manipulation</strong></h1>
  <p align="center">
    <a href='https://blarklee.github.io/' target='_blank'>Runfa Blark Li</a>
    ·
    <a href='' target='_blank'>David Kim</a>
    ·
    <a href='' target='_blank'>Xinshuang Liu</a>
    ·
    <a href='' target='_blank'>Dwait Bhatt</a>
    ·
    <a href='' target='_blank'>Keito Suzuki</a>
    ·
    <a href='' target='_blank'>Nikola Raicevic</a>
    ·
    <a href='' target='_blank'>Xin Lin</a>
    ·
    <a href='' target='_blank'>Ki Myung Brian Lee</a><br>
    ·
    <a href='' target='_blank'>Nikolay Atanasov</a>
    ·
    <a href='' target='_blank'>Truong Nguyen</a>
    <br>
    UC San Diego
    <br>
  </p>
</p>
<p align="center">
  <a href=''>
    <img src='https://img.shields.io/badge/arXiv-2503.19901-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a>
  <a href=''>
    <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'></a>
  <a href='https://blarklee.github.io/PhysGraph_website_official/'>
    <img src='https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=green'></a>
</p>


## 📹 Demo
<p align="center">
    <img src="assets/artimano_all_labeled.gif" align="center" width=60% >
    <br>
</p>
While recent learning-based approaches have made substantial progress on dexterous manipulation, bimanual tool-use remains particularly challenging. PhysGraph significantly outperforms SOTA baseline on challenging bimanual tool-use tasks in success rate and motion fidelity, supports zero-shot generalization to unseen tool/object in different tasks, and is embodiment-agnostic to popular robotic dex-hands (Shadow, Allegro, Inspire)


<!-- teaser image -->
## 🏠 Overview
<p align="center">
    <img src="assets/pipeline.png" alt="teaser" width="100%">
</p>
PhysGraph is a physically-grounded graph-transformer policy designed explicitly for bimanual tool-object manipulation. Rather than flattening the state as concurrent dexhand manipulations, we formulate the bimanual system as a dynamic kinematic graph, where nodes represent individual rigid bodies (links, tools, objects) and edges represent physical couplings. Our approach introduces two key innovations: (i) We propose a per-link tokenization strategy. Instead of pooling states into a global embedding, we process each link’s multi-modal states as a distinct token, preserving fine-grained local properties. (ii) Most crucially, we introduce a novel Physically-Grounded Bias Generator. Unlike generic Graph Transformers (e.g., Graphormer) that utilizes abstract static graph distances for chemical bonds, we inject a dynamic learning-based head-specific composite bias directly into the attention mechanism. The composite bias includes Spatial Bias (kinematic chain distance), Dynamic Edge Bias (static/dynamic contact states), Geometric Bias (Cartesian proximity), and Anatomical Priors (serial/synergies kinematics), which enable our policy to explicitly reason about the physical connectivity and contact logic, focusing attention on contacting fingers or coordinated joints, thereby improving the reliability and precision.

## 📑 Table of Contents
1. [Installation](#Installation)
2. [Prerequisites](#Prerequisites)
3. [Usage](#usage)
3. [Citation](#Citation)
5. [Acknowledgement](#acknowledgement)

---

## 🛠️ Installation
<a id="Installation"></a>

<details>
<summary>Steps:</summary>

1. Clone the repository and initialize submodules:
    ```bash
    git clone https://github.com/BlarkLee/PhysGraph.git
    git submodule init && git submodule update
    ```
2. Create a virtual environment named `physgraph` with Python 3.8. Note that IsaacGym only supports Python versions up to 3.8.
    ```bash
    conda create -y -n physgraph python=3.8
    conda activate physgraph
    pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
    ```
3. Download IsaacGym Preview 4 from the [official website](https://developer.nvidia.com/isaac-gym) and follow the installation instructions in the documentation. Test the installation by running an example script, such as `joint_monkey.py`, located in the `python/examples` directory.
4. Install additional dependencies.
    ```bash
    pip install git+https://github.com/ZhengyiLuo/smplx.git
    pip install git+https://github.com/KailinLi/bps_torch.git
    pip install fvcore~=0.1.5
    pip install --no-index --no-cache-dir pytorch3d==0.7.3 -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt1131/download.html
    pip install -r requirements.txt
    pip install -e . # include the current directory in the Python path. Or use: `export PYTHONPATH=.:$PYTHONPATH`
    pip install numpy==1.23.0 # downgrade numpy to 1.23.0 to avoid compatibility issues
    ```

</details>

---

## 📋 Prerequisites
<a id="Prerequisites"></a>

We follow the prerequisit of [ManipTrans](https://maniptrans.github.io/) to prepare the dataset.

<details>
<summary>Steps:</summary>

### `OakInk-V2` dataset
1. Download the OakInk-V2 dataset from its [official website](https://oakink.net/v2/) and extract it into the `data/OakInk-v2` directory. (You may skip downloading images; only annotated motion data is required.)

2. For each object mesh in `data/OakInk-v2/object_preview/align_ds`, generate the [COACD](https://github.com/SarahWeiii/CoACD) file by running:
    ```bash
    python physgraph_envs/lib/utils/coacd_process.py -i data/OakInk-v2/object_preview/align_ds/xx/xx.obj -o data/OakInk-v2/coacd_object_preview/align_ds/xx/xx.obj --max-convex-hull 32 --seed 1 -mi 2000 -md 5 -t 0.07
    # Or, if you have the ply file, you can use:
    python physgraph_envs/lib/utils/coacd_process.py -i data/OakInk-v2/object_preview/align_ds/xx/xx.ply -o data/OakInk-v2/coacd_object_preview/align_ds/xx/xx.ply --max-convex-hull 32 --seed 1 -mi 2000 -md 5 -t 0.07
    ```
3. For each generated COACD file in `data/OakInk-v2/coacd_object_preview/align_ds`, create a corresponding URDF file based on `assets/obj_urdf_example.urdf`.

4. Download the `body_upper_idx.pt` file from the [official website](https://oakink.net/v2/) and place it in the `data/smplx_extra` directory.

5. The directory structure should look like this:
    ```
    data
    ├── smplx_extra
    │   └── body_upper_idx.pt
    └── OakInk-v2
        ├── anno_preview
        ├── coacd_object_preview
        ├── data
        ├── object_preview
        └── program
    ```

</details>


### BiManual Tool-Use Policies

1. **Preprocessing**

    Preprocess data for both hands:
    ```bash
    # for Artimano Hand
    python main/dataset/mano2dexhand.py --data_idx 083f7@0 --side right --dexhand artimano --headless --iter 7000
    python main/dataset/mano2dexhand.py --data_idx 083f7@0 --side left --dexhand artimano --headless --iter 7000
    # for other hands, just replace `Artimano` with the corresponding hand name. Candidate hand names are `Shadow`, `Inspire`, `Allegro`. 
    ```
    Regarding `data_idx` of OakInk V2, for example, `083f7@0` refers to the primitive task indexed at `0` in the sequence labeled `scene_01__A001++seq__083f7a577484ba7929a9__2023-04-27-19-25-24` (for simplification, we only use the first 5 digits of the hash code).

2. **Training**
  Train bi-manual policies:
    ```bash
    python main/rl/train.py task=ResDexHand dexhand=artimano side=BiH headless=true num_envs=4096 learning_rate=2e-4 test=false randomStateInit=true dataIndices=[083f7@0] early_stop_epochs=10000 actionsMovingAverage=0.4 experiment=083f7@0_artimano
    ```
    The `early_stop_epochs` parameter can be adjusted based on the task complexity.

3. **Test**
  Test the bi-manual policy:
    ```bash
    python main/rl/train.py task=ResDexHand dexhand=inspire side=BiH headless=false num_envs=4 learning_rate=2e-4 test=true randomStateInit=false dataIndices=[083f7@0] actionsMovingAverage=0.4 checkpoint=runs/083f7@0_artimano__xxxxxx/nn/083f7@0_artimano.pth
    ```
---



## Citation
<a id="Citation"></a>
```

```

## 🙏 Acknowledgement
<a id="acknowledgement"></a>
We thank [OakInk V2](https://oakink.net/v2/) for the dataloader and [ManipTrans](https://maniptrans.github.io/) for the training pipeline used in this work.


