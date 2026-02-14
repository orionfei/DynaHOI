<h1 align="center">DynaHOI: Benchmarking Hand-Object Interaction for Dynamic Target</h1>

<div align='center'>
    <a href='https://scholar.google.com/citations?user=VBb03aoAAAAJ' target='_blank'><strong>BoCheng Hu</strong></a><sup> 1</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=tGTa-EAAAAAJ' target='_blank'><strong>Zhonghan Zhao</strong></a><sup> 1,2</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=tS287loAAAAJ' target='_blank'><strong>Kaiyue Zhou</strong></a><sup> 3</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=lFbTT5AAAAAJ' target='_blank'><strong>Hongwei Wang</strong></a><sup> 1</sup>&thinsp;
    <a href='https://scholar.google.com/citations?user=GhsXNiwAAAAJ' target='_blank'><strong>Gaoang Wang</strong></a><sup> 1</sup>&thinsp;
</div>

<div align='center'>
    <sup>1 </sup>Zhejiang University&ensp;  <sup>2 </sup>Shanghai AI Lab&ensp;  <sup>3 </sup>Chengdu Minto Technology Co., Ltd.&ensp; 
</div>

<div align="center">


[![arXiv](https://img.shields.io/badge/arXiv-2602.11919-b31b1b.svg)](https://arxiv.org/abs/2602.11919)
[![Dataset](https://img.shields.io/badge/ModelScope-Dataset-624AFF?logo=modelscope)](https://modelscope.cn/datasets/brandonHuu/DynaHOI-12M)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

</div>

## **News 📣**

- ✅ We’ve released the **current evaluation / inference code** used in our experiments.
- **Feb 13, 2026**: 🔥 We open-sourced **ObAct weights**.
- **Feb 11, 2026**: 📄 Our paper is now public.




## **TODO ✅**

- [x] Paper release
- [x] Model weights release (ObAct)
- [ ] Dataset release
- [ ] Training code release
- [ ] Release dynamic HOI trajectory collection scripts & tutorial



## **Setup ⚙️**

### **1) Unity Environment (Simulator) 🎮**

This project uses a **Unity simulator** for two purposes:

- **Dynamic HOI trajectory collection**
- **Online (closed-loop) evaluation** with a policy running in Python

**Steps**

1. Install **Unity Hub**, then install Unity **2022.3.58f1c1**.
2. Download the full Unity project (C# source + required assets) from [Google Drive](https://drive.google.com/drive/folders/10BlLxE14uEevkgVLK8Px3C7oMZAoaqAs?usp=sharing), Unity Controller folder. 
3. Open the project in Unity Hub, then double-click the `Scenes/Dynamic` scene file to load the predefined scene in the Unity Editor.
That’s it — your simulator should be ready ✅

------

### **2) Python Environment (Conda) 🐍**

```
conda create -n gr00t python=3.10
conda activate gr00t
pip install --upgrade setuptools
pip install -e .[base]
pip install --no-build-isolation flash-attn==2.7.1.post4
```

> We also recommend using **uv** for faster & cleaner dependency management.
> You can follow NVIDIA’s official setup guide [here](https://github.com/NVIDIA/Isaac-GR00T?tab=readme-ov-file#set-up-the-environment).



## **Online Evaluation 🧪 (Closed-loop)**

### **Overview 🔁**

- **Python** runs the policy inference and acts as the **server** (actively connects).
- **Unity** executes action logic and acts as the **client** (waits for Python connection).

Since **inference** and **simulation** often run on different machines (e.g., Unity on your local desktop, inference on a remote GPU server), you’ll likely need **port forwarding**.

### **Steps**

1. **SSH port forwarding** (remote ↔ local):

```bash
ssh -L 8765:127.0.0.1:8765 <your_remote_server>
```

2. Download **ObAct weights** from [Google Drive](https://drive.google.com/drive/folders/10BlLxE14uEevkgVLK8Px3C7oMZAoaqAs?usp=sharing).Then update `model_path` in the inference script:

    - scripts/eval_policy_baseline.py → **ObAct**.

    - scripts/eval_policy_our.py → **vanilla GR00T-N1.5**.

3. Run the policy-side inference:

```bash
python scripts/eval_policy_baseline.py
# or
python scripts/eval_policy_our.py
```

4. In Unity, click **Play** ▶️ in the GUI.

You should see real-time interaction in the **Game** window.



## **Fine-tuning 🛠️**

Script mapping:

- **Vanilla GR00T-N1.5 fine-tuning**:

  scripts/gr00t_finetune_our_18dim.py

- **ObAct baseline fine-tuning**:

  scripts/gr00t_finetune_our_18dim_baseline.py

Before running, configure key args like:

- dataset_path
- output_dir
- base_model_path
- (and any cluster / logging configs you use)

We recommend launching with setsid + torchrun:

```bash
setsid torchrun --standalone --nproc_per_node=4 scripts/gr00t_finetune_our_18dim.py > nohup_logs/xxx.log 2>&1 &
```



## **Dynamic HOI Trajectory Collection 📹🤖**

We provide:

- **atomic motion generation scripts**
- **automated grasping trajectory collection scripts**

in the [Google Drive](https://drive.google.com/drive/folders/10BlLxE14uEevkgVLK8Px3C7oMZAoaqAs?usp=sharing), Unity Controller folder:

A detailed tutorial is coming soon, including:

- how to generate **large-scale, valid motion parameters**
- how to collect data via **WebSocket** while **compressing videos**
- recommended Unity simulator settings (scene setup, playback rate, resolution, logging, etc.)



## **Citation 📚**

```
@article{hu2026dynahoi,
  title={DynaHOI: Benchmarking Hand-Object Interaction for Dynamic Target},
  author={Bocheng Hu and Zhonghan zhao and Kaiyue zhou and Hongwei Wang and Gaoang Wang},
  journal={arXiv preprint arXiv:2602.11919},
  year={2026}
}
```



## **Acknowledgements 🙌✨**

Huge thanks to these excellent open-source projects that made this work possible:

[GR00T](https://github.com/NVIDIA/Isaac-GR00T), [OpenVLA](https://github.com/openvla/openvla), [UP-VLA](https://github.com/CladernyJorn/UP-VLA), [openpi](https://github.com/Physical-Intelligence/openpi) and [lerobot](https://github.com/huggingface/lerobot). 