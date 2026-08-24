# Synesthesia via Direct Latent Augmentation

### Bypassing the Decode-Encode Loop for Cross-Modal Distillation

**ECCV 2026**

**Cristian Sbrolli¹, Nicolas Michel², Matteo Matteucci¹, Toshihiko Yamasaki²**

¹ Politecnico di Milano    ² The University of Tokyo

[![Paper](https://img.shields.io/badge/arXiv-2606.08336-b31b1b.svg)](https://arxiv.org/abs/2606.08336)
[![Conference](https://img.shields.io/badge/ECCV-2026-blue.svg)](#)
[![Code](https://img.shields.io/badge/Code-PyTorch-orange.svg)](#)
[![License](https://img.shields.io/badge/License-TODO-lightgrey.svg)](#license)

> **Can we learn from generated modalities without ever decoding them?**
>
> We show that the internal latents of generative models can serve directly as privileged multimodal supervision. Our **Direct Latent Augmentation (DLA)** bypasses the expensive decode-encode loop, while **MESSy** transfers the resulting audio and 3D knowledge into a purely visual model.

<p align="center">
  <img src="assets/teaser.png" width="95%">
</p>

---

## Overview

Multimodal models can exploit complementary information from images, audio, and 3D geometry, but they come with two major limitations:

1. **paired multimodal data is scarce and expensive**, and
2. **multimodal inference is computationally costly**.

Generative models provide a way to synthesize missing modalities, but conventional pipelines first **decode** their information-rich latent representations into raw signals—such as waveforms or 3D meshes—only for a downstream model to **encode them again**.

We call this unnecessary transformation the **Decode-Encode Loop**.

<p align="center">
  <b>Generative latent → Decoder → Raw modality → Encoder → Features</b>
</p>

Instead, we ask:

> **Why decode a representation that is already useful for learning?**

Our framework directly exploits **undecoded generative latents as privileged information**.

### Direct Latent Augmentation (DLA)

**DLA** intercepts the latent representations produced by frozen cross-modal generative models before their final decoding stage.

Rather than training on decoded synthetic audio or 3D data, the multimodal teacher directly consumes these semantically dense latent representations.

This avoids decoding artifacts and substantially reduces the computational cost of synthetic multimodal augmentation.

### Multilayer Explicit Simulated Synesthesia (MESSy)

Directly forcing an image-only student to reproduce a multimodal teacher representation can interfere with the student's native visual feature space.

**MESSy** instead equips the student with lightweight predictive heads that learn to predict the teacher's auxiliary-modality representations from visual tokens at multiple layers.

At inference time, these heads and all auxiliary modalities are removed.

The resulting model receives **RGB images only**, while retaining knowledge learned from synthetic audio and 3D supervision during training.

<p align="center">
  <img src="assets/method.png" width="95%">
</p>

---

## Key Results

### Undecoded latents outperform decoded synthetic data

Training the multimodal teacher directly on generative latents consistently outperforms training on fully decoded synthetic modalities.

| Teacher input                | Imagenette2-320 | Caltech101 | ImageNet-100 |
| :--------------------------- | --------------: | ---------: | -----------: |
| Image only                   |           79.63 |      55.10 |        61.57 |
| Raw synthetic                |           88.28 |      59.08 |        65.40 |
| Raw + Perceiver              |           88.92 |      60.21 |        66.13 |
| **Generative latents (DLA)** |       **93.10** |  **66.47** |    **71.59** |

DLA improves the multimodal teacher by **+11.37 pp on Caltech101** and **+10.02 pp on ImageNet-100** compared with the image-only baseline.

### MESSy transfers multimodal knowledge to RGB-only inference

| Method                    | Imagenette2-320 | Caltech101 | ImageNet-100 |
| :------------------------ | --------------: | ---------: | -----------: |
| Image-only                |           79.63 |      55.10 |        61.57 |
| KD                        |           80.66 |      56.49 |        63.97 |
| LUGPI                     |           81.85 |      60.12 |        65.70 |
| **MESSy (ours)**          |       **84.23** |  **63.52** |    **68.54** |
| DLA Teacher (upper bound) |           93.10 |      66.47 |        71.59 |

MESSy improves the RGB-only student by **+8.42 pp on Caltech101** and **+6.97 pp on ImageNet-100**, while requiring **only visual input at inference time**.

### Scaling to ImageNet-1K

The same framework scales to the full 1.28M-image ImageNet-1K training set:

| Method            | ImageNet-1K Top-1 |
| :---------------- | ----------------: |
| Image-only        |             64.17 |
| DLA Teacher       |             71.32 |
| **MESSy Student** |         **69.05** |

The RGB-only MESSy student improves over the image baseline by **+4.88 pp**.

### Computational savings

By stopping generation before full decoding, DLA also reduces the cost of creating synthetic privileged information:

| Modality  | Generation time |            Peak VRAM |
| :-------- | --------------: | -------------------: |
| **3D**    |      **−87.7%** |    **>600 MB saved** |
| **Audio** |      **−20.6%** | **−23.8% (~2.8 GB)** |

---

## Artificial Synesthesia

A central question is whether the student merely improves its classification accuracy or actually learns representations reflecting the modalities it never observes.

We find evidence for the latter.

After MESSy distillation, the visual embedding space becomes more aligned with both **acoustic** and **3D geometric** structure—even though the student receives no audio or 3D input at inference time.

For example, acoustic supervision particularly benefits visually ambiguous classes characterized by distinctive sounds, while 3D supervision improves classes where shape and morphology provide useful discriminative cues.

We refer to this emergent cross-modal structure as **Artificial Synesthesia**.

<p align="center">
  <img src="assets/synesthesia.png" width="90%">
</p>

---

## Method

Our training pipeline consists of three main stages:

**1. Generate privileged latents**

Given an image or its caption, frozen generative models synthesize complementary modalities such as audio and 3D geometry. DLA extracts their internal latent representations **before decoding**.

**2. Train a multimodal teacher**

Visual tokens and synthetic auxiliary latents are fused through a latent resampling architecture, producing a multimodal teacher that benefits from complementary visual, acoustic, and geometric information.

**3. Distill with MESSy**

An RGB-only Vision Transformer student is trained using classification and distillation objectives together with multilayer predictive heads that reconstruct auxiliary-modality summaries from visual representations.

At test time:

```text
                    TRAINING
 RGB ────────────────┐
                     ├──► Multimodal Teacher
 RGB → Generators ──►│       (Audio + 3D latents)
                     │
                     └── MESSy ──► RGB Student


                    INFERENCE
 RGB ───────────────────────────► RGB Student ──► Prediction
```

No generative model, audio, 3D data, multimodal teacher, or MESSy prediction head is required during inference.

---

## Installation

> **TODO:** Update this section to match the final repository environment.

```bash
git clone https://github.com/<username>/<repository>.git
cd <repository>

conda create -n messy python=3.XX
conda activate messy

pip install -r requirements.txt
```

---

## Data Preparation

We evaluate on:

* **Imagenette2-320**
* **Caltech101**
* **ImageNet-100**
* **ImageNet-1K**

The overall data preparation pipeline is:

```text
Images
  │
  ├──► Visual training data
  │
  └──► Cross-modal generators
             │
             ├──► Audio latents
             └──► 3D latents
```

> **TODO:** Add dataset download/preprocessing commands.

```bash
# Example
python scripts/prepare_data.py \
    --dataset <dataset> \
    --output <data_directory>
```

---

## Generating DLA Features

DLA extracts the intermediate representations of the frozen generative models instead of fully decoding the generated modalities.

> **TODO:** Replace with the repository's actual command.

```bash
python scripts/generate_latents.py \
    --dataset <dataset> \
    --modality audio 3d \
    --output <latent_directory>
```

The generated latents can then be reused across teacher/student training runs without repeatedly executing the generative models.

---

## Training

### Multimodal DLA Teacher

```bash
python train_teacher.py \
    --config configs/<dataset>/teacher_dla.yaml
```

### MESSy Student

```bash
python train_student.py \
    --config configs/<dataset>/messy.yaml \
    --teacher <teacher_checkpoint>
```

> **TODO:** Replace these commands with the final CLI/configuration structure.

---

## Evaluation

```bash
python evaluate.py \
    --config configs/<dataset>/messy.yaml \
    --checkpoint <checkpoint>
```

The final MESSy student is **unimodal**: evaluation requires RGB images only.

---

## Repository Structure

> **TODO:** Replace with the actual repository layout.

```text
.
├── assets/                 # README figures
├── configs/                # Experiment configurations
├── data/                   # Dataset utilities
├── models/
│   ├── teacher/            # DLA multimodal teacher
│   ├── student/            # RGB-only student
│   └── messy/              # MESSy prediction heads
├── generation/             # Audio / 3D latent extraction
├── scripts/                # Data and experiment scripts
├── train_teacher.py
├── train_student.py
├── evaluate.py
└── README.md
```

---

## Citation

If you find this work useful, please consider citing it:

```bibtex
@inproceedings{sbrolli2026synesthesia,
  title     = {Synesthesia via Direct Latent Augmentation:
               Bypassing the Decode-Encode Loop for Cross-Modal Distillation},
  author    = {Sbrolli, Cristian and
               Michel, Nicolas and
               Matteucci, Matteo and
               Yamasaki, Toshihiko},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

ArXiv:

```bibtex
@article{sbrolli2026synesthesia,
  title   = {Synesthesia via Direct Latent Augmentation:
             Bypassing the Decode-Encode Loop for Cross-Modal Distillation},
  author  = {Sbrolli, Cristian and
             Michel, Nicolas and
             Matteucci, Matteo and
             Yamasaki, Toshihiko},
  journal = {arXiv preprint arXiv:2606.08336},
  year    = {2026}
}
```

---

## Acknowledgements

This work was conducted at **Politecnico di Milano** and **The University of Tokyo**.

We thank the authors and maintainers of the datasets, generative models, and open-source libraries that made this work possible.

---

## License

> **TODO:** Add the repository license and update the badge at the top of this README.

---

## Contact

For questions regarding the paper or code, please open a GitHub issue or contact:

**Cristian Sbrolli**
Politecnico di Milano
`cristian.sbrolli@polimi.it`
