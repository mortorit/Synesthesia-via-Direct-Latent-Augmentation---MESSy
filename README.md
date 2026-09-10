<h1 align="center">Synesthesia via Direct Latent Augmentation</h1>

<h3 align="center">Bypassing the Decode-Encode Loop for Cross-Modal Distillation</h3>

<p align="center">
  <b>ECCV 2026</b>
</p>

<p align="center">
  <b>Cristian Sbrolli¹, Nicolas Michel², Matteo Matteucci¹, Toshihiko Yamasaki²</b><br>
  ¹ Politecnico di Milano &nbsp;&nbsp; ² The University of Tokyo
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2606.08336">
    <img src="https://img.shields.io/badge/arXiv-2606.08336-b31b1b.svg" alt="arXiv">
  </a>
  <img src="https://img.shields.io/badge/ECCV-2026-1f6feb.svg" alt="ECCV 2026">
  <a href="https://github.com/mortorit/Synesthesia-via-Direct-Latent-Augmentation---MESSy">
    <img src="https://img.shields.io/badge/Code-PyTorch-EE4C2C.svg" alt="PyTorch">
  </a>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#release-scope">Code Release</a> •
  <a href="#installation">Installation</a> •
  <a href="#data-and-latents">Data & Latents</a> •
  <a href="#training">Training</a> •
  <a href="#citation">Citation</a>
</p>

---

> **Can we learn from generated modalities without ever decoding them?**
>
> We show that the internal latents of generative models can serve directly as privileged multimodal supervision. **Direct Latent Augmentation (DLA)** bypasses the expensive decode-encode loop, while **MESSy** transfers the resulting audio and 3D knowledge into a purely visual model.

## Overview

Multimodal models can exploit complementary information from images, audio, and 3D geometry, but they face two practical limitations:

1. **Paired multimodal data is scarce and expensive.**
2. **Multimodal inference is computationally costly.**

Generative models offer a way to synthesize missing modalities. Conventional pipelines, however, first decode their information-rich latent representations into raw signals—such as waveforms or 3D geometry—only for a downstream model to encode them again.

We call this unnecessary transformation the **Decode-Encode Loop**:

```text
Generative latent → Decoder → Raw modality → Encoder → Features
```

Instead, we ask:

> **Why decode a representation that is already useful for learning?**

Our framework directly exploits **undecoded generative latents as privileged information**.

### Direct Latent Augmentation (DLA)

**DLA** intercepts the latent representations produced by frozen cross-modal generative models before their final decoding stage.

Rather than training on fully decoded synthetic audio or 3D data, the multimodal teacher consumes these semantically dense latent representations directly.

This avoids unnecessary decoding, reduces synthetic-data generation cost, and preserves information already available inside the generative model.

### Multilayer Explicit Simulated Synesthesia (MESSy)

Directly forcing an image-only student to reproduce a multimodal teacher representation can interfere with the student's native visual feature space.

**MESSy** instead introduces lightweight predictive heads at multiple transformer layers. These heads learn to predict the teacher's auxiliary-modality representations from visual features while leaving the student's visual representation free to develop naturally.

During training:

```text
 RGB image ────────────────────────────┐
                                      │
 Audio latent ──┐                     ├──► Multimodal DLA Teacher
                ├──► privileged input │
 3D latent ─────┘                     │
                                      │
 RGB image ───────────────────────────┴──► MESSy ──► RGB Student
```

At inference:

```text
 RGB image ───────────────────────────────► RGB Student ──► Prediction
```

The auxiliary latents, multimodal teacher, and MESSy prediction heads are **training-time only**.

---

## Key Results

### Undecoded latents outperform decoded synthetic data

Training the multimodal teacher directly on generative latents consistently outperforms both image-only training and training on decoded synthetic modalities.

| Teacher input                | Imagenette2-320 | Caltech101 | ImageNet-100 |
| :--------------------------- | --------------: | ---------: | -----------: |
| Image only                   |           79.63 |      55.10 |        61.57 |
| Raw synthetic                |           88.28 |      59.08 |        65.40 |
| Raw + Perceiver              |           88.92 |      60.21 |        66.13 |
| **Generative latents (DLA)** |       **93.10** |  **66.47** |    **71.59** |

DLA improves the multimodal teacher by **+11.37 pp on Caltech101** and **+10.02 pp on ImageNet-100** compared with the image-only baseline.

### MESSy transfers multimodal knowledge to RGB-only inference

| Method           | Imagenette2-320 | Caltech101 | ImageNet-100 |
| :--------------- | --------------: | ---------: | -----------: |
| Image-only       |           79.63 |      55.10 |        61.57 |
| KD               |           80.66 |      56.49 |        63.97 |
| LUGPI            |           81.85 |      60.12 |        65.70 |
| **MESSy (ours)** |       **84.23** |  **63.52** |    **68.54** |
| DLA Teacher      |           93.10 |      66.47 |        71.59 |

The RGB-only MESSy student improves over the image baseline by **+8.42 pp on Caltech101** and **+6.97 pp on ImageNet-100**.

### Scaling to ImageNet-1K

The framework also scales to the full ImageNet-1K training set:

| Method            | ImageNet-1K Top-1 |
| :---------------- | ----------------: |
| Image-only        |             64.17 |
| DLA Teacher       |             71.32 |
| **MESSy Student** |         **69.05** |

The RGB-only MESSy student improves over the image baseline by **+4.88 pp**.

### Computational savings

Stopping generation before full decoding also reduces the cost of producing synthetic privileged information:

| Modality  | Generation time |            Peak VRAM |
| :-------- | --------------: | -------------------: |
| **3D**    |      **−87.7%** |    **>600 MB saved** |
| **Audio** |      **−20.6%** | **−23.8% (~2.8 GB)** |

---

## Artificial Synesthesia

A central question is whether the student merely becomes a better classifier or actually learns representations reflecting modalities it never observes at inference time.

We find evidence for the latter.

After MESSy distillation, the visual representation becomes more aligned with both **acoustic** and **3D geometric** structure—even though the final student receives only RGB images.

Acoustic supervision is particularly useful for visually ambiguous classes associated with distinctive sounds, while 3D supervision helps when shape and morphology provide complementary discriminative cues.

We refer to this emergent cross-modal structure as **Artificial Synesthesia**.

---

## Release Scope

This repository is a **minimal training release** for DLA and MESSy.

It contains:

* `messy/teacher.py` — training of the multimodal DLA teacher from RGB images and precomputed audio/3D latents.
* `messy/distill_messy.py` — distillation of the multimodal teacher into an RGB-only student using logit distillation and multilayer MESSy prediction.
* `docs/latent_format.md` — expected latent directory structure and tensor formats.
* `requirements.txt` — Python dependencies required by the released training code.

### What is not bundled

The generative-model pipelines used to extract the original audio and 3D latents are **not included in this release**.

Those pipelines depend on large external checkpoints and evolving third-party generative-model implementations. The released training code therefore starts from **precomputed `.pt` latent files**.

Datasets and external generative-model checkpoints are likewise not redistributed.

This separation makes the training release smaller and allows exported latent representations to be reused across experiments without repeatedly running the generators.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/mortorit/Synesthesia-via-Direct-Latent-Augmentation---MESSy.git
cd Synesthesia-via-Direct-Latent-Augmentation---MESSy
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The release requires PyTorch 2.1 or newer. Depending on your CUDA configuration, you may prefer to install the appropriate PyTorch build before installing the remaining requirements.

Experiment tracking through Weights & Biases is supported but not required; training runs with W&B disabled by default.

---

## Data and Latents

The training scripts expect three aligned roots:

* RGB images
* continuous 3D latents
* discrete audio latents

All modalities must preserve the same relative class/sample structure.

A typical dataset therefore looks like:

```text
data/imagenet100/
├── images/
│   ├── train/
│   │   └── <class>/
│   │       └── <sample>.jpg
│   └── val/
│       └── <class>/
│           └── <sample>.jpg
│
├── pc_latents/
│   ├── train/
│   │   └── <class>/
│   │       └── <sample>.pt
│   └── val/
│       └── <class>/
│           └── <sample>.pt
│
└── audio_latents/
    ├── train/
    │   └── <class>/
    │       └── <sample>.pt
    └── val/
        └── <class>/
            └── <sample>.pt
```

For example,

```text
images/train/n01440764/example.jpg
pc_latents/train/n01440764/example.pt
audio_latents/train/n01440764/example.pt
```

form one aligned multimodal sample.

The loaders match modalities by their **relative path**, not by a separate metadata file.

### Latent formats

| Modality             | Expected format                                             | Default                     |
| :------------------- | :---------------------------------------------------------- | :-------------------------- |
| **3D**               | floating-point `[tokens, channels]` or `[channels, tokens]` | 64 channels, 256 tokens     |
| **Audio**            | integer `[codebooks, time]`                                 | 4 codebooks, 250 time steps |
| **Audio vocabulary** | discrete token IDs                                          | 2048 entries                |

3D sequences longer than the configured token count are truncated and shorter sequences are zero-padded.

Audio sequences are similarly padded or truncated according to the configured duration and frame rate. With the defaults of 5 seconds and 50 Hz, the expected length is **250 time steps**.

See [`docs/latent_format.md`](docs/latent_format.md) for the concise format specification.

---

## Training

The released pipeline has two stages:

```text
Precomputed DLA latents
        │
        ▼
┌──────────────────────┐
│ Multimodal Teacher   │
│ RGB + Audio + 3D     │
└──────────┬───────────┘
           │
           │ logits + intermediate modality representations
           ▼
┌──────────────────────┐
│ MESSy Distillation   │
│ RGB-only Student     │
└──────────┬───────────┘
           │
           ▼
     RGB-only model
```

### 1. Train the DLA teacher

```bash
python messy/teacher.py \
  --image_dataset_path data/imagenet100/images \
  --pc_dataset_path data/imagenet100/pc_latents \
  --audio_dataset_path data/imagenet100/audio_latents \
  --output_dir checkpoints/teacher \
  --val_split val \
  --wandb_mode disabled
```

The default teacher uses:

* RGB images
* continuous 3D latents
* discrete audio latents
* ViT-B/16
* Perceiver-style token resampling for the auxiliary modalities
* modality dropout during training

The best checkpoint is stored as:

```text
LatentViT_Resampled_IPA_sd42_best.pt
```

where `I`, `P`, and `A` denote image, point-cloud/3D, and audio inputs.

The saved checkpoint contains the model state, configuration, class names, and best validation accuracy.

#### W&B tracking

To enable online experiment tracking:

```bash
python messy/teacher.py \
  --image_dataset_path data/imagenet100/images \
  --pc_dataset_path data/imagenet100/pc_latents \
  --audio_dataset_path data/imagenet100/audio_latents \
  --output_dir checkpoints/teacher \
  --wandb_mode online \
  --wandb_project <project>
```

Add `--wandb_entity <entity>` when required by your W&B setup.

### 2. Train the MESSy student

Use the checkpoint produced by the teacher stage:

```bash
python messy/distill_messy.py \
  --image_dataset_path data/imagenet100/images \
  --pc_dataset_path data/imagenet100/pc_latents \
  --audio_dataset_path data/imagenet100/audio_latents \
  --teacher_checkpoint checkpoints/teacher/LatentViT_Resampled_IPA_sd42_best.pt \
  --output_dir checkpoints/messy \
  --val_split val
```

MESSy combines standard classification/distillation with auxiliary predictive objectives at multiple transformer layers.

By default, the prediction heads operate at ViT-B/16 layers:

```text
9, 10, 11
```

At each selected layer, the RGB student's representation is used to predict pooled teacher representations associated with the **audio** and **3D** tokens.

The best student is saved as:

```text
checkpoints/messy/messy_student_best.pt
```

---

## RGB-Only Inference

The central purpose of MESSy is to transfer multimodal information **without introducing multimodal inference requirements**.

Once distillation is complete, deployment requires only:

```text
messy_student_best.pt
+
RGB images
```

The following components can be discarded:

```text
✗ audio latents
✗ 3D latents
✗ generative models
✗ multimodal teacher
✗ MESSy prediction heads
```

The saved student checkpoint contains its `state_dict`, image-only model configuration, class names, and validation accuracy.

A dedicated inference/evaluation CLI is not included in this minimal release; the student checkpoint is intended to be loaded into the user's evaluation or deployment pipeline.

---

## Repository Structure

```text
.
├── docs/
│   └── latent_format.md       # Expected latent paths and tensor shapes
│
├── messy/
│   ├── __init__.py
│   ├── distill_messy.py       # MESSy teacher → RGB student distillation
│   └── teacher.py             # Multimodal DLA teacher training
│
├── .gitignore
├── README.md
└── requirements.txt
```

Unlike the complete experimental pipeline used for the paper, this repository does **not** contain separate generator, preprocessing, configuration, or evaluation directories.

---

## Implementation Details

The multimodal teacher projects each modality into a shared transformer embedding space.

For auxiliary modalities, long latent sequences are compressed with a **Perceiver-style resampler**:

```text
3D latent ──► latent adapter ──► token resampler ──┐
                                                   │
RGB patches ───────────────────────────────────────┼──► Transformer
                                                   │
Audio codes ─► latent adapter ──► token resampler ─┘
```

With the default ViT-B/16 configuration:

* image resolution: `224 × 224`
* 3D input length: `256`
* 3D resampled tokens: `100`
* audio input length: `250`
* audio resampled tokens: `100`
* audio codebooks: `4`
* audio vocabulary: `2048`
* resampler attention heads: `8`

The default MESSy configuration uses:

* transformer layers `9, 10, 11`
* distillation temperature `2.0`
* classification weight `α = 0.25`
* MESSy loss weight `1.0`

All of these values are configurable through the command-line interfaces.

---

## Reproducing the Paper Results

The numbers reported above are reference results from the paper.

Exact reproduction depends on factors including:

* the checkpoints used to obtain the exported generative latents,
* dataset version and preprocessing,
* random seed,
* training configuration,
* and hardware/software environment.

Because latent extraction pipelines and their external checkpoints are not distributed in this minimal release, the repository should be understood as releasing the **DLA latent-consumption and MESSy training pipeline**, rather than a fully self-contained reproduction of every upstream generation step.

---

## Citation

If you find this work useful, please consider citing:

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

Preprint:

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

## Contact

For questions regarding the paper or code, please open a GitHub issue or contact:

**Cristian Sbrolli**
Politecnico di Milano
`cristian.sbrolli@polimi.it`
