# Synesthesia via Direct Latent Augmentation

Code release for **Direct Latent Augmentation (DLA)** and **Multilayer Explicit Simulated Synesthesia (MESSy)** from:

> Cristian Sbrolli, Nicolas Michel, Matteo Matteucci, and Toshihiko Yamasaki. *Synesthesia via Direct Latent Augmentation: Bypassing the Decode-Encode Loop for Cross-Modal Distillation.* ECCV 2026.

[Paper](https://arxiv.org/abs/2606.08336)

## What is included

This repository contains the minimal training release:

1. `messy/teacher.py` trains the multimodal teacher on RGB images plus precomputed continuous 3D and discrete audio latents.
2. `messy/distill_messy.py` trains the RGB-only student with logit distillation and the multilayer MESSy prediction loss.
3. [Latent format](docs/latent_format.md) documents the expected file layout and tensor shapes.

The generative-model extraction pipelines are intentionally not bundled: they depend on large external checkpoints and changing third-party code. The training release starts from exported `.pt` latents so experiments are portable and easier to reproduce.

## Installation

```bash
git clone https://github.com/<owner>/<repository>.git
cd <repository>
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

PyTorch installation may need to follow the command recommended for your CUDA version at [pytorch.org](https://pytorch.org/get-started/locally/).

## Dataset layout

Images use the usual class-folder layout. Latent roots mirror the image tree:

```text
data/imagenet100/
  images/train/<class>/<sample>.jpg
  images/val/<class>/<sample>.jpg
  pc_latents/train/<class>/<sample>.pt
  pc_latents/val/<class>/<sample>.pt
  audio_latents/train/<class>/<sample>.pt
  audio_latents/val/<class>/<sample>.pt
```

See [docs/latent_format.md](docs/latent_format.md) for accepted tensor shapes. All three modalities must contain the same relative sample paths.

## Train the DLA teacher

```bash
python messy/teacher.py \
  --image_dataset_path data/imagenet100/images \
  --pc_dataset_path data/imagenet100/pc_latents \
  --audio_dataset_path data/imagenet100/audio_latents \
  --output_dir checkpoints/teacher \
  --val_split val \
  --wandb_mode disabled
```

The best checkpoint is saved with its model configuration and class names as `*.pt`. For W&B tracking, use `--wandb_mode online --wandb_project <project>` and provide `--wandb_entity` when required.

## Train the MESSy student

Use the teacher checkpoint produced above:

```bash
python messy/distill_messy.py \
  --image_dataset_path data/imagenet100/images \
  --pc_dataset_path data/imagenet100/pc_latents \
  --audio_dataset_path data/imagenet100/audio_latents \
  --teacher_checkpoint checkpoints/teacher/LatentViT_Resampled_IPA_sd42_best.pt \
  --output_dir checkpoints/messy \
  --val_split val
```

The default MESSy targets are the pooled audio and 3D token summaries from layers 9, 10, and 11 of ViT-B/16. At inference, load only `messy_student_best.pt` and use its `state_dict` with the image-only model; audio, 3D latents, teacher, and prediction heads are not needed.

## Reported reference results

The paper reports the following top-1 accuracies for the RGB-only student:

| Dataset | Image-only | MESSy |
| --- | ---: | ---: |
| Imagenette2-320 | 79.63 | 84.23 |
| Caltech101 | 55.10 | 63.52 |
| ImageNet-100 | 61.57 | 68.54 |

Exact results depend on latent extraction checkpoints, dataset version, random seed, and hardware.

## License

Add the project license before publishing the repository. This release does not redistribute external generator checkpoints or datasets; their licenses remain the responsibility of the user.
