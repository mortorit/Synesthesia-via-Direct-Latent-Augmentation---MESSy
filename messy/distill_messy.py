"""MESSy distillation from a latent multimodal teacher to an RGB student."""
import argparse
import os
from typing import Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from teacher import MultiModalViT, UnifiedLatentDataset, set_seed


class SynesthesiaHead(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))

    def forward(self, x):
        return self.net(x)


class LayerCapture:
    def __init__(self, model: nn.Module, layers: Iterable[int]):
        self.features: Dict[int, torch.Tensor] = {}
        self.handles = []
        for layer in layers:
            self.handles.append(model.encoder.layers[layer].register_forward_hook(self._hook(layer)))

    def _hook(self, layer):
        def capture(_, __, output):
            self.features[layer] = output[0] if isinstance(output, tuple) else output
        return capture

    def clear(self):
        self.features.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()


def modality_spans(config: dict) -> Dict[str, slice]:
    """Return token spans in the teacher sequence, excluding separators."""
    cursor = 1
    spans = {}
    patch_count = (config["image_size"] // 16) ** 2
    for index, modality in enumerate(config["train_modalities"]):
        if index:
            cursor += 1
        count = patch_count if modality == "image" else int(config[f"{modality}_out_tokens"])
        spans[modality] = slice(cursor, cursor + count)
        cursor += count
    return spans


def distillation_loss(student_logits, teacher_logits, labels, alpha, temperature):
    hard = F.cross_entropy(student_logits, labels)
    soft = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * temperature**2
    return alpha * hard + (1.0 - alpha) * soft


def evaluate(student, loader, device):
    student.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("label").to(device)
            images = batch["image"].to(device, non_blocking=True)
            correct += (student(images).argmax(1) == labels).sum().item()
            total += labels.numel()
    return correct / max(total, 1)


def make_config(args, num_classes):
    return {
        "train_modalities": ["image", "pc", "audio"],
        "validation_mode": "all",
        "image_size": args.image_size,
        "pc_latent_dim": args.pc_latent_dim,
        "pc_seq_len": args.pc_seq_len,
        "pc_out_tokens": args.pc_out_tokens,
        "audio_num_codebooks": args.audio_num_codebooks,
        "audio_vocab_size": args.audio_vocab_size,
        "audio_duration": args.audio_duration,
        "audio_framerate": args.audio_framerate,
        "audio_out_tokens": args.audio_out_tokens,
        "resampler_heads": args.resampler_heads,
        "resampler_dropout": args.resampler_dropout,
        "resampler_ff_mult": args.resampler_ff_mult,
        "pc_dropout_prob": args.pc_dropout_prob,
        "audio_dropout_prob": args.audio_dropout_prob,
        "vit_model_name": args.vit_model_name,
        "teacher_dropout": args.teacher_dropout,
        "teacher_attention_dropout": args.teacher_attention_dropout,
        "num_classes": num_classes,
    }


def run(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(args.image_size + 32), transforms.CenterCrop(args.image_size),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    cfg = make_config(args, 1)
    cfg.update({"image_dataset_path": args.image_dataset_path, "pc_dataset_path": args.pc_dataset_path,
                "audio_dataset_path": args.audio_dataset_path})
    train_ds = UnifiedLatentDataset(cfg, args.train_split, train_tf)
    val_ds = UnifiedLatentDataset(cfg, args.val_split, val_tf)
    cfg["num_classes"] = len(train_ds.class_names)
    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

    teacher = MultiModalViT(cfg).to(device)
    checkpoint = torch.load(args.teacher_checkpoint, map_location=device)
    checkpoint = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    teacher.load_state_dict(checkpoint, strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    student_cfg = dict(cfg)
    student_cfg.update({"train_modalities": ["image"], "validation_mode": "image",
                        "pc_dropout_prob": 0.0, "audio_dropout_prob": 0.0})
    student = MultiModalViT(student_cfg).to(device)
    layers = [int(layer) for layer in args.messy_layers]
    teacher_capture = LayerCapture(teacher, layers)
    student_capture = LayerCapture(student, layers)
    heads = nn.ModuleDict({f"{layer}_{mod}": SynesthesiaHead(student.hidden_dim)
                           for layer in layers for mod in ("pc", "audio")}).to(device)
    optimizer = torch.optim.AdamW(list(student.parameters()) + list(heads.parameters()),
                                  lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.min_lr)
    spans = modality_spans(cfg)
    os.makedirs(args.output_dir, exist_ok=True)
    best = 0.0

    try:
        for epoch in range(args.epochs):
            student.train(); heads.train(); total_loss = total = correct = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
                labels = batch.pop("label").to(device)
                teacher_batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                teacher_capture.clear(); student_capture.clear(); optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    teacher_logits = teacher(teacher_batch)
                student_logits = student({"image": teacher_batch["image"]})
                loss = distillation_loss(student_logits, teacher_logits, labels, args.alpha, args.temperature)
                syn_loss = torch.zeros((), device=device)
                for layer in layers:
                    student_cls = student_capture.features[layer][:, 0]
                    teacher_features = teacher_capture.features[layer]
                    for modality in ("pc", "audio"):
                        target = teacher_features[:, spans[modality], :].mean(dim=1).detach()
                        syn_loss = syn_loss + F.mse_loss(heads[f"{layer}_{modality}"](student_cls), target)
                loss = loss + args.messy_weight * syn_loss
                loss.backward(); optimizer.step()
                total_loss += loss.item() * labels.size(0); total += labels.size(0)
                correct += (student_logits.argmax(1) == labels).sum().item()
            val_acc = evaluate(student, val_loader, device); scheduler.step()
            print(f"epoch={epoch + 1} loss={total_loss / max(total, 1):.4f} train_acc={correct / max(total, 1):.4f} val_acc={val_acc:.4f}")
            if val_acc > best:
                best = val_acc
                torch.save({"state_dict": student.state_dict(), "config": student_cfg,
                            "class_names": train_ds.class_names, "val_accuracy": best},
                           os.path.join(args.output_dir, "messy_student_best.pt"))
    finally:
        teacher_capture.close(); student_capture.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_dataset_path", required=True); parser.add_argument("--pc_dataset_path", required=True)
    parser.add_argument("--audio_dataset_path", required=True); parser.add_argument("--teacher_checkpoint", required=True)
    parser.add_argument("--output_dir", default="checkpoints/messy"); parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val"); parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vit_model_name", default="vit_b_16", choices=["vit_b_16", "vit_l_16"])
    parser.add_argument("--image_size", type=int, default=224); parser.add_argument("--pc_latent_dim", type=int, default=64)
    parser.add_argument("--pc_seq_len", type=int, default=256); parser.add_argument("--pc_out_tokens", type=int, default=100)
    parser.add_argument("--audio_num_codebooks", type=int, default=4); parser.add_argument("--audio_vocab_size", type=int, default=2048)
    parser.add_argument("--audio_duration", type=float, default=5.0); parser.add_argument("--audio_framerate", type=float, default=50.0)
    parser.add_argument("--audio_out_tokens", type=int, default=100); parser.add_argument("--resampler_heads", type=int, default=8)
    parser.add_argument("--resampler_dropout", type=float, default=0.0); parser.add_argument("--resampler_ff_mult", type=int, default=2)
    parser.add_argument("--pc_dropout_prob", type=float, default=0.8); parser.add_argument("--audio_dropout_prob", type=float, default=0.8)
    parser.add_argument("--teacher_dropout", type=float, default=0.1); parser.add_argument("--teacher_attention_dropout", type=float, default=0.0)
    parser.add_argument("--messy_layers", nargs="+", type=int, default=[9, 10, 11]); parser.add_argument("--messy_weight", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.25); parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=32); parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=3e-5); parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01); parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
