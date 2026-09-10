# latent_multimodal_vit_resampler.py
import os
import math
import argparse
from collections import OrderedDict

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

try:
    import wandb
except ImportError:
    wandb = None


# ==========================================
# 1. TRANSFORMER COMPONENTS (ViT Backbone)
# ==========================================

class MLPBlock(nn.Module):
    """Standard Transformer MLP block."""
    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, mlp_dim)
        self.act = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(mlp_dim, in_dim)
        self.dropout_2 = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.act(x)
        x = self.dropout_1(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)
        return x


class EncoderBlock(nn.Module):
    """Standard Transformer Encoder block."""
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.ln_2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor, return_attention: bool = False):
        x = self.ln_1(input)
        attn_output, attn_weights = self.self_attention(
            x, x, x, need_weights=True, average_attn_weights=False
        )
        x = self.dropout(attn_output)
        x = x + input
        y = self.ln_2(x)
        y = self.mlp(y)
        output = x + y
        if return_attention:
            return output, attn_weights
        return output


class Encoder(nn.Module):
    """Transformer Model Encoder."""
    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.empty(1, seq_length, hidden_dim).normal_(std=0.02)
        )
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                EncoderBlock(num_heads, hidden_dim, mlp_dim, dropout, attention_dropout)
                for _ in range(num_layers)
            ]
        )
        self.ln = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, input: torch.Tensor, return_attention: bool = False):
        input = input + self.pos_embedding
        x = self.dropout(input)

        attention_weights = []
        for layer in self.layers:
            if return_attention:
                x, weights = layer(x, return_attention=True)
                if torch.is_grad_enabled():
                    weights.retain_grad()
                attention_weights.append(weights)
            else:
                x = layer(x, return_attention=False)

        output = self.ln(x)
        return (output, attention_weights) if return_attention else output


# ==========================================
# 2. MODALITY ADAPTERS + RESAMPLER
# ==========================================

class AudioLatentAdapter(nn.Module):
    """
    DISCRETE audio tokens (e.g., EnCodec/AudioGen codebooks).
    Input:  LongTensor [B, Num_Codebooks, Time]
    Output: FloatTensor [B, Time, D]
    """
    def __init__(self, num_codebooks: int, vocab_size: int, embed_dim: int):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, embed_dim) for _ in range(num_codebooks)]
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        B, C, T = x.shape
        fused = None
        for i, emb_layer in enumerate(self.embeddings):
            if i < C:
                emb = emb_layer(x[:, i, :])  # [B, T, D]
                fused = emb if fused is None else (fused + emb)

        if fused is None:
            device = x.device
            D = self.embeddings[0].embedding_dim
            fused = torch.zeros(B, T, D, device=device)

        return self.output_proj(fused)


class PCLatentAdapter(nn.Module):
    """
    CONTINUOUS 3D latents.
    Input:  FloatTensor [B, Tokens, C] OR [B, C, Tokens]
    Output: FloatTensor [B, Tokens, D]
    """
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"PC latent must be [B,T,C] or [B,C,T], got {tuple(x.shape)}")
        # If shape is [B, C, T], transpose to [B, T, C]
        if x.shape[1] == self.input_dim:
            x = x.transpose(1, 2)
        return self.proj(x)


class TokenResampler(nn.Module):
    """
    Perceiver-style resampler (cross-attn from M learnable queries to T tokens):
      Input:  x [B, T, D]  (long sequence)
      Output: y [B, M, D]  (fixed token budget)
    """
    def __init__(
        self,
        dim: int,
        num_latents: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        ff_mult: int = 2,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.latents = nn.Parameter(torch.randn(1, num_latents, dim) * 0.02)

        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

        self.ln_out = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, D = x.shape
        q = self.latents.expand(B, -1, -1)  # [B, M, D]

        qn = self.ln_q(q)
        xn = self.ln_kv(x)

        attn_out, _ = self.attn(qn, xn, xn, need_weights=False)  # [B, M, D]
        y = q + self.drop(attn_out)

        y = self.ln_out(y)
        y = y + self.ff(y)
        y = self.ln_out(y)
        return y


# ==========================================
# 3. MAIN MULTI-MODAL MODEL (ViT over fused tokens)
# ==========================================

class MultiModalViT(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # Preserve user-provided order (do NOT sort) so separators match modality order.
        self.modalities = list(config["train_modalities"])
        self.validation_modalities = (
            config["validation_mode"].split("+")
            if config["validation_mode"] != "all"
            else self.modalities
        )

        vit_params = {
            "vit_b_16": {"patch_size": 16, "num_layers": 12, "num_heads": 12, "hidden_dim": 768, "mlp_dim": 3072},
            "vit_l_16": {"patch_size": 16, "num_layers": 24, "num_heads": 16, "hidden_dim": 1024, "mlp_dim": 4096},
        }[config["vit_model_name"]]

        self.image_size = config["image_size"]
        self.patch_size = vit_params["patch_size"]
        self.hidden_dim = vit_params["hidden_dim"]

        self.adapters = nn.ModuleDict()
        self.resamplers = nn.ModuleDict()
        self.modality_pos = nn.ParameterDict()  # positional embeddings BEFORE resampling
        self.separators = nn.ParameterList()
        self.null_tokens = nn.ParameterDict()   # null tokens AFTER resampling

        total_tokens = 0

        # CLS
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        total_tokens += 1

        # Separators: exactly (#modalities - 1)
        num_seps = max(0, len(self.modalities) - 1)
        for _ in range(num_seps):
            self.separators.append(nn.Parameter(torch.zeros(1, 1, self.hidden_dim)))
        total_tokens += num_seps

        # Image patches
        if "image" in self.modalities:
            self.conv_proj = nn.Conv2d(3, self.hidden_dim, kernel_size=self.patch_size, stride=self.patch_size)
            num_img_patches = (self.image_size // self.patch_size) ** 2
            total_tokens += num_img_patches

        # PC latent -> embed -> add pos -> resample -> (pc_out_tokens)
        if "pc" in self.modalities:
            pc_in_len = int(config["pc_seq_len"])
            pc_out_len = int(config.get("pc_out_tokens", 100))

            self.adapters["pc"] = PCLatentAdapter(
                input_dim=int(config["pc_latent_dim"]),
                embed_dim=self.hidden_dim,
            )

            self.modality_pos["pc"] = nn.Parameter(
                torch.empty(1, pc_in_len, self.hidden_dim).normal_(std=0.02)
            )

            self.resamplers["pc"] = TokenResampler(
                dim=self.hidden_dim,
                num_latents=pc_out_len,
                num_heads=int(config.get("resampler_heads", 8)),
                dropout=float(config.get("resampler_dropout", 0.0)),
                ff_mult=int(config.get("resampler_ff_mult", 2)),
            )

            self.null_tokens["pc"] = nn.Parameter(torch.zeros(1, pc_out_len, self.hidden_dim))
            total_tokens += pc_out_len

        # Audio latent -> embed -> add pos -> resample -> (audio_out_tokens)
        if "audio" in self.modalities:
            audio_in_len = int(config["audio_duration"] * config["audio_framerate"])
            audio_out_len = int(config.get("audio_out_tokens", 100))

            self.adapters["audio"] = AudioLatentAdapter(
                num_codebooks=int(config["audio_num_codebooks"]),
                vocab_size=int(config["audio_vocab_size"]),
                embed_dim=self.hidden_dim,
            )

            self.modality_pos["audio"] = nn.Parameter(
                torch.empty(1, audio_in_len, self.hidden_dim).normal_(std=0.02)
            )

            self.resamplers["audio"] = TokenResampler(
                dim=self.hidden_dim,
                num_latents=audio_out_len,
                num_heads=int(config.get("resampler_heads", 8)),
                dropout=float(config.get("resampler_dropout", 0.0)),
                ff_mult=int(config.get("resampler_ff_mult", 2)),
            )

            self.null_tokens["audio"] = nn.Parameter(torch.zeros(1, audio_out_len, self.hidden_dim))
            total_tokens += audio_out_len

        self.total_tokens = total_tokens

        print(
            f"Model initialized. Total Sequence Length: {total_tokens} | "
            f"modalities={self.modalities} | "
            f"img={(self.image_size // self.patch_size) ** 2 if 'image' in self.modalities else 0}, "
            f"pc_in={config.get('pc_seq_len', 0)}->pc_out={config.get('pc_out_tokens', 0) if 'pc' in self.modalities else 0}, "
            f"audio_in={int(config.get('audio_duration', 0) * config.get('audio_framerate', 0)) if 'audio' in self.modalities else 0}"
            f"->audio_out={config.get('audio_out_tokens', 0) if 'audio' in self.modalities else 0}"
        )

        self.encoder = Encoder(
            seq_length=total_tokens,
            num_layers=vit_params["num_layers"],
            num_heads=vit_params["num_heads"],
            hidden_dim=self.hidden_dim,
            mlp_dim=vit_params["mlp_dim"],
            dropout=0.1,
            attention_dropout=0.0,
        )
        self.head = nn.Linear(self.hidden_dim, config["num_classes"])

        self.init_weights()

    def init_weights(self):
        if "image" in self.modalities:
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)

        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        for p in self.separators:
            nn.init.normal_(p, std=0.02)
        for k in self.null_tokens:
            nn.init.normal_(self.null_tokens[k], std=0.02)

    def _process_image(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x

    def forward(self, batch: dict) -> torch.Tensor:
        # Grad-CAM compatibility (raw tensor implies image-only input)
        if isinstance(batch, torch.Tensor):
            batch = {"image": batch}

        B = next(iter(batch.values())).shape[0]

        final_sequence = [self.class_token.expand(B, -1, -1)]
        sep_idx = 0

        for modality in self.modalities:
            # Separator before every modality after the first
            if len(final_sequence) > 1:
                final_sequence.append(self.separators[sep_idx].expand(B, -1, -1))
                sep_idx += 1

            tokens = None

            if modality == "image":
                tokens = self._process_image(batch["image"])

            elif modality == "pc":
                long_tokens = self.adapters["pc"](batch["pc"])  # [B, pc_in, D]
                long_tokens = long_tokens + self.modality_pos["pc"]
                tokens = self.resamplers["pc"](long_tokens)     # [B, pc_out, D]

                if self.training and self.config["pc_dropout_prob"] > 0.0:
                    mask = (torch.rand(B, 1, 1, device=tokens.device) > self.config["pc_dropout_prob"]).float()
                    tokens = tokens * mask + self.null_tokens["pc"].expand(B, -1, -1) * (1 - mask)
                elif (not self.training) and ("pc" not in self.validation_modalities):
                    tokens = self.null_tokens["pc"].expand(B, -1, -1)

            elif modality == "audio":
                long_tokens = self.adapters["audio"](batch["audio"])  # [B, audio_in, D]
                long_tokens = long_tokens + self.modality_pos["audio"]
                tokens = self.resamplers["audio"](long_tokens)        # [B, audio_out, D]

                if self.training and self.config["audio_dropout_prob"] > 0.0:
                    mask = (torch.rand(B, 1, 1, device=tokens.device) > self.config["audio_dropout_prob"]).float()
                    tokens = tokens * mask + self.null_tokens["audio"].expand(B, -1, -1) * (1 - mask)
                elif (not self.training) and ("audio" not in self.validation_modalities):
                    tokens = self.null_tokens["audio"].expand(B, -1, -1)

            if tokens is not None:
                final_sequence.append(tokens)

        x = torch.cat(final_sequence, dim=1)

        # Catch any mismatch early (usually separators count vs total_tokens)
        if x.shape[1] != self.total_tokens:
            raise RuntimeError(f"Token length mismatch: got {x.shape[1]}, expected {self.total_tokens}")

        encoded = self.encoder(x)
        return self.head(encoded[:, 0])


# ==========================================
# 4. DATASET (Images + Latents)
# ==========================================

def _safe_torch_load(path: str):
    # weights_only=True exists in newer torch; keep compatibility.
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class UnifiedLatentDataset(Dataset):
    def __init__(self, config: dict, split: str, image_transform=None):
        self.config = config
        self.split = split
        self.image_transform = image_transform
        self.modalities = list(config["train_modalities"])
        self.samples = []

        anchor_modality = self.modalities[0]
        anchor_root = os.path.join(config[f"{anchor_modality}_dataset_path"], split)
        if not os.path.exists(anchor_root):
            raise FileNotFoundError(f"Root not found: {anchor_root}")

        print(f"Scanning {anchor_root}...")
        for dirpath, _, filenames in os.walk(anchor_root):
            for filename in filenames:
                anchor_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(anchor_path, anchor_root)
                base_name, _ = os.path.splitext(rel_path)

                paths = {"class_name": os.path.basename(os.path.dirname(anchor_path))}
                is_valid = True

                for mod in self.modalities:
                    if mod == "image":
                        image_root = os.path.join(config["image_dataset_path"], split, base_name)
                        image_path = next((image_root + ext for ext in (".jpg", ".jpeg", ".JPEG", ".png")
                                           if os.path.exists(image_root + ext)), None)
                        if image_path:
                            paths["image"] = image_path
                        else:
                            is_valid = False
                            break
                    else:
                        p_pt = os.path.join(config[f"{mod}_dataset_path"], split, base_name + ".pt")
                        if os.path.exists(p_pt):
                            paths[mod] = p_pt
                        else:
                            is_valid = False
                            break

                if is_valid:
                    self.samples.append(paths)

        if not self.samples:
            raise RuntimeError(f"No valid samples found intersecting modalities: {self.modalities}")

        self.class_names = sorted(list(set(s["class_name"] for s in self.samples)))
        self.c2i = {n: i for i, n in enumerate(self.class_names)}
        for s in self.samples:
            s["label"] = self.c2i[s["class_name"]]

        print(
            f"Split '{split}': Found {len(self.samples)} valid multimodal samples across "
            f"{len(self.class_names)} classes."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        out = {"label": torch.tensor(s["label"], dtype=torch.long)}

        try:
            # 1) Image
            if "image" in self.modalities:
                img = datasets.folder.default_loader(s["image"])
                out["image"] = self.image_transform(img) if self.image_transform else img

            # 2) PC latent: enforce [pc_seq_len, pc_latent_dim]
            if "pc" in self.modalities:
                latent = _safe_torch_load(s["pc"])
                target_len = int(self.config["pc_seq_len"])
                latent_dim = int(self.config["pc_latent_dim"])

                if latent.ndim != 2:
                    raise ValueError(f"PC latent must be [T,C] or [C,T], got {tuple(latent.shape)}")

                # If [C,T], transpose to [T,C]
                if latent.shape[0] == latent_dim and latent.shape[1] != latent_dim:
                    latent = latent.transpose(0, 1)

                if latent.shape[1] != latent_dim:
                    raise ValueError(f"PC latent dim mismatch: expected last dim {latent_dim}, got {latent.shape[1]}")

                curr_len = latent.shape[0]
                if curr_len > target_len:
                    latent = latent[:target_len, :]
                elif curr_len < target_len:
                    pad = torch.zeros(target_len - curr_len, latent.shape[1])
                    latent = torch.cat([latent, pad], dim=0)

                out["pc"] = latent.float()

            # 3) Audio latent: enforce [num_codebooks, audio_in_len]
            if "audio" in self.modalities:
                latent = _safe_torch_load(s["audio"]).long()
                if latent.ndim != 2:
                    raise ValueError(f"Audio latent must be [C,T], got {tuple(latent.shape)}")

                target_len = int(self.config["audio_duration"] * self.config["audio_framerate"])
                curr_len = latent.shape[-1]

                if curr_len > target_len:
                    latent = latent[:, :target_len]
                elif curr_len < target_len:
                    pad = torch.zeros(latent.shape[0], target_len - curr_len, dtype=torch.long)
                    latent = torch.cat([latent, pad], dim=1)

                out["audio"] = latent

        except Exception as e:
            print(f"Error loading index {idx} ({s}): {e}")
            return self.__getitem__((idx + 1) % len(self))

        return out


# ==========================================
# 5. TRAINING LOOP
# ==========================================

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(config: dict):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config["image_size"]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(config["image_size"] + 32),
        transforms.CenterCrop(config["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    set_seed(config["seed"])

    modal_str = "+".join(m[0].upper() for m in config["train_modalities"])
    run_name = f"LatentViT_Resampled_{modal_str}_sd{config['seed']}"

    if wandb is not None and config["wandb_mode"] != "disabled":
        wandb.init(project=config["wandb_project"], entity=config["wandb_entity"],
                   group=f"LatentResample_{modal_str}", name=run_name,
                   config=config, mode=config["wandb_mode"], reinit=True)

    try:
        train_ds = UnifiedLatentDataset(config, "train", train_transform)
        val_ds = UnifiedLatentDataset(config, config["val_split"], val_transform)

        train_loader = DataLoader(
            train_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=True,
        )
    except Exception as e:
        print(f"Dataset Init Failed: {e}")
        if wandb is not None and config["wandb_mode"] != "disabled":
            wandb.finish()
        return

    config["num_classes"] = len(train_ds.class_names)
    model = MultiModalViT(config).to(config["device"])

    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=config["min_lr"])

    best_val_acc = 0.0

    print(f"--- Starting Run: {run_name} ---")
    for epoch in range(config["epochs"]):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]")
        for batch in pbar:
            labels = batch.pop("label").to(config["device"])
            for k, v in batch.items():
                batch[k] = v.to(config["device"], non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            _, preds = torch.max(logits, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)
            pbar.set_postfix(acc=f"{(correct.float()/total):.4f}", L=model.total_tokens)

        train_acc = correct.double() / total
        train_loss /= total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="[Val]"):
                labels = batch.pop("label").to(config["device"])
                for k, v in batch.items():
                    batch[k] = v.to(config["device"], non_blocking=True)

                logits = model(batch)
                _, preds = torch.max(logits, 1)
                val_correct += torch.sum(preds == labels.data)
                val_total += labels.size(0)

        val_acc = val_correct.double() / val_total
        scheduler.step()

        if wandb is not None and config["wandb_mode"] != "disabled":
            wandb.log({"Train Loss": train_loss, "Train Acc": train_acc,
                       "Val Acc": val_acc, "LR": scheduler.get_last_lr()[0]})
        print(f"Ep {epoch+1}: Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            os.makedirs(config["output_dir"], exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "config": config,
                        "class_names": train_ds.class_names, "val_accuracy": best_val_acc},
                       os.path.join(config["output_dir"], f"{run_name}_best.pt"))

    if wandb is not None and config["wandb_mode"] != "disabled":
        wandb.finish()


# ==========================================
# 6. CONFIG & ENTRY POINT
# ==========================================

def get_args():
    parser = argparse.ArgumentParser()

    # Project Setup
    parser.add_argument("--wandb_project", type=str, default="MultimodalLatentsImagenet")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="disabled")
    parser.add_argument("--output_dir", type=str, default="./latents_imagenet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--val_split", type=str, default="val")

    # Data Paths
    parser.add_argument("--image_dataset_path", type=str, required=True)
    parser.add_argument("--pc_dataset_path", type=str, required=True)
    parser.add_argument("--audio_dataset_path", type=str, required=True)

    # Modalities
    parser.add_argument("--train_modalities", nargs="+", default=["image", "pc", "audio"])
    parser.add_argument("--validation_mode", type=str, default="all")
    parser.add_argument("--pc_dropout_prob", type=float, default=0.8)
    parser.add_argument("--audio_dropout_prob", type=float, default=0.8)

    # Latent Specs
    parser.add_argument("--pc_latent_dim", type=int, default=64)
    parser.add_argument("--pc_seq_len", type=int, default=256)  # BEFORE resampling

    parser.add_argument("--audio_num_codebooks", type=int, default=4)
    parser.add_argument("--audio_vocab_size", type=int, default=2048)
    parser.add_argument("--audio_duration", type=float, default=5.0)
    parser.add_argument("--audio_framerate", type=float, default=50.0)  # audio_in = duration*framerate

    # Resampler budgets (defaults give ~399 tokens total with image224/patch16)
    parser.add_argument("--pc_out_tokens", type=int, default=100)
    parser.add_argument("--audio_out_tokens", type=int, default=100)
    parser.add_argument("--resampler_heads", type=int, default=8)
    parser.add_argument("--resampler_dropout", type=float, default=0.0)
    parser.add_argument("--resampler_ff_mult", type=int, default=2)

    # Training Hyperparams
    parser.add_argument("--vit_model_name", type=str, default="vit_b_16", choices=["vit_b_16", "vit_l_16"])
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return vars(parser.parse_args())


if __name__ == "__main__":
    base_config = get_args()

    num_runs = int(base_config.pop("num_runs"))
    base_seed = int(base_config["seed"])

    print(f"Starting {num_runs} experimental runs...")

    for i in range(num_runs):
        print(f"\n=== Run {i+1}/{num_runs} ===")
        cfg = base_config.copy()
        cfg["seed"] = base_seed + i
        run_experiment(cfg)

    print("\nAll experimental runs complete.")
