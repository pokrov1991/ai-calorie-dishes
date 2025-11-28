import math
import os
import random
from pathlib import Path

import timm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config
from .dataset import DishDataset, build_or_load_vocab, dish_collate_fn, get_transforms


# Воспроизводимость
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(cfg):
    if cfg.DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.DEVICE)


# Модель
class DishCalorieModel(nn.Module):
    """
    Мультимодальная модель:
    - Ингредиенты -> Embedding + усреднение
    - Масса -> скалярный признак
    - Картинка -> timm image_model (EfficientNet/ResNet) -> вектор признаков
    - Всё склеиваем и прогоняем через MLP -> калорийность
    """

    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.cfg = cfg
        self.pad_idx = cfg.PAD_IDX

        # Текстовая часть (ингредиенты)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=cfg.EMBEDDING_DIM,
            padding_idx=cfg.PAD_IDX,
        )

        # Визуальная часть (картинки)
        self.image_model = timm.create_model(
            cfg.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0, # вернёт вектор признаков
        )
        image_feat_dim = self.image_model.num_features

        # Общий признак
        # [mean_emb (D_emb)] + [mass (1)] + [image_feat (image_feat_dim)]
        input_dim = cfg.EMBEDDING_DIM + 1 + image_feat_dim

        layers = []
        hidden_dim = cfg.HIDDEN_DIM

        # Первый слой
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(cfg.DROPOUT))

        # Дополнительные скрытые слои
        for _ in range(cfg.NUM_HIDDEN_LAYERS - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(cfg.DROPOUT))

        # Выход (1 число - калорийность)
        layers.append(nn.Linear(hidden_dim, 1))  # скаляр: калории

        self.mlp = nn.Sequential(*layers)

    def forward(self, ingredients, lengths, mass, image):
        # Эмбеддинги ингредиентов
        emb = self.embedding(ingredients)
        # Маска паддинга
        pad_mask = (ingredients != self.pad_idx).unsqueeze(-1)
        emb_masked = emb * pad_mask
        # Сумма по ингредиентам
        sum_emb = emb_masked.sum(dim=1)
        # Среднее по длине
        lengths = lengths.clamp(min=1).unsqueeze(1)
        mean_emb = sum_emb / lengths

        # Масса
        mass = mass.unsqueeze(1)

        # Картинка
        img_feat = self.image_model(image)

        # Склейка
        x = torch.cat([mean_emb, mass, img_feat], dim=1)
        out = self.mlp(x).squeeze(1)

        return out
    

# DataLoader
def build_dataloaders(cfg, vocab):
    from functools import partial

    train_tfms = get_transforms(cfg, ds_type="train")
    val_tfms = get_transforms(cfg, ds_type="val")

    train_ds = DishDataset(cfg, split=cfg.TRAIN_SPLIT, vocab=vocab, transform=train_tfms)
    val_ds = DishDataset(cfg, split=cfg.VAL_SPLIT, vocab=vocab, transform=val_tfms)

    collate = partial(dish_collate_fn, pad_idx=cfg.PAD_IDX)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate,
    )

    return train_loader, val_loader


# Метрики
def compute_epoch_metrics(
    sq_err_sum,
    abs_err_sum,
    y_sum,
    y_sq_sum,
    n_samples,
):
    mse = sq_err_sum / n_samples
    rmse = math.sqrt(mse)
    mae = abs_err_sum / n_samples

    y_mean = y_sum / n_samples
    sst = y_sq_sum - n_samples * (y_mean ** 2)
    if sst <= 0:
        r2 = float("nan")
    else:
        r2 = 1.0 - sq_err_sum / sst

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


# Обучение / валидация одной эпохи
def run_epoch(
    model,
    loader,
    device,
    optimizer = None,
    cfg = None,
    epoch = 0,
    train = True
):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    n_samples = 0
    sq_err_sum = 0.0
    abs_err_sum = 0.0
    y_sum = 0.0
    y_sq_sum = 0.0

    for step, batch in enumerate(loader):
        ingredients = batch["ingredients"].to(device)
        lengths = batch["lengths"].to(device)
        mass = batch["mass"].to(device)
        target = batch["target"].to(device)
        image = batch["image"].to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            preds = model(
                ingredients=ingredients,
                lengths=lengths,
                mass=mass,
                image=image,
            )
            loss = F.mse_loss(preds, target)

            if train:
                loss.backward()
                if cfg is not None and cfg.GRAD_CLIP is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                optimizer.step()

        batch_size = target.size(0)
        n_samples += batch_size
        total_loss += loss.item() * batch_size

        diff = preds.detach() - target
        sq_err_sum += torch.sum(diff ** 2).item()
        abs_err_sum += torch.sum(torch.abs(diff)).item()
        y_sum += torch.sum(target).item()
        y_sq_sum += torch.sum(target ** 2).item()

        if train and cfg is not None and (step + 1) % cfg.PRINT_EVERY == 0:
            print(f"[Epoch {epoch}] step {step+1}/{len(loader)} loss={loss.item():.4f}")

    if n_samples == 0:
        metrics = {
            "loss": float("nan"),
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
        }
        return float("nan"), metrics

    avg_loss = total_loss / n_samples
    metrics = compute_epoch_metrics(
        sq_err_sum=sq_err_sum,
        abs_err_sum=abs_err_sum,
        y_sum=y_sum,
        y_sq_sum=y_sq_sum,
        n_samples=n_samples,
    )
    metrics["loss"] = avg_loss
    return avg_loss, metrics


# Основная функция train
def train(cfg):
    set_seed(cfg.SEED)
    device = get_device(cfg)

    print(f"Using device: {device}")
    os.makedirs(Path(cfg.CHECKPOINT_PATH).parent, exist_ok=True)

    # Вокабуляр
    vocab = build_or_load_vocab(cfg)
    vocab_size = max(vocab.values(), default=1) + 1  # учитываем PAD/UNK
    print(f"Vocab size (with PAD/UNK): {vocab_size}")

    # DataLoader
    train_loader, val_loader = build_dataloaders(cfg, vocab)

    # Модель / оптимизатор
    model = DishCalorieModel(vocab_size=vocab_size, cfg=cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    best_val_loss = float("inf")

    for epoch in range(1, cfg.N_EPOCHS + 1):
        print(f"Epoch {epoch}/{cfg.N_EPOCHS}")

        train_loss, train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            cfg=cfg,
            epoch=epoch,
            train=True,
        )

        val_loss, val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            cfg=cfg,
            epoch=epoch,
            train=False,
        )

        print(
            f"[Train] loss={train_metrics['loss']:.4f}, "
            f"MAE={train_metrics['mae']:.2f}, "
            f"RMSE={train_metrics['rmse']:.2f}, "
            f"R2={train_metrics['r2']:.4f}"
        )
        print(
            f"[Val]   loss={val_metrics['loss']:.4f}, "
            f"MAE={val_metrics['mae']:.2f}, "
            f"RMSE={val_metrics['rmse']:.2f}, "
            f"R2={val_metrics['r2']:.4f}"
        )

        # Сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                    "config": cfg.__dict__,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                cfg.CHECKPOINT_PATH,
            )
            print(f"Saved new best model to {cfg.CHECKPOINT_PATH}")


if __name__ == "__main__":
    cfg = Config()
    train(cfg)
