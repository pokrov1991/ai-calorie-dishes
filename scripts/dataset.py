import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from PIL import Image
import timm
import torchvision.transforms as T


def parse_ingredients(ingr_str):
    """
    'ingr_0000000111;ingr_0000000112;...' -> ['ingr_0000000111', 'ingr_0000000112', ...]
    """
    if pd.isna(ingr_str):
        return []
    parts = [p.strip() for p in str(ingr_str).split(";")]
    return [p for p in parts if p]  # убираем пустые строки


def build_or_load_vocab(cfg):
    """
    Строим словарь ингредиентов по train-части.
    Формат: { 'ingr_0000000111': 2, 'ingr_0000000112': 3, ... }
    Индексы:
      0 - PAD
      1 - UNK
      начиная с 2 - реальные ингредиенты
    """
    vocab_path = Path(cfg.VOCAB_PATH)

    if vocab_path.exists():
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return vocab

    # Строим по train-части
    df = pd.read_csv(cfg.DATA_PATH)
    df_train = df[df[cfg.COL_SPLIT] == cfg.TRAIN_SPLIT]

    counter = Counter()
    for s in df_train[cfg.COL_INGREDIENTS]:
        for token in parse_ingredients(s):
            counter[token] += 1

    vocab = {}
    next_idx = 2 # 0 = PAD, 1 = UNK

    for token, freq in counter.items():
        if freq >= cfg.MIN_INGR_FREQ:
            vocab[token] = next_idx
            next_idx += 1

    os.makedirs(vocab_path.parent, exist_ok=True)
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    return vocab


class DishDataset(Dataset):
    """
    Dataset для одного сплита (train / val / test) из общего dish.csv.
    Возвращает словарь:
      {
        "ingredients", # индексы ингредиентов
        "mass",        # общий вес блюда
        "target",      # калорийность (для train/val)
        "image"        # изображение блюда
      }
    """

    def __init__(self, cfg, split, vocab, transform = None):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.vocab = vocab
        self.pad_idx = cfg.PAD_IDX
        self.unk_idx = cfg.UNK_IDX
        self.transform = transform

        df = pd.read_csv(cfg.DATA_PATH)
        df_split = df[df[cfg.COL_SPLIT] == split].reset_index(drop=True)

        self.dish_ids = df_split[cfg.COL_DISH_ID].tolist()
        self.ingredients_raw = df_split[cfg.COL_INGREDIENTS].tolist()
        self.mass = df_split[cfg.COL_MASS].astype(float).tolist()

        # target может отсутствовать для test; тогда используем NaN/None
        if cfg.COL_CALORIES in df_split.columns:
            self.targets = df_split[cfg.COL_CALORIES].astype(float).tolist()
        else:
            self.targets = [None] * len(df_split)

    def __len__(self):
        return len(self.dish_ids)

    def encode_ingredients(self, ingr_str):
        tokens = parse_ingredients(ingr_str)
        idxs = []

        for tok in tokens:
            idx = self.vocab.get(tok, self.unk_idx)
            idxs.append(idx)

        # Если ингредиентов нет, вернём пустой тензор (обработаем в collate_fn)
        if len(idxs) == 0:
            return torch.empty(0, dtype=torch.long)

        return torch.tensor(idxs, dtype=torch.long)
    
    def _load_image(self, dish_id):
        # Загружаем data/images/<dish_id>/rgb.png, применяем transform.
        img_path = self.cfg.IMAGE_ROOT / dish_id / "rgb.png"
        # На всякий случай, если вдруг файла нет
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found for dish_id={dish_id}: {img_path}")

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)
        return img
    
    def __getitem__(self, idx):
        ingr_tensor = self.encode_ingredients(self.ingredients_raw[idx])
        mass = torch.tensor(self.mass[idx], dtype=torch.float32)

        target_value = self.targets[idx]
        if target_value is None or pd.isna(target_value):
            target = torch.tensor(0.0, dtype=torch.float32) # для test не используется
        else:
            target = torch.tensor(float(target_value), dtype=torch.float32)

        dish_id = self.dish_ids[idx]
        image = self._load_image(dish_id)

        return {
            "ingredients": ingr_tensor,
            "mass": mass,
            "target": target,
            "image": image,
        }

def dish_collate_fn(batch, pad_idx):
    """
    Collate-функция для упаковки батча с переменной длиной списка ингредиентов.
    """
    batch_size = len(batch)
    lengths = [item["ingredients"].shape[0] for item in batch]
    max_len = max(lengths) if lengths else 0

    if max_len == 0:
        # деградационный случай, но на реальных данных ингредиенты должны быть
        ingredients_batch = torch.full((batch_size, 1), pad_idx, dtype=torch.long)
        lengths_tensor = torch.ones(batch_size, dtype=torch.long)
    else:
        ingredients_batch = torch.full((batch_size, max_len), pad_idx, dtype=torch.long)
        lengths_tensor = torch.zeros(batch_size, dtype=torch.long)
        for i, item in enumerate(batch):
            ingr = item["ingredients"]
            L = ingr.shape[0]
            if L > 0:
                ingredients_batch[i, :L] = ingr
            lengths_tensor[i] = max(L, 1)

    mass_batch = torch.stack([item["mass"] for item in batch], dim=0)
    target_batch = torch.stack([item["target"] for item in batch], dim=0)
    image_batch = torch.stack([item["image"] for item in batch], dim=0)

    return {
        "ingredients": ingredients_batch,
        "lengths": lengths_tensor,
        "mass": mass_batch,
        "target": target_batch,
        "image": image_batch,
    }


def get_transforms(cfg, ds_type = "train"):
    """
    Возвращает torchvision-трансформации, согласованные с timm-моделью.
    """
    timm_cfg = timm.get_pretrained_cfg(cfg.IMAGE_MODEL_NAME)
    mean = timm_cfg.mean
    std = timm_cfg.std
    size = timm_cfg.input_size[1:]

    if ds_type == "train":
        return T.Compose([
            T.Resize(size),
            # T.RandomHorizontalFlip(),
            # T.RandomRotation(5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        return T.Compose([
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
