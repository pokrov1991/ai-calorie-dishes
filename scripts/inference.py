from functools import partial

import torch
from torch.utils.data import DataLoader

from .dataset import DishDataset, dish_collate_fn, get_transforms
from .utils import DishCalorieModel, compute_epoch_metrics


def load_trained_model(cfg, device):
    """
    Загружает обученную модель, словарь ингредиентов и чекпоинт из cfg.CHECKPOINT_PATH.
    """
    ckpt = torch.load(cfg.CHECKPOINT_PATH, map_location=device)

    vocab = ckpt["vocab"]
    vocab_size = max(vocab.values(), default=1) + 1 # учитываем PAD/UNK

    model = DishCalorieModel(vocab_size=vocab_size, cfg=cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, vocab, ckpt


def build_loader_for_split(cfg, vocab, split, ds_type = "val"):
    """
    Строит Dataset и DataLoader для произвольного сплита ('train' / 'test' / 'val').
    ds_type управляет трансформациями:
    - 'train' -> аугментации
    - 'val'   -> только resize + normalize
    """
    tfms = get_transforms(cfg, ds_type=ds_type)
    dataset = DishDataset(cfg, split=split, vocab=vocab, transform=tfms)

    collate = partial(dish_collate_fn, pad_idx=cfg.PAD_IDX)
    loader = DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False, # важно для привязки dish_ids к батчам
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate,
    )
    return dataset, loader


@torch.no_grad()
def evaluate_on_loader(model, loader, dataset, device):
    """
    Прогоняет модель по всему loader и:
    - считает итоговые метрики (MSE, MAE, RMSE, R2),
    - возвращает список dict'ов по каждому блюду:
        {
            "dish_id",
            "true_cal",
            "pred_cal",
            "abs_err"
        }
    """
    model.eval()

    n_samples = 0
    sq_err_sum = 0.0
    abs_err_sum = 0.0
    y_sum = 0.0
    y_sq_sum = 0.0

    records = []
    offset = 0 # индекс в dataset.dish_ids, чтобы сопоставить с батчами

    for batch in loader:
        ingredients = batch["ingredients"].to(device)
        lengths = batch["lengths"].to(device)
        mass = batch["mass"].to(device)
        target = batch["target"].to(device)
        image = batch["image"].to(device)

        preds = model(
            ingredients=ingredients,
            lengths=lengths,
            mass=mass,
            image=image,
        )

        diff = preds - target
        bs = target.size(0)

        sq_err_sum += torch.sum(diff ** 2).item()
        abs_err_sum += torch.sum(torch.abs(diff)).item()
        y_sum += torch.sum(target).item()
        y_sq_sum += torch.sum(target ** 2).item()
        n_samples += bs

        # так как shuffle=False, порядок совпадает с dataset
        dish_ids = dataset.dish_ids[offset : offset + bs]
        offset += bs

        for i in range(bs):
            records.append(
                {
                    "dish_id": dish_ids[i],
                    "true_cal": target[i].item(),
                    "pred_cal": preds[i].item(),
                    "abs_err": abs(diff[i].item()),
                }
            )

    metrics = compute_epoch_metrics(
        sq_err_sum=sq_err_sum,
        abs_err_sum=abs_err_sum,
        y_sum=y_sum,
        y_sq_sum=y_sq_sum,
        n_samples=n_samples,
    )
    # для консистентности с train/val
    metrics["loss"] = metrics["mse"]

    return metrics, records
