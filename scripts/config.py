from pathlib import Path

class Config:
    # Пути
    DATA_PATH = Path("data/dish.csv") # основной CSV с колонкой split
    IMAGE_ROOT: Path = Path("data/images") 
    VOCAB_PATH = Path("models/ingredient_vocab.json")
    CHECKPOINT_PATH = Path("models/dish_calorie_model.pt")

    # Колонки в CSV
    COL_DISH_ID = "dish_id"
    COL_CALORIES = "total_calories"
    COL_MASS = "total_mass"
    COL_INGREDIENTS = "ingredients"
    COL_SPLIT = "split"

    # Названия сплитов
    TRAIN_SPLIT = "train"
    VAL_SPLIT = "test"

    # Вокабуляр ингредиентов
    MIN_INGR_FREQ = 1 # минимальная частота для включения ингредиента в словарь
    PAD_IDX = 0       # индекс паддинга
    UNK_IDX = 1       # индекс <UNK>

    # Даталоадеры
    BATCH_SIZE = 128
    NUM_WORKERS = 0 # лучше начать с 0, потом можно увеличить

    # Модель: текстовая / табличная часть (ингредиенты + масса)
    EMBEDDING_DIM = 64
    HIDDEN_DIM = 128
    NUM_HIDDEN_LAYERS = 2
    DROPOUT = 0.2

     # Модель: Изображения / timm
    IMAGE_MODEL_NAME = "tf_efficientnet_b0"  # как в других примерах
    IMAGE_SIZE = 224 # для информации (в коде не использую), т.к. логика берёт размер из timm_cfg

    # Обучение
    N_EPOCHS = 20
    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    GRAD_CLIP = 1.0

    # 'mps' для моего mac. 'cuda' для BM. 'auto' для автоматического выбора
    DEVICE = "mps"

    # Воспроизводимость
    SEED = 42

    # Логгинг
    PRINT_EVERY = 100 # как часто печатать шаги внутри эпохи
