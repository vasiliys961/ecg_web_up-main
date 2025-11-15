# -*- coding: utf-8 -*-
"""Скрипт для анализа CSV-файла с признаками и возврата результатов в формате списка словарей.
Использует ансамбль из MLP, 1D CNN и ResNet1D моделей.
"""
# === Импорты ===
# ВАЖНО: Все необходимые библиотеки должны быть импортированы до их использования!
import os
import numpy as np
import pandas as pd
# --- ИМПОРТ PYTORCH ДОЛЖЕН БЫТЬ ЗДЕСЬ ---
import torch
import torch.nn as nn
# ---------------------------------------

# === Пути к файлам моделей и нормализации ===
# Эти пути относительны к корню проекта ecg_web_app
# Они будут использоваться при вызове из app.py, но в analyze передается model_dir

# === Словарь переводов scp_codes на русский язык ===
DIAGNOSIS_TRANSLATION = {
    'SR': 'Синусовый ритм',
    'NORM': 'Нормальная ЭКГ',
    'ABQRS': 'Аберрантный QRS комплекс',
    'IMI': 'Инфаркт миокарда (нижняя стенка)',
    'ASMI': 'Инфаркт миокарда (переднеперегородочная стенка)',
    'LVH': 'Гипертрофия левого желудочка',
    'NDT': 'Неспецифические изменения ST-T',
    'LAFB': 'Блокада передней ветви левой ножки пучка Гиса',
    'AFIB': 'Фибрилляция предсердий',
    'PVC': 'Преждевременное желудочковое сокращение',
    'IRBBB': 'Неполная блокада правой ножки пучка Гиса',
    'VCLVH': 'Гипертрофия желудочков или левого желудочка',
    'STACH': 'Синусовая тахикардия',
    'IVCD': 'Внутрижелудочковая блокада',
    'SARRH': 'Синусовый ритм с аберрантным проведением',
    'ISCAL': 'Ишемия миокарда (нижняя стенка)',
    'SBRAD': 'Синусовая брадикардия',
    'QWAVE': 'Патологический Q-волновой комплекс',
    'CRBBB': 'Полная блокада правой ножки пучка Гиса',
    'CLBBB': 'Полная блокада левой ножки пучка Гиса',
    'ILMI': 'Инфаркт миокарда (нижнебоковая стенка)',
    'LOWT': 'Низкий T-волновой комплекс',
    'PAC': 'Преждевременное предсердное сокращение',
    'AMI': 'Острый инфаркт миокарда (передняя стенка)',
}

# === Топ-24 диагноза (в правильном порядке!) ===
TOP_24_CODES = ['SR', 'NORM', 'ABQRS', 'IMI', 'ASMI', 'LVH', 'NDT', 'LAFB', 'AFIB', 'PVC',
                'IRBBB', 'VCLVH', 'STACH', 'IVCD', 'SARRH', 'ISCAL', 'SBRAD', 'QWAVE',
                'CRBBB', 'CLBBB', 'ILMI', 'LOWT', 'PAC', 'AMI']

# === Определения архитектур моделей ===

def create_mlp_model(input_dim, num_classes=24):
    """Создает MLP модель, совместимую с сохраненными весами."""
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.5),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.4),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, num_classes),
        # nn.Sigmoid() # Не включаем, так как используем BCEWithLogitsLoss
    )

class ECG1DCNN(nn.Module):
    def __init__(self, num_classes=24, input_channels=1, input_length=531):
        super(ECG1DCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32) # Адаптивный пулинг для фиксированного размера
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 32, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.classifier(x)
        return x # BCEWithLogitsLoss

# --- ResNet1D (адаптированная) ---
class ResidualBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(ResidualBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet1d(nn.Module):
    def __init__(self, block, layers, num_classes=24, input_channels=1, input_length=531):
        super(ResNet1d, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # Используем AdaptiveAvgPool1d для фиксированного выхода
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, 3, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x # BCEWithLogitsLoss

def resnet18_1d(**kwargs):
    model = ResNet1d(ResidualBlock1d, [2, 2, 2, 2], **kwargs)
    return model

# === Логика загрузки моделей ансамбля ===

def load_ensemble_models(device, model_dir="models"):
    """
    Загружает три модели и возвращает словарь.
    Args:
        device (torch.device): Устройство для загрузки моделей.
        model_dir (str): Путь к папке с моделями.
    Returns:
        dict: Словарь с загруженными моделями.
    """
    models = {}
    print(f"Загрузка моделей из: {model_dir}")
    
    # - Загружаем MLP -
    mlp_path = os.path.join(model_dir, "ecg_model.pth")
    if os.path.exists(mlp_path):
        try:
            model_mlp = create_mlp_model(input_dim=531, num_classes=24).to(device)
            checkpoint_mlp = torch.load(mlp_path, map_location=device)
            
            # Проверяем, есть ли 'state_dict' в чекпоинте
            if 'state_dict' in checkpoint_mlp:
                state_dict_mlp = checkpoint_mlp['state_dict']
            elif 'model_state_dict' in checkpoint_mlp:
                state_dict_mlp = checkpoint_mlp['model_state_dict']
            else:
                state_dict_mlp = checkpoint_mlp # Прямо state_dict
            
            # Проверяем, нужно ли убирать префикс 'net.'
            new_state_dict = {}
            for k, v in state_dict_mlp.items():
                if k.startswith('net.'):
                    new_key = k[4:] # Убираем 'net.'
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v
            model_mlp.load_state_dict(new_state_dict)
            model_mlp.eval()
            models["MLP"] = model_mlp
            print("✅ MLP загружена")
        except Exception as e:
            print(f"❌ Ошибка загрузки MLP: {e}")
    else:
        print(f"❌ Файл MLP не найден: {mlp_path}")

    # - Загружаем 1D CNN -
    cnn_path = os.path.join(model_dir, "ecg_1dcnn_best.pth")
    if os.path.exists(cnn_path):
        try:
            model_cnn = ECG1DCNN(num_classes=24, input_channels=1, input_length=531).to(device)
            checkpoint_cnn = torch.load(cnn_path, map_location=device)
            
            # Проверяем, есть ли 'state_dict' в чекпоинте
            if 'state_dict' in checkpoint_cnn:
                state_dict_cnn = checkpoint_cnn['state_dict']
            elif 'model_state_dict' in checkpoint_cnn:
                state_dict_cnn = checkpoint_cnn['model_state_dict']
            else:
                state_dict_cnn = checkpoint_cnn # Прямо state_dict
            
            # Проверяем, нужно ли убирать префикс 'net.'
            new_state_dict_cnn = {}
            for k, v in state_dict_cnn.items():
                if k.startswith('net.'):
                    new_key = k[4:] # Убираем 'net.'
                    new_state_dict_cnn[new_key] = v
                else:
                    new_state_dict_cnn[k] = v
            model_cnn.load_state_dict(new_state_dict_cnn)
            model_cnn.eval()
            models["CNN"] = model_cnn
            print("✅ 1D CNN загружена")
        except Exception as e:
            print(f"❌ Ошибка загрузки 1D CNN: {e}")
    else:
        print(f"❌ Файл 1D CNN не найден: {cnn_path}")

    # - Загружаем ResNet1D -
    resnet_path = os.path.join(model_dir, "ecg_resnet1d_features_best.pth")
    if os.path.exists(resnet_path):
        try:
            model_resnet = resnet18_1d(num_classes=24, input_channels=1, input_length=531).to(device)
            checkpoint_resnet = torch.load(resnet_path, map_location=device)
            
            # Проверяем, есть ли 'state_dict' в чекпоинте
            if 'state_dict' in checkpoint_resnet:
                state_dict_resnet = checkpoint_resnet['state_dict']
            elif 'model_state_dict' in checkpoint_resnet:
                state_dict_resnet = checkpoint_resnet['model_state_dict']
            else:
                state_dict_resnet = checkpoint_resnet # Прямо state_dict
            
            # Проверяем, нужно ли убирать префикс 'net.'
            new_state_dict_resnet = {}
            for k, v in state_dict_resnet.items():
                if k.startswith('net.'):
                    new_key = k[4:] # Убираем 'net.'
                    new_state_dict_resnet[new_key] = v
                else:
                    new_state_dict_resnet[k] = v
            model_resnet.load_state_dict(new_state_dict_resnet)
            model_resnet.eval()
            models["ResNet"] = model_resnet
            print("✅ ResNet1D загружена")
        except Exception as e:
            print(f"❌ Ошибка загрузки ResNet1D: {e}")
    else:
        print(f"❌ Файл ResNet1D не найден: {resnet_path}")

    if not models:
        raise RuntimeError("❌ Не удалось загрузить ни одну модель для ансамбля.")
    return models

def get_ensemble_predictions(X_dense, X_1d, models_dict, device):
    """Делает предсказания ансамблем.
    Args:
        X_dense (np.ndarray): Признаки для MLP, форма (N, 531).
        X_1d (np.ndarray): Признаки для CNN/ResNet, форма (N, 1, 531).
        models_dict (dict): Словарь с загруженными моделями.
        device (torch.device): Устройство.
    Returns:
        np.ndarray: Вероятности, форма (N, 24).
    """
    all_preds = []
    # Конвертируем в torch.Tensor и переносим на устройство
    X_dense_torch = torch.tensor(X_dense, dtype=torch.float32).to(device)
    X_1d_torch = torch.tensor(X_1d, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        # - MLP -
        if "MLP" in models_dict:
            model_mlp = models_dict["MLP"]
            logits_mlp = model_mlp(X_dense_torch)
            # Применяем сигмоиду для получения вероятностей
            probs_mlp = torch.sigmoid(logits_mlp)
            # Переносим обратно на CPU и конвертируем в numpy
            all_preds.append(probs_mlp.cpu().numpy())
        
        # - 1D CNN -
        if "CNN" in models_dict:
            model_cnn = models_dict["CNN"]
            logits_cnn = model_cnn(X_1d_torch)
            probs_cnn = torch.sigmoid(logits_cnn)
            all_preds.append(probs_cnn.cpu().numpy())
        
        # - ResNet1D -
        if "ResNet" in models_dict:
            model_resnet = models_dict["ResNet"]
            logits_resnet = model_resnet(X_1d_torch)
            probs_resnet = torch.sigmoid(logits_resnet)
            all_preds.append(probs_resnet.cpu().numpy())
    
    if not all_preds:
        raise RuntimeError("❌ Не удалось получить предсказания ни от одной модели.")
    
    # - Усредняем предсказания -
    # all_preds - это список массивов numpy, каждый размером (N, 24)
    # np.mean по оси 0 усреднит по моделям
    ensemble_proba = np.mean(np.array(all_preds), axis=0)
    return ensemble_proba

# === Основная функция анализа ===
def analyze(csv_file_path, model_dir="models"):
    """
    Анализирует CSV-файл с признаками и возвращает список диагнозов с вероятностями.
    Args:
        csv_file_path (str): Путь к файлу CSV.
        model_dir (str): Путь к папке с моделями и файлами нормализации.
    Returns:
        list[dict]: Список словарей с ключами 'code', 'ru_name', 'probability'.
    Raises:
        FileNotFoundError: Если не найдены файлы моделей или нормализации.
        ValueError: Если файл CSV имеет неправильный формат.
        RuntimeError: Если не удалось загрузить модели или получить предсказания.
    """
    print(f"Начало анализа CSV файла: {csv_file_path}")
    print(f"Папка моделей: {model_dir}")

    # --- Определяем устройство ---
    # ПРИНУДИТЕЛЬНО используем CPU для избежания проблем с MPS
    device = torch.device("cpu")
    print("⚠️ Принудительно используем CPU.")
    # ------------------------------------------------------------------------------
    print(f"Используемое устройство: {device}")

    # --- Загружаем параметры нормализации ---
    mean_path = os.path.join(model_dir, "ecg_train_mean.npy")
    std_path = os.path.join(model_dir, "ecg_train_std.npy")
    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        raise FileNotFoundError(f"Файлы нормализации не найдены: {mean_path}, {std_path}")
    train_mean = np.load(mean_path)
    train_std = np.load(std_path)
    print(f"✅ Параметры нормализации загружены. Mean shape: {train_mean.shape}, Std shape: {train_std.shape}")

    # --- Загружаем данные ---
    try:
        df_new = pd.read_csv(csv_file_path)
        print(f"✅ Данные загружены. Shape: {df_new.shape}")
    except Exception as e:
        raise ValueError(f"Ошибка загрузки файла {csv_file_path}: {e}")

    # --- Подготавливаем данные ---
    try:
        # Извлекаем признаки
        feature_columns = [col for col in df_new.columns if col != 'ecg_id']
        if len(feature_columns) != 531:
            raise ValueError(f"Ожидается 531 признак, но найдено {len(feature_columns)}. Проверь файл {csv_file_path}.")
        X_new = df_new[feature_columns].values
        print(f"✅ Признаки извлечены. Shape: {X_new.shape}")

        # --- Добавлено: Проверка и обработка NaN/Inf в исходных данных ---
        if np.isnan(X_new).any() or np.isinf(X_new).any():
            print("⚠️ ВНИМАНИЕ: В исходных признаках обнаружены NaN или Inf!")
            # Заполняем NaN и Inf перед нормализацией
            # Заменяем NaN на среднее значение из обучающей выборки
            nan_mask = np.isnan(X_new)
            if np.any(nan_mask):
                print("  -> Заменяем NaN на значения из train_mean.")
                X_new = np.where(nan_mask, train_mean, X_new)
            
            # Заменяем Inf на большие конечные числа
            X_new = np.nan_to_num(X_new, nan=0.0, posinf=1e6, neginf=-1e6)
            print("  -> Заменены Inf на конечные значения.")
            
        print(f"DEBUG (исходные данные): Min: {np.min(X_new)}, Max: {np.max(X_new)}, Mean: {np.mean(X_new)}")

        # Нормализация (используем параметры из обучения!)
        # np.where(std == 0, 1.0, std) уже применено при сохранении
        X_new = (X_new - train_mean) / train_std
        print("✅ Данные нормализованы.")

        # --- Добавлено: Проверка на NaN/Inf после нормализации ---
        if np.isnan(X_new).any() or np.isinf(X_new).any():
            print("⚠️ ВНИМАНИЕ: В нормализованных признаках обнаружены NaN или Inf после нормализации!")
            # Заполняем их
            X_new = np.nan_to_num(X_new, nan=0.0, posinf=1e6, neginf=-1e6)
            print("  -> Заменены NaN/Inf на конечные значения.")

        print(f"DEBUG (нормализованные данные): Min: {np.min(X_new)}, Max: {np.max(X_new)}, Mean: {np.mean(X_new)}")

        # Для 1D моделей добавляем размерность канала
        X_new_1d = X_new[:, np.newaxis, :]  # Формат: (N, 1, 531)
        print(f"✅ Данные подготовлены для 1D моделей. Shape: {X_new_1d.shape}")

        # Данные для MLP (без дополнительной размерности)
        X_new_dense = X_new  # Формат: (N, 531)
        print(f"✅ Данные подготовлены для MLP. Shape: {X_new_dense.shape}")

    except Exception as e:
        raise ValueError(f"Ошибка подготовки данных: {e}")

    # --- Загрузка моделей ансамбля ---
    print("Загрузка моделей...")
    try:
        models_dict = load_ensemble_models(device, model_dir)
        if not models_dict:
            raise RuntimeError("Не удалось загрузить ни одну модель ансамбля.")
        print(f"✅ Загружены модели: {list(models_dict.keys())}")
    except Exception as e:
        raise RuntimeError(f"Ошибка загрузки моделей: {e}")

    # --- Получаем предсказания ансамбля ---
    print("🔮 Получаем предсказания ансамбля...")
    try:
        predictions_proba = get_ensemble_predictions(X_new_dense, X_new_1d, models_dict, device)
        if predictions_proba.size == 0:
            raise RuntimeError("Не удалось получить предсказания.")
        print(f"✅ Предсказания получены. Shape: {predictions_proba.shape}")

        # --- Добавлено: Проверка на NaN/Inf в предсказаниях ---
        if np.isnan(predictions_proba).any():
            print("⚠️ ВНИМАНИЕ: В предсказаниях обнаружены NaN!")
        if np.isinf(predictions_proba).any():
            print("⚠️ ВНИМАНИЕ: В предсказаниях обнаружены Inf!")
        print(f"DEBUG (предсказания): Min: {np.min(predictions_proba)}, Max: {np.max(predictions_proba)}, Mean: {np.mean(predictions_proba)}")
        print(f"DEBUG (первые 5 вероятностей для 1-й ЭКГ): {predictions_proba[0][:5]}")

    except Exception as e:
        raise RuntimeError(f"Ошибка получения предсказаний: {e}")

    # --- Форматируем и возвращаем результаты ---
    results = []
    # Берем результаты для первого ЭКГ в файле (или для всех, если нужно)
    # В данном случае анализируем только первый ЭКГ из файла
    if predictions_proba.shape[0] > 0:
        probs = predictions_proba[0]
        print(f"DEBUG: Первые 3 вероятности из модели (для 1-й ЭКГ): {probs[:3]}")
        for i, code in enumerate(TOP_24_CODES):
            ru_name = DIAGNOSIS_TRANSLATION.get(code, code)
            prob = probs[i]
            print(f"DEBUG: Диагноз {code}, индекс {i}, вероятность из probs[i]: {prob}, тип: {type(prob)}")
            # Проверка на допустимость значения перед добавлением
            if not (np.isnan(prob) or np.isinf(prob)):
                results.append({
                    'code': code,
                    'ru_name': ru_name,
                    'probability': float(prob) # Преобразуем в стандартный тип Python
                })
            else:
                print(f"⚠️ Пропущен диагноз {code} из-за недопустимого значения вероятности: {prob}")
                # Можно добавить с вероятностью 0.0, если нужно показать все диагнозы
                # results.append({'code': code, 'ru_name': ru_name, 'probability': 0.0})

        # Сортируем по убыванию вероятности
        results.sort(key=lambda x: x['probability'], reverse=True)
        print(f"✅ Формирование списка результатов завершено. Найдено {len(results)} диагнозов.")
    else:
        print("⚠️ Предсказания пусты, список результатов будет пуст.")

    # ВАЖНО: Финальная отладка содержимого results
    if results:
        print(f"DEBUG: Первый элемент results перед return: {results[0]}")
    else:
        print("DEBUG: results пуст перед return.")

    print(f"✅ Анализ CSV завершен. Найдено {len(results)} диагнозов.")
    # --- ГАРАНТИРОВАННЫЙ ВОЗВРАТ РЕЗУЛЬТАТА ---
    return results
    # ------------------------------------------

