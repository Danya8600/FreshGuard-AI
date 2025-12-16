# FreshGuard AI — Fruit Classifier (Flask + PyTorch)

Веб‑приложение и CLI‑утилита для классификации фруктов по фото с использованием PyTorch модели.  
Проект демонстрирует применение принципов **SOLID**, паттернов проектирования (GoF: порождающие/структурные/поведенческие) и паттернов распределения обязанностей **GRASP** без изменения логики ML‑инференса. [web:48][web:72]

## 🚀 Возможности

- **Web UI (Flask)**: загрузка изображения → предсказание класса и вероятность (top‑1). [web:30]
- **Превью изображения**: до отправки (FileReader) и после нажатия `Predict` (data URL / base64). [web:121]
- **CLI/batch inference**: обработка папки изображений с сохранением результатов в `predictions.csv` (top‑k).
- **Общая инфраструктура**: инференс используется и в web, и в batch (одна точка правды).

## 🛠️ Стек

| Компонент | Версия |
|-----------|--------|
| Python | 3.9+ |
| Flask | ✅ |
| PyTorch | ✅ |
| torchvision | ✅ |
| Pillow | ✅ |

## 📁 Структура проекта


├── app.py # Flask UI + GRASP Controller

├── infer_freshguard.py # Batch inference (top-k)

├── model.py # Ядро + SOLID + паттерны

├── README.md # Документация

└── out_freshguard/

└── best_model.pth # Чекпоинт модели

## 🚀 Быстрый старт

### 1) Установка

python -m venv .venv

Windows:
.venv\Scripts\activate

Linux/Mac:
source .venv/bin/activate

pip install -U pip
pip install flask pillow torch torchvision

text

### 2) Чекпоинт

Требуется файл `./out_freshguard/best_model.pth`:
{
"model_state": state_dict,
"classes": ["apple_fresh", "apple_rotten", ...],
"config": {"model_name": "mobilenet_v3_small", "img_size": 192}
}

text

### 3) Запуск Web UI

python app.py

text
[http://127.0.0.1:5000/](http://127.0.0.1:5000/)

### 4) Batch inference

1. Изображения → `./real_photo/`
2. ```
   python infer_freshguard.py
Результат → ./out_freshguard/predictions.csv

🔄 Pipeline инференса
PIL Image
  ↓ Resize(img_size) → ToTensor() → Normalize()

  ↓ model.forward() → logits

  ↓ softmax() → probabilities

  ↓ Postprocessor (Top1/TopK)

🏗️ Архитектура (SOLID + Patterns + GRASP)

SOLID принципы

Принцип	Реализация

SRP	model.py: отдельные классы для каждой обязанности

OCP	Новая модель → расширяем TorchvisionModelFactory

LSP	IPostprocessor реализации взаимозаменяемы

ISP	Узкие интерфейсы: IPreprocessor, IClassifier, IPostprocessor

DIP	Web/CLI → абстракции, PyTorch → через build_predictor_from_ckpt()

GoF паттерны

Тип	Паттерн	Класс

Порождающий	Factory Method	TorchvisionModelFactory.create()

Структурный	Facade	PredictorService.predict()

Поведенческий	Strategy	Top1Postprocessor ↔ TopKPostprocessor

GRASP паттерны

Паттерн	Реализация

Controller	app.py:index() делегирует сервису

Information Expert	Постпроцессоры знают "как интерпретировать вероятности"

Pure Fabrication	CsvPredictionWriter — чистый IO-класс

Low Coupling	Слои общаются через интерфейсы

📊 Пример CSV результата

filename,pred_class,pred_prob,top1_class,top1_prob,top2_class,top2_prob,top3_class,top3_prob
img1.jpg,apple_fresh,0.92,apple_fresh,0.92,apple_rotten,0.05,banana_fresh,0.02
⚠️ Примечания
Base64 в Web UI: изображение после Predict сохраняется в HTML (data URL). [web:121]

CPU по умолчанию: для Web UI (device=None). Batch использует GPU если доступен.

Совместимость: логика идентична оригиналу (тот же softmax + argmax + round(..., 2)). [web:105]

🔮 Расширение
Top‑3/5 в Web UI

Замер времени инференса

Логирование (Observer)

Аугментации в реальном времени

FreshGuard AI — учебный проект, демонстрирующий современную архитектуру ML-приложения с разделением обязанностей и принципами SOLID/GRASP.