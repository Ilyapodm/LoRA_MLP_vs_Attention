# LoRA: Attention vs MLP vs Both

Текущий этап реализации: **шаги 1–2 — окружение и сборка модели через PEFT**.
Научный протокол согласован в [документах](lora_docs_updated/02-protocol.md):
ViT-Base in21k, Food-101, 18 180 train / 2 020 val / 25 250 test, пять эпох,
12 LR-проб и 12 итоговых моделей. В LoRA-методах по 1 552 229 trainable params,
LP — 77 669. Food-101, Trainer, сохранение/resume обучения и итоговая статистика
**ещё не реализованы**. `test_params.py` — техническая проверка, не train.py.

## Что уже есть

- `configs.py`: LP, A-iso (r20), M-iso (r16), B-iso (r10/r8), точный выбор Linear,
  штатный PEFT, общий старт head, аудит ranks/alpha/scaling и trainable whitelist.
- `bootstrap.py`: информация о Python/пакетах/CUDA/диске, безопасная проверка записи
  в существующую постоянную папку. Сам не монтирует Drive и не подтверждает
  облачную доставку локальной папки ПК.
- `test_params.py`: ViT-Base с реальными размерами, три шага на искусственных входах,
  проверка loss/gradients, обновления head и неизменности frozen weights.
- `colab.ipynb`: установка, загрузка архива исходников с SHA-256, Drive и GPU-проверки.
- `tests/test_configs.py`: проверки бюджета, форм и очереди без ML-пакетов.

## Без зависимостей

```bash
python3 -m unittest discover -s tests -v
python3 bootstrap.py --output artifacts/environment.json
```

## CPU-проверка на отдельном окружении

Нужен Python 3.11 или 3.12. Не устанавливайте пакеты в системный Python.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python test_params.py --output artifacts/cpu-smoke.json
```

По умолчанию CPU smoke использует искусственные 32×32 изображения с интерполяцией
position embeddings. Размеры весов, 12 блоков и ранги настоящие; это экономит
время проверки gradients, **не проверяет память обучения 224×224**.

`requirements.txt` фиксирует bootstrap-кандидат. Он не помечается принятым общим
окружением до проверки Colab/локальной GPU и фиксации полного `pip freeze`.

## Google Colab: пошагово

1. Создать архив исходников:

   ```bash
   python3 package_bootstrap.py
   ```

2. Открыть `colab.ipynb` через Upload notebook на [Google Colab](https://colab.research.google.com/).
3. Выбрать GPU, выполнить установку пакетов, перезапустить session.
4. Продолжить с ячейки загрузки `artifacts/colab-bootstrap.zip`.
5. Подключить Drive, указать свою папку/ID, выполнить проверки по порядку.
6. Проверить отчёты на Drive. При ошибке остановиться; не считать тест пройденным.

Синтетический GPU smoke проверяет 224×224, а отдельный профиль Both — 16×32
microbatch/accumulation. Он не заменяет проверку на реальных изображениях,
Trainer, validation и переносимого resume на следующем этапе.

Для загрузки pretrained weights требуется полный model commit SHA. Notebook
сохраняет его один раз в общем `protocol/model_revision.json`; все участники
используют это значение, не разрешают main независимо. Качество на синтетических
данных не является результатом эксперимента.

## Чего ждать от проверок

`test_params.py` не требует скачивания модели, если не передан `--pretrained`.
Выходной JSON имеет `stage=technical` и `scientific_result=false`. Существующий
отчёт не перезаписывается: выберите новый `--output`. Искусственные входы не
используются для LR search. Test split Food-101 здесь вообще не читается.

Следующие этапы: `data.py` и общий manifest, затем `train.py` с Trainer,
проверка accumulation/save/resume/sync, runtime profile, и только потом LR search.
