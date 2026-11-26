# LoRA: Attention vs MLP vs Both

Текущий этап реализации: **шаги 1–3 — окружение, сборка PEFT и подготовка данных**.
Научный протокол согласован в [документах](lora_docs_updated/02-protocol.md):
ViT-Base in21k, Food-101, 18 180 train / 2 020 val / 25 250 test, пять эпох,
12 LR-проб и 12 итоговых моделей. В LoRA-методах по 1 552 229 trainable params,
LP — 77 669. Подготовка Food-101 реализована в data.py, но manifest на исходном датасете ещё
не создан/проверен. Trainer, сохранение/resume обучения и итоговая статистика
**ещё не реализованы**. `test_params.py` — техническая проверка, не train.py.

## Что уже есть

- `configs.py`: LP, A-iso (r20), M-iso (r16), B-iso (r10/r8), точный выбор Linear,
  штатный PEFT, общий старт head, аудит ranks/alpha/scaling и trainable whitelist.
- `data.py`: PCG64-выборка 180/20 на класс, immutable manifest с hashes изображений,
  точные transforms, отдельные RNG данных и epoch sampler; test loader отсутствует.
- `bootstrap.py`: информация о Python/пакетах/CUDA/диске, безопасная проверка записи
  в существующую постоянную папку. Сам не монтирует Drive и не подтверждает
  облачную доставку локальной папки ПК.
- `test_params.py`: ViT-Base с реальными размерами, три успешных обновления на искусственных входах,
  проверка loss/gradients, обновления head и неизменности frozen weights.
  AMP skips и scale сохраняются; десять пропусков подряд останавливают проверку.
- `colab.ipynb`: установка, загрузка архива исходников с SHA-256, Drive и GPU-проверки.
- `tests/test_configs.py`: проверки бюджета, форм и очереди без ML-пакетов.

## Без зависимостей

```bash
python3 -m unittest discover -s tests -v
python3 bootstrap.py --output artifacts/environment.json
```

## CPU-проверка на отдельном окружении

Локально проверен Python 3.12, в Colab — 3.13. Не устанавливайте пакеты в системный Python.

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
3. Выбрать GPU, загрузить архив и установить отдельный venv по ячейкам. Kernel не перезапускать.
4. Архив для загрузки: `artifacts/colab-bootstrap.zip`; запуск project scripts идёт через Python отдельного venv.
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

Следующие этапы: создание общего manifest на реальных данных, затем `train.py` с Trainer,
проверка accumulation/save/resume/sync, runtime profile, и только потом LR search.

## Подготовка Food-101 (шаг 3)

Один раз координатор закрепляет commit SHA публичного датасета и передаёт его всем.
Команда использует официальный train, не создаёт test loader. Возможно скачивание
полного split/cache; требуется свободный диск и доступ к Hugging Face.

```bash
.venv/bin/python data.py --dataset-revision FULL_40_CHARACTER_SHA --manifest artifacts/protocol/data_manifest.json
.venv/bin/python data.py --verify --manifest artifacts/protocol/data_manifest.json
```

Повторная подготовка не перезаписывает существующий manifest. В нём ровно
18 180 train и 2 020 val, SHA-256 исходных изображений и самопроверяемый content hash.
`--verify` проверяет структуру/баланс/ID/хеш manifest офлайн, не заменяет проверку
image bytes при подготовке и чтении. EpochSampler должен подключаться к Trainer
с передачей номера эпохи; это интегрируется и проверяется на следующем этапе.

Если проверочный Python был установлен в /tmp и после перерыва исчез, .venv может
содержать битую ссылку. Пересоздайте его с существующим Python 3.12. Для окружения
самого агента runtime теперь хранится под игнорируемым .runtime/, а не /tmp.

## Выполненные проверки (26.11.2026)

- 10 unittest-проверок прошли без пропусков на Python 3.12.13 / CPU PyTorch 2.8.0.
- Настоящая структура ViT-Base + PEFT: LP/A/M/B по три optimizer steps на CPU,
  отдельно синтетические входы 32×32 и 224×224. Совпали начальные head и числа
  параметров; адаптеры получают gradients, frozen weights не меняются.
- Evidence: artifacts/cpu-smoke-20261126.json, cpu-smoke-224-20261126.json,
  environment-20261126.json, cpu-freeze-20261126.txt. Артефакты игнорируются Git.
- CPU-проверки отдельно не подтверждают CUDA/AMP/VRAM или качество на Food-101.

В актуальном Colab проверено подключение Tesla T4. Системный runtime использует
Python 3.13 и более новый PyTorch; notebook устанавливает закреплённый стек в
/content/lora_env и не заменяет kernel. Обычный venv в Colab может не иметь ensurepip,
поэтому используем venv --without-pip и host pip --python для установки пакетов.
GPU-проверки и замеры считаются пройденными только по завершённым отчётам.

На Tesla T4 (14,56 GiB) завершены проверки CUDA/AMP для LP/A/M/B на 224×224:
каждый метод выполнил три успешных обновления, одинаковая head и frozen weights
проверены. У каждого был один стартовый AMP overflow: scale 65536→32768,
после чего три обновления прошли. Исходная строгая проверка gradients ошибочно
останавливалась на этом допустимом пропуске; исправлена обработка GradScaler.

Both с batch 16 / accumulation 32 выполнил три обновления без OOM и skips:
peak allocated **2,034 GiB**, reserved **2,102 GiB**, цикл **16,29 с**.
Это синтетические входы, без реального data loader, validation и Trainer;
скорость не переносим в бюджет 24 обучений. Pretrained Both тоже прошёл три
обновления; model SHA: `b4569560a39a0f1af58e3ddaf17facf20ab919b0`.

Отчёты, полный pip freeze и SOURCE_MANIFEST сохранены локально в
`artifacts/colab-T4-20261126/`. Перенос проверен SHA-256; исходный неуспешный
smoke и вывод десяти успешных unittest сохранены в `initial-attempt/`.
Проверенная computational source release: `839e471a2f4390ad4d8bd08e789c3aa260570cd43060afb582cbce47f5df544e`.
Последующие изменения README/notebook меняют hash архива; они не являются
повторным GPU-запуском. Стек пока принят только для этой технической проверки:
данные, Trainer/resume/sync и локальные GPU ещё требуют приёмки.
