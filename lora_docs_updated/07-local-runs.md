# Локальные LR-очереди: инструкция

Редакция 26.11.2026. Основной ресурс — три постоянных ПК. Colab необязателен.
Каждый run целиком выполняется на одной карте. Прерванные техническими ошибками
runs повторяются с начала с теми же config/LR/seed, в новой attempt-папке.
Переносимый resume и побитовое совпадение продолжения не являются допуском к LR search.
Все научные настройки из 02-protocol.md сохранены: 101 класс, 18 180/2 020,
пять эпох, три LR для каждого из LP/A/M/B, seed поиска 142.

## 1. Одинаковое окружение и данные

Репозиторий клонируется каждым владельцем; все используют одну согласованную версию:

```bash
git clone https://github.com/Ilyapodm/LoRA_MLP_vs_Attention.git
cd LoRA_MLP_vs_Attention
git rev-parse HEAD
```

На каждом ПК: Python **3.12**, отдельный venv, одна видимая CUDA GPU.
Linux/macOS (обучение с CUDA — на поддерживаемом ПК NVIDIA):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Windows CMD:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
```

В PowerShell можно обходиться без активации: вместо `python` во всех следующих
командах использовать `.\.venv\Scripts\python.exe`.
Файл .gitattributes сохраняет LF и одинаковые байты исходников при клонировании
на Windows/Linux; это важно для проверки source_hash в плане.
Установить PyTorch из официального cu126 индекса, затем requirements:

```bash
python -m pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

Команды ниже выполняются Python активированного venv из корня проекта.
Общий `data-protocol.zip` координатор передаёт отдельно: архив находится в его
artifacts/, игнорируется Git и сам при clone не появится. Изображений в нём нет.
Распаковать содержимое архива непосредственно в `artifacts/protocol/`, без лишней
вложенной папки. Предварительная офлайн-проверка:

```bash
python data.py --verify --manifest artifacts/protocol/data_manifest.json
```
Там должны быть data_manifest.json, model_revision.json, dataset_revision.json.
Код/manifest/revisions одинаковые на всех ПК. Parquet+Arrow Food-101 занимают
около 10 GiB. Изображения и checkpoints не читаются с Drive во время обучения.

**Preflight и очередь должны использовать одну и ту же папку HF cache.** Если
preflight уже выполнен со стандартным cache и HF_HOME не задавался, можно
продолжать без HF_HOME: данные будут переиспользованы из стандартной папки.
Менять путь после загрузки не нужно — другой путь вызовет повторное скачивание.
Одинаковый абсолютный путь на разных ПК не требуется.

Для новой установки удобно выбрать cache внутри проекта. Выполнить из корня
репозитория **до preflight** и сохранять те же настройки при запуске очереди.
Linux/bash:

```bash
export HF_HOME="$PWD/cache/huggingface"
export HF_HUB_DISABLE_XET=1
```

Windows CMD:

```bat
set "HF_HOME=%CD%\cache\huggingface"
set "HF_HUB_DISABLE_XET=1"
```

PowerShell:

```powershell
$env:HF_HOME = Join-Path (Get-Location).Path 'cache\huggingface'
$env:HF_HUB_DISABLE_XET = '1'
```

Эти переменные действуют в текущем терминале и наследуются всеми заданиями
очереди. В новом терминале повторить настройку до запуска. HF_HUB_DISABLE_XET=1
выбирает обычную HTTP-загрузку вместо Xet; на настройки обучения не влияет.
Если ранее отдельно задавались HF_HUB_CACHE или HF_DATASETS_CACHE, они также
должны указывать на те же папки при preflight и --run.

## 2. Один общий план

Готовый `runs.json` включён в репозиторий; координатор передаёт тот же файл всем. Посмотреть свои задания:

```bash
python run_queue.py --dry-run --plan runs.json --protocol artifacts/protocol --executor rtx4070ti
```

План содержит полный набор 12 заданий и hash вычислительных исходников, данных,
model/dataset revisions и общего профиля. После изменения кода старый план
блокируется: до первого научного запуска создаём новый. Не редактировать план
после начала серии. Изменение профиля требует нового плана и повторной приёмки.
Исходный профиль: microbatch16/accumulation32/eval32, AMP FP16, SDPA, TF32 off,
workers0. Другой общий профиль выбирается только до серии по согласованным OOM-правилам.

## 3. Допуск каждой карты

Заменить executor на имя своего ПК: rtx4070ti / rtx3090 / rtx3060.
`--output` — локальный постоянный диск, не временная VM и не Drive.

```bash
python run_queue.py --preflight --plan runs.json --protocol artifacts/protocol --executor rtx4070ti --output results
```

Команда проверит CUDA/версии, все выбранные изображения, модули/gradients LP/A/M/B
и короткий Both через Trainer с общим профилем. Успех создаёт admission.json;
его путь печатается в конце. При ошибке посмотреть logs внутри preflight-папки.
Каждый ПК может начать свою очередь после собственного успешного preflight;
проверки остальных выполняются параллельно. Общие код, план и профиль фиксированы.
Если потребуется изменить их, результаты разных планов не смешиваются.
Состояние на 26.11.2026: RTX 3060 12GB прошла полный preflight на Linux,
Python 3.12.13, torch 2.8.0+cu126. Все 18 180/2 020 изображений и проверки
LP/A/M/B прошли. Короткий pretrained Both 772/32 × 2: job 20,43 с,
allocated/reserved 2,312/2,496 GiB, четыре applied updates без skips.
Локальное свидетельство координатора:
`results/rtx3060/preflight/c8f2aa82e5a14c0ab28da3de94fcf61a/admission.json`.
Preflight 4070 Ti и 3090 ещё ожидается. У каждого ПК будет собственный admission;
файл от 3060 нельзя использовать для допуска другой карты.

## 4. Запуск своей очереди

```bash
python run_queue.py --run --plan runs.json --protocol artifacts/protocol --executor rtx4070ti --output results --admission ПУТЬ_К_admission.json
```

Одна команда последовательно выполнит четыре задания данного ПК. Каждая модель
обучается отдельным процессом; по завершении память освобождается. Каждый run
использует полный manifest, 5 эпох, AdamW/linear scheduler и выбирает best по
целому val correct_count; при равенстве — ранняя эпоха. Test loader отсутствует.
Технические поднаборы, dummy-метрика и --resume-bundle в LR-режиме запрещены.

Команду запускать с теми же HF_HOME/HF_HUB_CACHE/HF_DATASETS_CACHE, что и preflight.
Повторная загрузка гигабайтов при уже подготовленных данных — повод сначала
сверить эти пути. Во время первоначальной загрузки GPU может простаивать;
прогресс Trainer появится после подготовки данных и модели.

Готовые результаты с COMPLETE/hashes проверяются и пропускаются при повторном
запуске очереди. Numerical_invalid сохраняется как исход поиска, очередь идёт
дальше; низкая конечная accuracy тоже является валидным результатом.
Техническая ошибка останавливает очередь, причина и log сохраняются. После
диагностики повторить ту же команду с `--retry-failed`; будет новая attempt,
старые файлы сохраняются. Это не подбор seed и не выбор лучшей попытки по качеству.
`queue.lock` защищает от двух очередей в одном executor-каталоге. После аварийного
выключения stale lock удаляется вручную только после проверки, что старый процесс остановлен.

Trainer checkpoints локальные, последние две версии и ранний best сохраняются
штатным Trainer. Компактный best export: LP head, PEFT LoRA+classifier.
Результат: config/environment/status/history/updates/metrics, best/, checksums.json,
COMPLETE последним. Test-поля null. COMPLETE ставится только для valid/numerical_invalid.

## 5. Распределение

| ПК | LP | Attention A-iso | MLP M-iso | Both B-iso |
| --- | --- | --- | --- | --- |
| RTX 4070 Ti | 1e-4 | 1e-3 | 1e-2 | 1e-4 |
| RTX 3090 | 1e-3 | 1e-2 | 1e-4 | 1e-3 |
| RTX 3060 | 1e-2 | 1e-4 | 1e-3 | 1e-2 |

Это четыре LR-прогона на карту. Каждый метод выполняется на каждой карте один
раз: не закрепляем Attention/MLP/Both за разным оборудованием. Это начальное
равномерное назначение, не обещание одинакового времени — скорости ПК ещё не
измерены. Разные GPU остаются ограничением сравнения LR; побитового совпадения
между картами не предполагаем. Стоимость методов строго сравнивается только
в одинаковых условиях, иначе публикуем фактические времена с оборудованием.

После выбора LR предлагается по четыре final run на ПК:
4070 Ti — seed42 (LP,A,M,B); 3090 — seed43 (A,M,B,LP); 3060 — seed44 (M,B,LP,A).
Все четыре метода одного seed остаются на одной карте. Настройки batch/precision
одинаковые независимо от объёма VRAM. Эта очередь запускает только LR search;
final-план и обучение реализованы отдельно в [08-final-runs.md](08-final-runs.md).
Оценка test — следующий этап.

## 6. Сбор и выбор LR

Передать координатору папки исполнителей после завершения очередей, включая
COMPLETE и best exports. Непрерывная синхронизация не требуется.

```bash
python run_queue.py --collect --plan runs.json --protocol artifacts/protocol --results results/rtx4070ti results/rtx3090 results/rtx3060 --output selected_lrs.json
```

Нужны все 12 разрешённых исходов. Проверяются hashes, идентичность плана/заданий,
пять эпох и размер val, отсутствие test-метрик; дубли и повреждения блокируют сбор.
Выбор LR каждого метода — максимум correct_count, при равенстве меньший LR.
На границе сетку не расширяем. Если все три LR метода numerical_invalid — остановка
и диагностика, произвольного fallback нет. selected_lrs.json не является разрешением
читать test. Final freeze/очередь описаны в [08-final-runs.md](08-final-runs.md),
evaluation ещё не реализован.

Новый план до начала серии создаётся в новом файле: `python run_queue.py --make-plan --protocol artifacts/protocol --plan runs-new.json`; затем этот путь указывается во всех командах. Существующий план не перезаписывается.


## Какие метрики сохраняются

LR — гиперпараметр, не метрика. В каждом научном LR run сохраняются:

| Файл | Содержимое |
| --- | --- |
| history.json | train loss/grad norm/LR по optimizer steps; val loss, accuracy, correct_count/sample_count после каждой из пяти эпох |
| metrics.json | best val accuracy/count, best epoch, завершённые эпохи/шаги; wall train job, val/save время и train residual; GPU allocated/reserved peak; applied/skipped updates |
| updates.json | LR после обновления, AMP scale и флаг пропуска каждой попытки |
| environment.json | фактическая GPU/VRAM, Python, библиотеки, CUDA build и ОС |
| config.json | полный общий план, method/LR/seed и закреплённые source/data/model/profile |
| best/ | веса лучшей модели и metadata |

Train accuracy сейчас отдельно не вычисляется. Test-поля в LR search null;
test accuracy, предсказания и статистика, а также LoRA norms относятся к будущему
final/evaluate этапу. Время включает реальное чтение/преобразование данных;
GPU peak здесь общий train+val, не чистый backward. Загрузка данных/модели до
старта Trainer не входит в wall_train_job_seconds. train loss — агрегат по
логируемой accumulation-группе, не отдельная оценка всего train после эпохи.

## 3060 и завершение этапа

Очереди независимые: нет синхронизации batch между картами. Более быстрый ПК
выполняет тот же microbatch быстрее и не ждёт 3060 на каждом шаге. Однако весь
этап LR заканчивается после последней очереди; равные четыре задания могут
создать хвост на более медленном ПК. Сейчас это начальное назначение, не
оптимизация по ещё не измеренным скоростям.

16/32 выбран как общий исходный профиль и теперь подтверждён preflight на 3060.
В коротком Trainer Both на T4 peak allocated/reserved был 2,312/2,496 GiB;
Проверки 4070 Ti и 3090 ещё необходимы. Больший microbatch на быстрой карте
может повысить throughput, но требует замера. Разные microbatch при том же
512 меняют разбиение dropout/FP16 операций, поэтому текущий план оставляет
общий профиль. Если он не проходит, профиль пересматриваем до научной серии,
создаём новый план и повторяем допуск всех карт.

После сбора 12 LR результатов selected_lrs.json даёт четыре выбранных LR.
Далее используются final/freeze: новые модели с исходных весов, seeds42/43/44,
пять эпох и best по val; команды — в [08-final-runs.md](08-final-runs.md).
Однократная оценка test25250 ещё не реализована. Анализ качества — по итоговым моделям и трём seed;
точный рейтинг скорости разных GPU из этих данных не строится.

Если в уже существующем venv возникает `No module named pip`, выполнить
`python -m ensurepip --upgrade`, затем повторить установку. Точные суффиксы
`+cu126` в команде выше заменяют ранее установленный CPU-вариант PyTorch;
`torch==2.8.0` без суффикса может посчитать `2.8.0+cpu` уже подходящей версией.
