# Локальные LR-очереди: инструкция

Редакция 26.11.2026. Основной ресурс — три постоянных ПК. Colab необязателен.
Каждый run целиком выполняется на одной карте. Прерванные техническими ошибками
runs повторяются с начала с теми же config/LR/seed, в новой attempt-папке.
Переносимый resume и побитовое совпадение продолжения не являются допуском к LR search.
Все научные настройки из 02-protocol.md сохранены: 101 класс, 18 180/2 020,
пять эпох, три LR для каждого из LP/A/M/B, seed поиска 142.

## 1. Одинаковое окружение и данные

На каждом ПК: Python **3.12**, отдельный venv, одна видимая CUDA GPU.
Установить PyTorch из официального cu126 индекса, затем requirements:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

Команды ниже выполняются Python активированного venv из корня проекта.
Общий `data-protocol.zip` распаковать в `artifacts/protocol/`.
Там должны быть data_manifest.json, model_revision.json, dataset_revision.json.
Код/manifest/revisions одинаковые на всех ПК. Для cache можно задать HF_HOME
на локальный диск; Parquet+Arrow Food-101 занимают около 10 GiB.
Изображения и checkpoints не читаются с Drive во время обучения.

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
Сначала пройти preflight на всех трёх ПК, затем начать LR search.
В этой среде агентом не измерены эти три GPU; справка не означает их фактический допуск.

## 4. Запуск своей очереди

```bash
python run_queue.py --run --plan runs.json --protocol artifacts/protocol --executor rtx4070ti --output results --admission ПУТЬ_К_admission.json
```

Одна команда последовательно выполнит четыре задания данного ПК. Каждая модель
обучается отдельным процессом; по завершении память освобождается. Каждый run
использует полный manifest, 5 эпох, AdamW/linear scheduler и выбирает best по
целому val correct_count; при равенстве — ранняя эпоха. Test loader отсутствует.
Технические поднаборы, dummy-метрика и --resume-bundle в LR-режиме запрещены.

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
одинаковые независимо от объёма VRAM. Final CLI/test gate — следующий этап, эта
очередь запускает только LR search.

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
читать test: final freeze/очередь и evaluation ещё нужно реализовать.

Новый план до начала серии создаётся в новом файле: `python run_queue.py --make-plan --protocol artifacts/protocol --plan runs-new.json`; затем этот путь указывается во всех командах. Существующий план не перезаписывается.
