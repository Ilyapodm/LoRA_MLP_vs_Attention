# V2: равный бюджет параметров и фиксированный ранг

Редакция 27.11.2026. Это инструкция **активного** протокола food101-local-v2.
Научное описание — [02-protocol.md](02-protocol.md), анализ — [05-analysis.md](05-analysis.md).
Старые сценарии удалены из рабочего дерева; полный снимок до чистки — Git
`1c074e7`. Пилотные results/rtx* не удалены и не используются как десятиэпоховые
пробы v2. Актуальный CLI один: study_queue.py.

## Что запускаем

Шесть уникальных конфигураций: LP, A-iso, M-iso, B-iso, A-16, B-16.
M-16 — только название M-iso во второй серии, повторного обучения нет.
Каждая LR-проба: весь train18180/val2020, 10 эпох, seed142, один из
LR {0.0001, 0.001, 0.01, 0.03}. Итого 24 пробы. Final: выбранный LR каждой
конфигурации × seeds42/43/44, заново с pretrained, 18 запусков. Всего 42 новых
обучения; прежние 12 пятиэпоховых проб — отдельный пилот.

Профиль общий: microbatch16/accumulation32/eval32, FP32 weights + AMP FP16,
SDPA, TF32 off, workers0. 36 attempted updates за эпоху, 360 за полный run.
Полные группы — 512 изображений, последняя — 260. При OOM профиль пересматривают
для всей серии **до** запуска LR, а не отдельно для метода или GPU.

## 1. Подготовка

Тот же Python3.12, CUDA build12.6 и requirements.txt, что в пилоте.
Новых ML-зависимостей нет. Получить один согласованный checkout/архив v2.
Прежние data_manifest.json/model_revision.json/dataset_revision.json оставить
в artifacts/protocol. Использовать **тот же HF cache**, что при пилоте,
восстановив прежние HF_HOME/HF_HUB_CACHE/HF_DATASETS_CACHE в новом терминале.
Не менять эти пути на новые без необходимости: это вызывает повторное скачивание.

```bash
python data.py --verify --manifest artifacts/protocol/data_manifest.json
python -m unittest discover -s tests -v
```

Stdlib-проверки плана и целостности работают без ML; CPU integration использует
маленькие модели, полные counts и десять эпох. Они не являются научными accuracy
или GPU benchmark. GPU-приёмка выполняется отдельно на каждом ПК.

## 2. Общий LR-план

Координатор создаёт план один раз после фиксации вычислительных исходников:

```bash
python study_queue.py --make-lr-plan --protocol artifacts/protocol --plan runs-v2.json
```

Если runs-v2.json уже предоставлен вместе с исходниками, переиспользовать его.
Существующий план не перезаписывается. После изменения кода до серии нужен
новый путь плана, например runs-v2-release2.json, и новый preflight всех GPU.
После начала серии исходники исполнителей не обновляют.

Для согласованного OOM fallback до серии есть `--runtime-profile FILE.json`
только у --make-lr-plan: полный JSON профиля с теми же ключами, что common.profile
в плане. Допустимы только 16/32,8/64,4/128, eval32/16/8 и общий boolean
gradient_checkpointing; precision/backend/workers остаются фиксированными.
Новый профиль требует нового имени плана и нового допуска всех GPU.

Все три ПК получают одинаковый runs-v2.json. Посмотреть восемь своих заданий:

```bash
python study_queue.py --dry-run --plan runs-v2.json --protocol artifacts/protocol --executor rtx3060
```

Ключ executor заменить своим: rtx4070ti / rtx3090 / rtx3060. На каждой карте
присутствуют все шесть методов; дополнительные два задания равномерно
распределяют четвёртые LR. Всего по восемь задач на GPU. Точное назначение
хранится в плане, run ID не зависит от абсолютных путей.

## 3. Новый preflight каждой GPU

```bash
python study_queue.py --preflight --plan runs-v2.json --protocol artifacts/protocol --executor rtx3060 --output results/v2/lr
```

Старый v1 admission не допускается. Проверяются:

1. Фактические GPU, Python/package/CUDA versions и запись на локальный диск.
2. Все train/val изображения и совпадение manifest/revisions/class mapping.
3. Шесть моделей, включая A-16/B-16: точные targets/ranks/alpha/whitelist,
   одинаковая head, B=0, gradients, три применённых updates и frozen weights.
4. Реальный pretrained **B-16** через Trainer: 772/32 × 2 эпохи, общий профиль,
   группы 512+260, validation/checkpoints/best reload и учёт AMP skips.

Успех напечатает собственный admission.json. Он закрепляет hash LR-плана,
release/profile и фактическое окружение. Не редактировать или обезличивать этот
файл и environment.json. Передавать исходные файлы в архиве, не через копирование
текста в редактор/чат. Проверка hashes при сборе ожидает исходные байты.

## 4. Выполнить LR-очередь

```bash
python study_queue.py --run --plan runs-v2.json --protocol artifacts/protocol --executor rtx3060 --output results/v2/lr --admission ПУТЬ_ИЗ_V2_PREFLIGHT/admission.json
```

Один run — отдельный процесс. Очередь последовательная, карты независимы.
Результаты: results/v2/lr/<executor>/runs/<run_id>/attempts/<attempt_id>.
Окружение и receipt проверяются перед очередью и каждым worker.

Готовые COMPLETE проверяются и пропускаются. Numerical_invalid расходует один
кандидат LR, сохраняет причину и допускает переход к следующему. Technical error
останавливает очередь: после диагностики та же команда с --retry-failed создаёт
новую attempt с тем же config/LR/seed и исходными весами. Старые попытки не
перезаписываются. Низкое качество не является причиной retry. Stale queue.lock
после аварии удаляют вручную только убедившись, что старый процесс остановлен.

## 5. Собрать LR и зафиксировать final

Получить три каталога с исходными файлами, включая best/, checksums.json,
COMPLETE, config/environment/status/history/updates/metrics. Полные checkpoints
не требуются для приёмки завершённой пробы, но сохраняются локально до неё.

```bash
python study_queue.py --collect --plan runs-v2.json --protocol artifacts/protocol --results results/v2/lr/rtx4070ti results/v2/lr/rtx3090 results/v2/lr/rtx3060 --output selected-lrs-v2.json
python study_queue.py --make-final-plan --lr-plan runs-v2.json --plan final-runs-v2.json --protocol artifacts/protocol --results results/v2/lr/rtx4070ti results/v2/lr/rtx3090 results/v2/lr/rtx3060 --selection selected-lrs-v2.json
```

Нужны все 24 разрешённых исхода. Победитель — максимальный best val correct_count,
tie-break меньший LR; best epoch внутри run — ранняя при равенстве. Если все
четыре LR метода numerical_invalid, final блокируется. Произвольного fallback,
расширения сетки или выбора по test нет. --selection можно опустить: при
создании final LR всё равно пересчитываются из проверенных исходных артефактов.

Создаются final-runs-v2.json и artifacts/protocol/freeze-v2.json. Для другого
пути freeze есть --freeze ПУТЬ, его затем передают всем final-командам.
Final требует тех же source/profile/data, что v2 LR. Freeze содержит шесть LR,
состав обеих серий и 18 итоговых моделей. Всем ПК передать одинаковые план,
freeze и исходники; исходные LR-результаты достаточно хранить у координатора.

## 6. Final-очереди

По шесть методов на GPU, один seed-блок на карте:

| GPU | Seed | Порядок |
| --- | ---: | --- |
| rtx4070ti | 42 | LP, A-iso, M-iso, B-iso, A-16, B-16 |
| rtx3090 | 43 | A-iso, M-iso, B-iso, A-16, B-16, LP |
| rtx3060 | 44 | M-iso, B-iso, A-16, B-16, LP, A-iso |

```bash
python study_queue.py --dry-run --plan final-runs-v2.json --protocol artifacts/protocol --executor rtx3060
python study_queue.py --run --plan final-runs-v2.json --protocol artifacts/protocol --executor rtx3060 --output results/v2/final --admission ПУТЬ_ИЗ_V2_PREFLIGHT/admission.json
```

Используется свой **v2** LR admission: Trainer/release/profile/окружение не
изменились. Каждый final начинается заново; LR-веса и LP-веса не переносятся.
Seed задания управляет head/LoRA, Trainer, sampler и аугментациями. Научный CLI
не принимает произвольные LR/seed/epochs, probe, resume или остановку по эпохе.

Успех — TRAINED и training_checksums.json. Training_status.json неизменяем,
status.json — отдельный progress для будущей оценки. Метрики/веса обучения
не дописываются test-данными; test-поля остаются null. Numerical_invalid final
получает TRAINING_FAILED и **блокирует** серию; --retry-failed не повторяет его
и не заменяет LR/seed. Технические сбои допускают новую полную попытку после
диагностики. Повторный запуск проверяет и пропускает TRAINED.

## 7. Собрать подтверждённые обучения

```bash
python study_queue.py --collect --plan final-runs-v2.json --protocol artifacts/protocol --results results/v2/final/rtx4070ti results/v2/final/rtx3090 results/v2/final/rtx3060 --output trained-finals-v2.json
```

Сбор требует 18 trained моделей, проверяет 10 val-эпох/360 updates/best/weights
checksums и сохраняет обе серии с общей M-iso, а также план 18 статистических
сравнений. test_evaluated=false: TRAINED не означает завершение исследования.

wall_train_job включает train+val+checkpoint+best reload. Export и setup/download
измеряются отдельно. GPU peaks относятся к train+val job. Стоимость сравнивают
внутри одного GPU/профиля; времена разных ПК не дают чистого рейтинга методов.

Оценка test с проверкой freeze, загрузка компактных best, predictions, norms,
реализация статистических расчётов и final COMPLETE остаются следующим этапом.
Определения анализа зафиксированы в плане, но test пока не открывается.

## Распространение исходников

После создания общего runs-v2.json можно собрать небольшой source-only архив:

```bash
python package_study.py --output artifacts/study-v2-source.zip
```

Архив содержит только актуальные исходники, tests/docs и общий план LR v2,
manifest с hashes; без устаревших сценариев, результатов, весов, cache, venv и .git. Data protocol
передаётся отдельно прежним data-protocol.zip. Final-план и freeze появятся позже,
после 24 LR-исходов, и передаются отдельно.
