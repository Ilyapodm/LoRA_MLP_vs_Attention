# Final: план, обучение и локальные очереди

Следующий этап после всех **12** LR-исходов. Реализованы фиксация дизайна,
12 обучений и сбор подтверждённых training-артефактов. Оценка test, predictions,
LoRA norms и итоговая статистика — отдельный следующий этап; эти команды test
не открывают и не создают final COMPLETE.

## 1. Общие условия

Сохраняются прежние модель/revisions, manifest 18 180/2 020, пять эпох,
AdamW/linear decay, FP32+AMP FP16 и общий runtime profile. Каждая модель создаётся
заново из pretrained ViT; веса LR-пробы и LP не переносятся в final.
Seed задания используется для head/LoRA, Trainer, sampler и аугментаций.

Файлы действующего LR-пайплайна не изменены: старые runs.json, admission и
готовые результаты совместимы. Новый final release дополнительно закрепляет
final_protocol.py, final_results.py, final_train.py и final_queue.py. После
создания final-плана эти файлы также нельзя менять в рабочей копии исполнителя.
Изменения документации/тестов не меняют вычислительный hash.

Команды выполняются из корня проекта в прежнем venv. Сохранить те же HF_HOME и
другие настройки cache, что при LR. Используется успешный **собственный**
admission исполнителя от исходного LR-плана: неизменные Trainer/profile/стек
позволяют переиспользовать GPU-приёмку. Окружение и фактическая GPU проверяются
перед очередью и каждым worker. Если окружение изменилось, сначала повторить
исходный `run_queue.py --preflight` с тем же runs.json.

## 2. Создать один final-план

Собрать у координатора все три LR-каталога. При необходимости сначала получить
selected_lrs.json обычным `run_queue.py --collect` из инструкции 07.

```bash
python final_queue.py --make-plan --lr-plan runs.json --protocol artifacts/protocol --results results/rtx4070ti results/rtx3090 results/rtx3060 --selection selected_lrs.json --plan final_runs.json
```

`--selection` необязателен: LR в любом случае **заново выбираются по проверенным
исходным артефактам**. Переданный selection сверяется с ними. Недостающие,
повреждённые или повторные пробы, чужой план, все три numerical_invalid одного
метода и неверный выбор LR блокируют создание final-плана.

Создаются final_runs.json и artifacts/protocol/freeze.json. Они включают
ссылку на LR-план, все исходы поиска, hashes LR-артефактов, выбранные LR,
вычислительный release и назначения. Абсолютные пути результатов не участвуют
в идентичности: папки можно передавать между ПК. Для другого расположения freeze
есть `--freeze ПУТЬ`, который затем указывается во всех final-командах.
Существующие plan/freeze не перезаписываются. Если потребуется новый дизайн,
использовать новые имена и сохранить предыдущий.

Передать всем исполнителям **одни исходники, final_runs.json и freeze.json**,
а также прежний data protocol. В старом data-protocol.zip нового freeze нет:
его нужно скопировать отдельно. Исходные LR-артефакты нужны координатору при
создании плана; для обучения на остальных ПК достаточно общего final-плана,
freeze, данных и собственного admission.

## 3. Проверить назначения

```bash
python final_queue.py --dry-run --plan final_runs.json --protocol artifacts/protocol --executor rtx3060
```

| Исполнитель | Seed | Порядок методов |
| --- | ---: | --- |
| rtx4070ti | 42 | LP → A-iso → M-iso → B-iso |
| rtx3090 | 43 | A-iso → M-iso → B-iso → LP |
| rtx3060 | 44 | M-iso → B-iso → LP → A-iso |

Все три seed метода используют один выбранный LR. Произвольные LR/seed/epochs,
probe, ранняя остановка и resume в final CLI отсутствуют. Batch и precision
берутся из исходного профиля поиска, под текущий объём VRAM не подстраиваются.

## 4. Запустить очередь

Подставить свой executor и путь к собственному admission:

```bash
python final_queue.py --run --plan final_runs.json --protocol artifacts/protocol --executor rtx3060 --output results/final --admission ПУТЬ_ИЗ_LR_PREFLIGHT/admission.json
```

Одна задача — отдельный процесс. Каталоги:
`results/final/<executor>/runs/<run_id>/attempts/<attempt_id>/`.
В корне исполнителя queue.lock; логи в logs/. При повторном запуске TRAINED
полностью проверяются и пропускаются. Повреждённый TRAINED не заменяется новой
попыткой. После технического сбоя сначала диагностика, затем та же команда с
`--retry-failed`: новый attempt и обучение с начала, прежние файлы сохраняются.
После аварийного выключения stale queue.lock удаляется вручную только после
проверки, что прежняя очередь/worker больше не работают.

Numerical_invalid сохраняется с причиной и TRAINING_FAILED, **останавливает**
final-очередь и блокирует полноту серии. `--retry-failed` не повторяет такой run
и не подменяет LR/seed. Низкая accuracy корректного пятиэпохового run допускается
и не является основанием повторить попытку.

## 5. Результат обучения

| Артефакт | Содержимое |
| --- | --- |
| config.json | final-план/hash freeze, метод, LR, seed и общий профиль |
| environment.json | фактические GPU/стек/ОС |
| history.json / updates.json | история, пять val-оценок, applied/skipped updates и LR |
| metrics.json | best val/count/epoch, 180 attempted updates, время и GPU peaks; test-поля null |
| best/ | LP head либо PEFT adapter+classifier, metadata |
| checkpoints/ | штатные epoch checkpoints, сохраняется более ранний best |
| training_status.json | неизменяемый итог обучения: trained либо numerical_invalid |
| training_checksums.json | hashes config/environment/history/updates/metrics/training_status и best |
| TRAINED / TRAINING_FAILED | маркер, записанный последним после проверки комплекта |
| status.json | текущий progress; будущая evaluation сможет менять его отдельно |

TRAINED подтверждает **обучение**, а не оценку на test. Статус progress не входит
в training checksums, поэтому будущий сбой evaluation не меняет подтверждённые
веса и метрики обучения. Sealed training-файлы нельзя дописывать или исправлять;
будущие test-артефакты должны сохраняться отдельно.

wall_train_job_seconds измеряется вокруг Trainer.train(): train+val+checkpoint
и загрузка best в конце. Export измеряется отдельно в export_seconds.
train_loop_residual_seconds = job − validation − checkpoint, включая чтение
данных и логирование. GPU peaks относятся к training job с validation;
setup/download и будущий test в эти временные интервалы не входят.

## 6. Собрать обучения

Передать координатору каталоги исполнителей, включая best, TRAINED и checksums:

```bash
python final_queue.py --collect --plan final_runs.json --protocol artifacts/protocol --results results/final/rtx4070ti results/final/rtx3090 results/final/rtx3060 --output trained_finals.json
```

Сбор требует ровно 12 подтверждённых trained runs, проверяет полные validation
histories, выбор best, update audit, отсутствие test-метрик и hashes. Дубликаты,
чужие/повреждённые результаты и numerical_invalid блокируют полную сводку.
trained_finals.json имеет test_evaluated=false. Наличие freeze фиксирует дизайн,
но готового evaluate/test loader на этом этапе ещё нет.
