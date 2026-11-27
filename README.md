# LoRA в Attention, MLP и Both: активный протокол v2

Сравниваем адаптацию google/vit-base-patch16-224-in21k к Food-101 в двух сериях:
равное число trainable parameters и фиксированный ранг r16. Данные прежние:
18 180 train / 2 020 val, официальный validation25250 служит закрытым test.

**V2: 10 эпох, шесть уникальных конфигураций, LR {1e-4,1e-3,1e-2,3e-2}.**
24 LR-пробы seed142 + 18 final (seeds42/43/44) = **42 новых обучения**.
M-16 идентичен M-iso и используется в обеих сериях без дублирования.

| Метод | r Attention / MLP | Trainable с head | Серия |
| --- | --- | ---: | --- |
| LP | 0 / 0 | 77 669 | общий baseline |
| A-iso | 20 / 0 | 1 552 229 | равные параметры |
| M-iso = M-16 | 0 / 16 | 1 552 229 | обе серии |
| B-iso | 10 / 8 | 1 552 229 | равные параметры |
| A-16 | 16 / 0 | 1 257 317 | равный ранг |
| B-16 | 16 / 16 | 2 731 877 | равный ранг |

Источники истины: [протокол](lora_docs_updated/02-protocol.md),
[основания дизайна](lora_docs_updated/10-v2-design.md),
[пошаговый запуск v2](lora_docs_updated/09-v2-runs.md),
[план анализа](lora_docs_updated/05-analysis.md).

## Быстрый порядок

Python3.12, requirements.txt с прежним CUDA PyTorch2.8/cu126, одна видимая GPU.
Новых ML-зависимостей нет. Прежний data protocol в artifacts/protocol и тот же
HF cache. На каждом ПК использовать одну согласованную source release и общий
runs-v2.json. Рабочий каталог команд — корень проекта.

Если runs-v2.json не предоставлен, координатор создаёт его **один раз**:

```bash
python study_queue.py --make-lr-plan --plan runs-v2.json --protocol artifacts/protocol
```

План не перезаписывается. Все исполнители получают один и тот же файл.
На каждом ПК подставить свой executor (rtx4070ti / rtx3090 / rtx3060):

```bash
python study_queue.py --dry-run --plan runs-v2.json --executor rtx3060
python study_queue.py --preflight --plan runs-v2.json --executor rtx3060
python study_queue.py --run --plan runs-v2.json --executor rtx3060 --admission ПУТЬ_ИЗ_V2_PREFLIGHT/admission.json
```

План содержит по восемь LR-задач на GPU. Новый preflight проверяет все шесть
моделей и pretrained B-16 с реальным профилем. Старый v1 admission не допускается.
Результаты по умолчанию в results/v2/lr/<executor>. Полные команды сбора 24
проб, final-плана/freeze и очередей 18 final — в [инструкции](lora_docs_updated/09-v2-runs.md).

Один run целиком на одной карте, отдельный процесс. После технического сбоя
повтор с начала того же задания в новой attempt с --retry-failed. LR numerical_invalid
расходует кандидата и позволяет продолжить поиск; final numerical_invalid блокирует
серию без подмены LR/seed. Готовые результаты проверяются и пропускаются.

## Реализовано

- study_configs.py — шесть моделей, PEFT-сборка, targets/ranks/freeze/head, две серии и18 гипотез.
- study_runtime.py — общий профиль, data pins, CUDA/package admission и локальный lock.
- study_trainer.py — общий Trainer, sample-weighted loss, sampler, timings и AMP audit.
- file_utils.py — hashes, эксклюзивные JSON и компактный best export.
- study_protocol.py — release/data/profile hashes, 24 LR / 18 final, selection/freeze.
- study_train.py — full-data LR/final, 10 val-эпох, 360 updates, best и диагностика.
- study_results.py — строгая приёмка и сбор; LR COMPLETE / final TRAINED.
- study_queue.py — coordinator CLI, preflight, очереди и retries.
- study_check.py — параметры/gradients/freeze всех шести, реальный B-16 preflight.
- package_study.py — согласованный source-only архив с планом и hashes.

Общий профиль: 16/32/eval32, effective512, FP32+AMP FP16, SDPA, TF32 off,
workers0. Best — максимум целого val correct_count, ties ранняя эпоха; LR ties
меньшее значение. Все final начинаются с pretrained; LP/LR weights не переносятся.

**Test evaluation, predictions, norms, численный статистический анализ и final
COMPLETE ещё не реализованы.** На этом этапе final заканчивается TRAINED с
нулевым доступом к test loader и test-полями null. Дизайн анализа зафиксирован
в плане; замена статистической семьи не означает, что p-values уже вычислены.

## Проверки

```bash
python -m unittest discover -s tests -v
```

CPU integration использует маленькие модели при полных counts18180/2020 и десяти
эпохах: 360 updates, ранний best, LP/PEFT export, numerical_invalid/interruption.
Реальные ViT A-16/B-16 также проверяются на CPU: shapes/ranks/alpha/gradients/freeze.
Это не научная accuracy и не подтверждение памяти/скорости новой GPU-конфигурации;
на каждой карте перед v2 нужен реальный preflight.

## История и локальный сценарий

Рабочее дерево содержит только актуальную серию v2. Старые пятиэпоховые CLI,
конфигурации, Colab notebook/bootstrap, переносимые bundles/remote sync и их
тесты удалены. Полный снимок до чистки сохранён коммитом `1c074e7`.
При необходимости исследовать исторический код/планы использовать Git.

Результаты пилота в results/rtx*, dataset/cache/venv и локальные доказательства
не удалялись. Их metadata/checksums не переписываются. Текущие команды работают
с results/v2 и не принимают старые schema1 результаты.

После изменения вычислительного release нужен новый общий runs-v2.json и новый
preflight каждой GPU. Не смешивать уже выданный архив/план до чистки с текущими
исходниками. Во время научной серии использовать фиксированный checkout/архив.
