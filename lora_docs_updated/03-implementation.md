# Локальная архитектура актуальной серии

Один run целиком на одном постоянном ПК, отдельный процесс, одна видимая CUDA GPU.
Три независимые очереди RTX4070Ti/3090/3060. Cache, checkpoints и результаты —
локальный диск. После технического сбоя повтор того же задания с начала в новой
attempt. Готовые исходные артефакты передаются координатору архивом.

Научный контракт — [02](02-protocol.md), команды — [09](09-v2-runs.md).
Прежние сценарии доступны в Git (`1c074e7` — снимок перед удалением).

## Модули

| Файл | Ответственность |
| --- | --- |
| study_configs.py | единственный реестр шести моделей, ranks/alpha, PEFT/ViT, whitelist/freeze/head и scientific recipe |
| study_runtime.py | профиль, immutable data pins, фактический GPU/package environment, admission и queue lock |
| data.py | manifest, transforms, isolated RNG, EpochSampler и collator; только train/val |
| study_trainer.py | штатный Trainer с sample-weighted loss, sampler, timings и applied/skipped audit |
| study_protocol.py | code/data/profile hashes, планы24/18, LR selection provenance и freeze |
| study_train.py | worker полного manifest,10 эпох, export и terminal status; test loader отсутствует |
| study_results.py | stage-aware checksum verification, выбор LR и сбор trained-моделей |
| study_queue.py | единый coordinator CLI и локальные очереди, процессы/retries/preflight |
| study_check.py | gradients/freeze всех шести конфигураций и технический pretrained B-16 |
| file_utils.py | SHA256, exclusive JSON writes, compact LP/PEFT best export |
| bootstrap.py / check_data.py | приёмка окружения/диска и всех selected images |
| package_protocol.py / package_study.py | общий data protocol и source-only release |

## Общие примитивы

Сборка PEFT одна для всех placements/ranks. До обёртки проверяются точные имена
и формы Linear, отдельно инициализируется head. После неё проверяются ranks,
alpha/r=1, active head и точный набор trainable IDs. Базовые веса заморожены;
совместно с адаптерами обучается classifier. M-16 — alias M-iso, не новая модель.

Trainer и Accelerate выполняют обучение/AdamW/AMP/scheduler/checkpoints.
compute_loss_func нормирует сумму CE по числу изображений всей accumulation-группы,
включая последнюю группу260. EpochSampler передаёт номер эпохи датасету;
shuffle/crops/flip совпадают между методами одного seed. LoRA dropout остаётся
случайностью модели, не перезаписывается RNG аугментаций.

Перед запуском worker проверяет release/plan/protocol/admission и assignment.
В scientific CLI нет произвольных LR/seed/epochs/subset/ранней остановки.
Технический subset существует только в study_check, не в scientific worker.

## Артефакты

LR создаёт config/environment/status/history/updates/metrics/best, checksums.json
и COMPLETE последним. Валидный result требует десять полных val-оценок,
360 attempted updates, согласованные applied/skipped/scheduler и ранний best при ties.

Final сохраняет TRAINED/training_checksums и immutable training_status.json.
status.json — отдельный progress для будущей evaluation. Metrics/best обучения
не переписываются test-данными. Численно невалидный final блокирует серию.
Предсказания test и итоговый COMPLETE требуют отдельного этапа evaluation,
который пока не реализован.

Штатные локальные Trainer checkpoints сохраняются для выбора и загрузки best,
но переносимое продолжение/remote sync не являются функциями проекта.
Компактный export — LP head или PEFT adapter+classifier в FP32 safetensors.
Частичные/повреждённые артефакты не принимаются и не чинятся пересчётом hashes.

## Время, память и версия

Training job измеряется вокруг Trainer.train(), включая val/checkpoint/best reload.
CUDA synchronize на границах измерения. Setup и export отдельно. GPU peaks
сбрасываются один раз до train и относятся к train+val, без вложенных reset.
Стоимость сравнивается внутри одинаковых GPU/profile/CPU/disk/cache.

Все вычислительные файлы перечислены в study_protocol.SOURCE_FILES. Изменение
реализации требует нового общего плана/архива до серии и повторного preflight.
Документация/тесты не меняют computational hash; source archive имеет отдельный
manifest всех распространяемых файлов. Передача metadata сохраняет исходные байты.
