# LoRA: Attention vs MLP vs Both

Основной сценарий — три постоянных ПК: RTX 4070 Ti, RTX 3090, RTX 3060.
Один run целиком на одном ПК, checkpoints на локальном диске. После технического
сбоя повторяем тот же run с начала в новой attempt-папке. Готовые результаты
передаём координатору после завершения. Colab и перенос незавершённого обучения
не являются условиями запуска основной серии.

Научный протокол: google/vit-base-patch16-224-in21k, Food-101, все 101 класс,
18 180 train / 2 020 val (180/20 каждого класса), пять эпох. LP, A-iso r20,
M-iso r16, B-iso r10/r8; штатный PEFT, LoRA-методы по 1 552 229 trainable params,
LP — 77 669. Поиск: каждый метод × LR {1e-4,1e-3,1e-2}, seed142, 12 runs.
Затем планируются 12 final (seeds42/43/44) и официальный test25250.

Реализованы `technical` и **`lr_search`**, локальная очередь, preflight и сбор LR.
**Final/evaluate/test gate и итоговая статистика ещё не реализованы.**

Полная инструкция и таблица распределения: [07-local-runs.md](lora_docs_updated/07-local-runs.md).

## Быстрый порядок

Python **3.12**, отдельный venv на каждом ПК, один видимый GPU. Сначала установить
CUDA PyTorch из cu126 индекса, затем requirements. Общий data-protocol.zip
распаковать в artifacts/protocol. Изображения скачиваются в локальный HF cache.

```bash
python -m pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

Preflight и очередь должны использовать один HF cache. Для новой установки
можно выбрать `cache/huggingface` внутри проекта; настройка HF_HOME для
bash/CMD/PowerShell дана в [инструкции](lora_docs_updated/07-local-runs.md#1-одинаковое-окружение-и-данные).
Если preflight уже скачал данные в стандартный cache без HF_HOME, продолжайте
с теми же настройками. Смена пути приводит к повторному скачиванию. Переменные
cache нужно восстанавливать в каждом новом терминале перед запуском.

Готовый общий runs.json включён в репозиторий. Посмотреть свои задания:

```bash
python run_queue.py --dry-run --plan runs.json --protocol artifacts/protocol --executor rtx4070ti
```

После обновления вычислительного кода до первого научного запуска план нужно
пересоздать. Все ПК получают **одни и те же исходники и runs.json**; во время
серии использовать зафиксированный checkout/архив, не обновлять код в рабочем
каталоге исполнителя. На каждом ПК заменить executor своим именем:

```bash
python run_queue.py --preflight --plan runs.json --protocol artifacts/protocol --executor rtx4070ti --output results
python run_queue.py --run --plan runs.json --protocol artifacts/protocol --executor rtx4070ti --output results --admission ПУТЬ_ИЗ_PREFLIGHT/admission.json
```

Каждый ПК начинает очередь после своего успешного preflight; проверки остальных
могут идти параллельно. Каждая очередь последовательно выполняет
четыре LR-задачи, каждый run в отдельном процессе. Готовые COMPLETE проверяются и
пропускаются. Численно невалидный LR сохраняется, очередь продолжается;
техническая ошибка останавливает её. После диагностики `--retry-failed` начинает
новую попытку с теми же настройками. Логи/история старых попыток остаются.

```bash
python run_queue.py --collect --plan runs.json --protocol artifacts/protocol --results results/rtx4070ti results/rtx3090 results/rtx3060 --output selected_lrs.json
```

Сбор требует все 12 исходов, проверяет hashes и выбирает LR по val correct_count,
при равенстве меньший LR. Все три numerical_invalid одного метода блокируют выбор.
LR search не создаёт test loader, test-поля null. Старт final — отдельный следующий этап.

## Код

- configs.py — модель, ranks/alpha, trainable whitelist, head init, PEFT.
- data.py / check_data.py — manifest, image hashes, transforms/RNG/sampler, приёмка данных.
- train.py — Trainer, sample-weighted loss, technical и full-data lr_search, local best export.
- experiment.py — неизменяемый план, проверка результатов и выбор LR.
- run_queue.py — очереди ПК, допуск окружения/GPU, остановка/повтор и сбор результатов.
- checkpoints.py / check_resume.py / check_storage.py — дополнительные bundle/resume/sync проверки;
  обязательная переносимость продолжения не используется в локальной научной серии.
- package_protocol.py / package_bootstrap.py — общий protocol archive и source bundle.
- colab.ipynb — вспомогательные технические проверки, не основной запуск LR.

Научный lr_search не принимает probe/stop/dummy/resume/произвольные LR или seed:
задание берётся из проверенного плана, обучается полный manifest пять эпох.
Сохраняются config/environment/status/history/updates/metrics, best export,
checksums и COMPLETE последним. LP export — head, PEFT export — LoRA+classifier;
Trainer checkpoints для LP содержат полный ViT.

## Проверки и ограничения

```bash
python -m unittest discover -s tests -v
```

Без ML-зависимостей доступны stdlib-тесты плана, данных и файловой целостности;
ML/CUDA проверки пропускаются с явным сообщением. CPU integration test проходит
полный путь LR на маленькой модели с 18 180 train / 2 020 val × 5: 180 updates,
пять val-оценок, ранний best при равенстве, export/COMPLETE и numerical_invalid.
Это проверка логики, не измерение GPU и не научная accuracy.

Ранее в Colab T4 подтверждены LP/A/M/B gradients/freeze/params, pretrained Both,
реальные данные и короткий Trainer 772/32 × 2: 31,34 с, allocated/reserved
2,312/2,496 GiB. Эти числа не являются benchmark трёх ПК или полным бюджетом.
Восстановление LP после переноса совпало точно. У Both продолжение AMP отличалось
до 1,3e-5; переносимый resume не принят. Это не основание объявлять ошибочным
обычный полный run; локальная очередь начинает повтор с исходных весов.
26.11.2026 полный локальный preflight на RTX 3060 12GB прошёл: проверены все
20 200 изображений, параметры/градиенты LP/A/M/B и pretrained Both через Trainer
772/32 × 2 с общим профилем 16/32/eval32. Job — 20,43 с, allocated/reserved —
2,312/2,496 GiB, четыре applied updates без skips. Это технический замер,
не научное обучение и не benchmark полной очереди. Preflight 4070 Ti/3090
ещё ожидается; каждый ПК может начать LR search после своего успешного допуска.

Dataset SHA: 83488de741c1bd1ce27aa6a2b33e19c7bdf92ca9.
Model SHA: b4569560a39a0f1af58e3ddaf17facf20ab919b0.
Manifest hash: 8337fa066463bd272d5609998187f1f8cc730f3fa23c0eabdb7c555449e9c0cf.
Независимая CPU/Colab подготовка дала один hash; все выбранные изображения проверены.
Артефакты в artifacts/ и cache/ игнорируются Git; data-protocol.zip содержит
только manifest/revisions/checksums, не изображения. Научные runs пока не выполнялись.

Новый план до начала серии создаётся в новом файле: `python run_queue.py --make-plan --protocol artifacts/protocol --plan runs-new.json`; затем этот путь указывается во всех командах. Существующий план не перезаписывается.
