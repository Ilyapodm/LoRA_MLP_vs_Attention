# Принятая техническая архитектура

Редакция 25.11.2026. Технические решения зафиксированы по поручению пользователя. Реализация начата: bootstrap, сборка PEFT и synthetic smoke; Food-101/Trainer pipeline и обучение ещё не запускались. Научный протокол: 20 200 изображений, пять эпох, PEFT LoRA, LP/A/M/B, 12 LR + 12 final = 24 запуска. Источник истины по эксперименту — [02-protocol.md](02-protocol.md), ресурсы — [04-team-and-plan.md](04-team-and-plan.md).

Принятые правила ниже обязательны для первой реализации. Runtime-проверка ещё должна установить совместимые версии полного стека, вместимость batch, реальные скорости и пути/права хранилища. Это проверка работоспособности выбранной архитектуры, не повторный выбор методов, данных, LR-сетки или эпох.

## 1. PEFT и небольшой общий пайплайн

Адаптеры создаёт Hugging Face PEFT через LoraConfig/get_peft_model; обучение, AMP и resume ведёт Transformers Trainer. Собственную LoRA, собственный универсальный training loop, Docker, DDP, DeepSpeed, quantization и scheduler заданий не разрабатываем. Один процесс выполняет один run; после выхода GPU память освобождается. Код ведут пользователь и ассистент, участники выполняют назначенные задания.

| Будущий файл | Ответственность |
| --- | --- |
| configs.py | четыре конфигурации, полные имена Linear, ranks/alpha и формулы |
| data.py | общий manifest, dataset, transforms, sampler и collator |
| train.py | действия train/resume/evaluate/export, проверки, Trainer, артефакты и sync |
| colab.ipynb | установка закреплённого окружения, Drive/cache, вызов того же train.py |
| analysis.ipynb | валидация результатов, CSV, статистика и графики |
| test_params.py | критические проверки сборки, gradients, accumulation и resume |
| requirements.txt | точные проверенные версии библиотек |
| runs.json | очередь LR/final, назначения и идентичность конфигураций |

Stage (technical/lr_search/final) задаёт научное назначение; action (train/resume/evaluate/export) — действие над запуском. Colab и ПК отличаются только путями и hardware metadata. CLI не принимает произвольные изменения параметров согласованного протокола без нового protocol_id.

## 2. Окружение и версия кода

Целевая структура ViT — Transformers **4.57.1**, для которой проверены имена модулей в исходнике. Полную совместимость PEFT/torch/torchvision/Accelerate/Datasets/Pillow/NumPy и Python ещё проверяем коротким запуском в Colab и на самой маленькой участвующей локальной GPU. После успешной проверки фиксируем точные версии и Python major/minor; на ПК отдельное venv, в Colab штатное окружение с установкой закреплённых пакетов и перезапуском kernel при необходимости. Ошибка совместимости означает остановку bootstrap и согласованное исправление стека до серии, не автозамену пакетов на latest.

CUDA build должен поддерживаться установленным драйвером; одинаковые драйверы на разных машинах не требуются. Python/package/CUDA runtime/driver versions сохраняются в environment.json каждого attempt. Модель, processor и dataset закрепляются полными commit revisions, а не main.

Код распространяется одной неизменяемой release-версией: Git commit и архив исходников с SHA-256. До инициализации Git источником версии служит архив с хешем. На старте сверяем хеш исполняемых файлов; документационные изменения сами по себе не меняют computational code hash. Requirements и release manifest архивируются вместе с результатами.

Почему так: готовые PEFT/Trainer сокращают собственный код; закреплённая release-версия не даёт пяти исполнителям незаметно разойтись в реализации.

## 3. Сборка модели и head

Загружаем ViTForImageClassification на FP32 весах, num_labels=101 и единую таблицу label2id/id2label. Classifier создаём заново и явно инициализируем в отдельном RNG-контексте: truncated normal mean=0, std=config.initializer_range, стандартные bounds PyTorch a=-2,b=2, bias=0; эта процедура одинакова для всех методов. Сохраняем хеш head до PEFT и проверяем активную head после обёртки.

Для A/M/B используем generic PeftModel без текстовых task_type; modules_to_save=[classifier], bias=none, обычная инициализация LoRA init_lora_weights=True (B=0). Полные имена Linear получаем из закреплённой структуры, не общий suffix dense. Разные ranks/alpha Both задаём точными pattern keys и проверяем результат на каждом слое. Для LP PEFT не требуется: замороженный backbone и обучаемая head, тот же data/training pipeline.

Перед каждым run проверяем 12 блоков, формы Q/K/V/O и FC1/FC2, 101 выход classifier; число LoRA-модулей A=48, M=24, B=72, LP=0; ranks/scaling и trainable whitelist. Для A/M/B ровно 1 474 560 LoRA + 77 669 head = 1 552 229 trainable; LP 77 669. Активная head одна, исходный backbone совпадает с revision. Нулевое начальное обновление адаптера проверяется на детерминированном входе в eval mode.

Smoke test нескольких optimizer steps проверяет head/LoRA gradients и неизменность frozen weights. При B=0 первый gradient A может быть нулевым: B проверяем на первом шаге, A после следующих. В LoRA-методах не оборачиваем весь backbone в no_grad; при обучении model.train(), при оценке model.eval(). Признаки CLS для LP заранее не кешируем: у него те же меняющиеся train crops, что у LoRA.

Почему так: одинаковая head даёт общий старт, а проверка реальных модулей исключает адаптацию не тех матриц при формально верной сумме параметров.

## 4. Manifest и точные преобразования

Один ответственный создаёт manifest до очереди: классы обходятся в порядке исходных ID, внутри класса исходные row indices сортируются, один NumPy Generator(PCG64), seed=42, выдаёт перестановки; первые 200 выбираются, первые 180 из них train, следующие 20 val. Сохраняем результат и версию алгоритма; последующие сессии читают manifest, не генерируют его заново.

ID изображения = dataset revision + source split + row index. Manifest содержит ID/label, SHA-256 исходных image bytes, ordered train/val lists, counts, label mapping, свой hash и dataset revision. Проверяем декодирование всех выбранных train/val, ровно 180/20 на каждый из 101 класса и отсутствие пересечений. Испорченные изображения не пропускаются тихо: ошибка подготовки требует общего исправления manifest до серии. Test IDs идут в исходном порядке официального validation фиксированной revision; ID/labels/hash его списка проверяются при итоговой оценке.

| Этап | Фиксированная процедура |
| --- | --- |
| Train | RGB → RandomResizedCrop 224, scale=(0.08,1.0), ratio=(3/4,4/3), BILINEAR, antialias=True → HorizontalFlip p=0.5 → ToTensor FP32 → Normalize |
| Val/test | RGB → Resize короткой стороны до 224, BILINEAR, antialias=True → CenterCrop 224 → ToTensor FP32 → Normalize |
| Normalize | mean/std из image processor той же model revision; значения сохранены в config, повторной нормализации нет |

Задержанная загрузка/декодирование изображений, не весь dataset float-тензоров в RAM. Label mapping единый, метки torch.long. Collator возвращает pixel_values и labels; model не получает image/ID/служебные поля. Для Trainer фиксируем remove_unused_columns=False и label_names=[labels].

Почему так: manifest фиксирует сами изображения, а полный рецепт transforms исключает различия из-за defaults разных библиотек.

## 5. Диски и постоянное хранилище

Исходный Parquet HF cache и рабочие checkpoints — на локальном диске runtime: /content в Colab, заданный local_work_dir на ПК. Каждая сессия переиспользует cache; каждый run отдельный процесс, без повторного скачивания при наличии cache. В первой версии нет streaming, отдельного архива поднабора или копии всего dataset через save_to_disk. Изображения на каждом batch с Drive не читаем.

Постоянное хранилище — **одна папка Google Drive владельца проекта**, с доступом участников на запись в собственные каталоги executors/<executor_id>. Корневой каталог/реестр ведёт пользователь как координатор; разные пользователи не пишут одни и те же файлы. В Colab Drive монтируется. На ПК checkpoints всегда пишутся в постоянный local_work_dir, а готовый комплект артефактов после каждого run исполнитель вручную загружает в свою Drive-папку; координатор/analysis проверяет SHA-256 после чтения из общего хранилища. Это выбранный первый вариант для любых ОС, без установки Drive-клиента/rclone. Для переноса незавершённого run исполнитель вручную передаёт последний подтверждённый epoch bundle, включая resume и best_model. В завершённом облачном комплекте resume bundles не обязательны: best export/history/metadata/метрики/predictions и checksums обязательны по стадии; локальные resume сохраняются до приёмки результата. До LR search каждый исполнитель делает пробную запись/чтение/проверку хеша. Подключение Drive и выдача прав выполняются владельцами аккаунтов, их доступность ещё не проверена.

Работающие на ПК обязаны иметь постоянный локальный checkpoint даже при задержке доставки в Drive; локальный диск ПК — подтверждённое постоянное хранилище, /content — нет. Для ПК незавершённая Drive-доставка не теряет checkpoint и не требует останавливать обучение, пока локальный файл подтверждён; финальная сборка требует подтверждённой доставки. Для Colab ошибка remote sync останавливает обучение по правилам раздела 9.

До загрузки проверяем свободные RAM/диск; ориентир свободного локального диска 20 GB, но допуск определяется фактическим cache + временными копиями + запасом под checkpoints. Измеряем размер LP и LoRA resume bundles, резервируем место для последних двух версий и best всех назначенных runs. Drive-квоту также проверяем фактически; цифра adapter size не включает большой LP resume и optimizer state.

Почему так: локальное чтение быстро и не зависит от мелких Drive I/O; постоянная копия переживает удаление Colab VM.

## 6. Случайности данных и воспроизводимость

Инициализация head, LoRA, sampler и model dropout используют разделённые управляемые RNG. Порядок train определяется seed и номером эпохи собственным sampler/generator, не случайностями создания адаптеров. Аугментации определяются устойчивым SHA-256 от seed/epoch/sample_id/version; локальный RNG-контекст восстанавливает внешние Python/NumPy/torch CPU states, не меняя поток model dropout. Встроенный Python hash не используем.

DataLoader: num_workers=0, pin_memory=True на CUDA, persistent_workers=False, без скрытой параллельной генерации transforms. Датасет/самплер получает фактический номер эпохи до первого batch, включая resume. Проверяем совпадение ordered sample IDs и параметров crops/flip для одного seed между методами, LR-пробами и после восстановления. Между seed данные manifest те же, но shuffle/augmentations и начальные head/LoRA различаются. Dropout masks между разными архитектурами совпадать не обязаны.

Полной побитовой воспроизводимости на разных GPU не обещаем. На одинаковом стеке/устройстве проверяем продолжение после границы эпохи против непрерывного запуска, с заранее записанными численными допусками, ID порядка и LR/global_step. Режим полной deterministic algorithms не включаем как дополнительное требование серии; фиксируем seeds/backend/math flags, cudnn.benchmark=False.

Почему так: методы получают одинаковые входы одного seed, а код остаётся небольшим и восстановление не зависит от скрытых worker RNG.

## 7. Batch, loss и вычислительный профиль

Профиль общий для всей серии и закрепляется в runtime_profile.json до LR search. Порядок проверки фиксирован: microbatch16/accumulation32; при OOM microbatch8/accumulation64; затем microbatch4/accumulation128. Везде effective batch512. Если минимальный профиль не проходит, включаем non-reentrant gradient checkpointing для всех LoRA-методов, проверяем gradients и выбираем первый помещающийся профиль в том же порядке. LP использует тот же microbatch/accumulation; checkpointing ему не нужен при frozen backbone. Если и этот режим не помещается на устройстве, устройство не допускается к серии; автоматического изменения данных/эпох нет. Eval batch выбираем 32, при OOM 16, затем 8, единый для всей серии.

Precision: FP32 model weights и AdamW states, CUDA AMP FP16 и GradScaler; SDPA без output_attentions, TF32 для matmul/cuDNN выключен, float32 matmul precision highest. Для всех площадок те же math flags/backend; если SDPA не работает на закреплённом стеке, bootstrap останавливается до изменения общего профиля. torch.compile, FlashAttention-пакет, BF16 и quantization в первой версии выключены. Профиль проверяем минимум на Colab и на самой маленькой локальной GPU, которую реально будем использовать.

Epoch train содержит 35 полных accumulation-групп по 512 и одну из 260 изображений: 36 attempted updates, 180 за пять эпох. Последний microbatch16 или8 содержит четыре изображения. Сохраняем все примеры: loss accumulation нормируется по числу изображений группы, без drop_last, padding или дублирования. Сначала проверяем эквивалентность gradients/update большого batch и накопления на маленьком детерминированном примере без dropout/AMP, включая неполную группу. Используем Trainer; если его loss handling не проходит проверку, в том же train.py задаём минимальную sample-weighted loss function/Trainer adjustment, не новый training loop. Scheduler считает согласованные updates, не микробатчи; фактические attempted/applied/skipped и LR сохраняем.

Если после старта OOM вызван занятостью GPU/неподдержанным устройством, освобождаем ресурс либо переносим run на прошедшее проверку устройство с тем же профилем. Изменение microbatch/backend/precision только для одного метода запрещено; новый общий профиль оформляется новой версией и повторением зависимых runs.

Почему так: одинаковый effective batch сохраняет режим оптимизации, а проверка нормировки не даёт последнему короткому batch случайно получить другой вес.

## 8. Численная диагностика

Проверяем finite loss/logits и сохраняемых trainable weights. Для grad norm проверяем значения после unscale, clipping выполняется штатным AMP/Trainer-порядком. Логируем scaler и attempted/applied/skipped optimizer updates. Изолированный AMP skip допустим, scheduler не должен делать вид, что пропущенное optimizer update применилось: фактическое поведение стека проверяем и записываем.

**Десять consecutive skipped updates** — фиксированный инженерный порог остановки с technical_failed: это около заметной части короткого run, причина требует диагностики. Он не означает, что LR доказанно плохой; не назначаем numerical_invalid только по skips и не заменяем кандидат другим LR. Нефинитные forward loss/logits либо веса означают numerical_invalid по протоколу. При восстановившихся после отдельного skip обновлениях run продолжается. На итоговую оценку допускаем валидные полностью завершённые пять эпох с явным числом пропусков; scheduler может иметь меньше applied steps, чем attempted, это отражаем в history.

Никакого индивидуального спасения LR через другую precision. Исправление ошибки реализации или общий новый precision/profile документируется до повторения зависимых проб. После всех невалидных LR метода final блокируется.

Почему так: единичная настройка scaler не становится ложным провалом, но run, который перестал обновлять веса, не тратит часы незаметно.

## 9. Checkpoint, sync и resume

Штатный Trainer checkpoint создаётся каждую эпоху: model/adapter+head, optimizer, scheduler, scaler при наличии, RNG и trainer state. Для LP resume сохраняет полный ViT штатным способом; компактный LP export содержит только head. Best export LoRA — PEFT adapter+classifier, FP32 safetensors; LP head также FP32 safetensors. Base model/processor revisions входят в export metadata, оригинальные веса не дублируем в LoRA export.

Epoch bundle включает текущий resume, best_model в формате загрузки Trainer, компактный best export, best epoch/correct_count, history, config/data/code hashes и описание файлов с SHA-256. best_model для LP содержит полные веса ViT в штатном формате; файл одной head никогда не подставляем в Trainer.load_best_model как полный checkpoint. Для LoRA best_model использует PEFT-compatible формат. Optimizer state нужен у текущего resume, а не у отдельной best_model-копии. Best может быть из более ранней эпохи; его не теряем при ротации. Пути переносимые относительные; при копировании локально восстанавливаем ссылки Trainer best_model_checkpoint на существующий путь, затем проверяем reload. Resume заканчивается на суммарных пяти эпохах, не начинает новые пять.

Выгрузка: локальный bundle → новая immutable папка epoch/attempt на постоянном хранилище → сверка файлов/размеров/SHA-256 → READY последним. На Drive не полагаемся на атомарное переименование. Для Colab максимум три попытки sync с паузами 5 и 15 секунд между ними; после третьей неудачи epoch loop останавливается до следующей эпохи и выдаёт interrupted/storage_sync_failed с последним подтверждённым номером эпохи. Защищён только подтверждённый bundle, а не «успешная» локальная запись в /content. Сохраняем последние две подтверждённые версии и best, старые удаляем лишь после подтверждения новой. Для ПК локальный постоянный bundle подтверждается сразу, Drive-доставка отслеживается отдельно.

Перенос на другую GPU разрешён при тех же code/data/profile/package versions с записью оборудования по каждому segment; такой run не используется для сравнения времени/памяти с непрерывными. Accuracy остаётся результатом документированного процесса, а не обещанной побитовой репликой. Изменение профиля или вычислительного кода несовместимо с обычным resume и требует нового protocol/release решения.

Почему так: сохраняется не только результат модели, но и место в процессе обучения. Проверка remote copy защищает от выбора недокопированного checkpoint.

## 10. Идентичность runs и действия над результатом

Run ID = hash canonical JSON: protocol_id, stage, config, seed, нормализованный числовой LR, data hash и runtime_profile hash. 1e-3 и 0.001 дают одну запись; абсолютные пути/GPU/исполнитель в hash не входят. Вычислительный release закреплён protocol manifest, actual code hash есть у каждой попытки. Новый существенный код/profile/data требует новой версии; обычный resume допускается только с тем же computational code hash. Изменение только документации не меняет его.

Попытки хранятся отдельно. Один координатор назначает run ID, один исполнитель имеет активный attempt; скрипт отказывается перезаписывать verified completed outputs и незавершённую папку другого attempt. Слияние результатов делает только analysis. Никакой общей базы или конкурентного append CSV.

Final проходит pending → training → trained → evaluating → completed. Ошибки: interrupted, technical_failed, numerical_invalid с причиной. LR search завершается статусом valid или numerical_invalid; прерванная проба ещё не даёт результата. Action evaluate загружает best без optimizer и обучения, проверяет hashes и дописывает недостающие predictions/norms/metrics; export также не меняет веса. Переоценка того же checkpoint не новый seed и не дополнительная попытка подбора.

Test loader и метрики доступны только stage=final + protocol/freeze.json, где записаны hashes выбранных LR всех четырёх методов и окончательной очереди. Public test bytes могут оказаться в HF cache при скачивании: это не использование test. Technical и lr_search не создают test loader, не читают test labels для решений и не вычисляют test метрики.

Почему так: сбой оценки не заставляет переучивать модель; пять сессий не портят один файл и не выдают один run за разные результаты.

## 11. Распределение заданий

Реестр assignments.json ведёт пользователь как координатор: run ID, исполнитель, device/profile, attempt, статус и ссылка. Исполнители запускают готовые команды одной release-версии, не изменяют configs. Для final единица назначения — LP/A/M/B одного seed; по возможности все четыре идут на одном устройстве. Порядок: seed42 LP→A→M→B; seed43 A→M→B→LP; seed44 M→B→LP→A, чтобы один метод не был всегда первым.

Для LR search предпочтителен один тип GPU. Если недоступен, координатор распределяет блоки по одному LR для всех четырёх методов и ротирует устройства так, чтобы методы не закреплялись за одной картой. Доступность Colab не резервируем как гарантированную; после квот используем проверенные ПК. Передача готовых заданий происходит по границам runs; вынужденный mid-run перенос следует правилам resume. Отклонения от парных блоков сохраняются как hardware limitation, сравнение accuracy не объявляется побитовым экспериментом.

Почему так: распределение ускоряет вычисления, не связывая результат метода постоянно с конкретной картой. Простого ручного реестра достаточно для 24 конфигураций.

## 12. Артефакты и схема метрик

```text
experiment/
  protocol/                 # manifest/labels/revisions/release/profile/runs/assignments/freeze
  executors/<executor_id>/
    runs/<run_id>/attempts/<attempt_id>/
      config.json
      environment.json      # hardware/package/code hashes по сегментам
      status.json           # schema version, stage/action/status/reason
      history.json          # epoch/step/loss/LR/scaler/applied/skipped
      metrics.json          # correct_count/sample_count/accuracy/best epoch/timings/peaks
      predictions.npz       # final: sample_id/true_label/pred_label + manifest hash
      norms.json            # только final LoRA, best checkpoint
      best/                 # компактный FP32 export, safetensors
      checkpoints/          # последние два epoch bundles: resume + Trainer best_model + metadata
      checksums.json
      COMPLETE              # проверен полный комплект артефактов stage
  results.csv               # производная сводка final
  lr_trials.csv             # все 12 конфигураций поиска и их статусы
```

schema_version=1; stage/action/status/reason обязательны. Для каждой оценки храним correct_count/sample_count и accuracy fraction без округления, для presentation переводим в проценты. Best checkpoint определяется точным correct_count; tie-break — ранняя эпоха, затем меньший LR при выборе между пробами. MB=10^6 bytes, GiB=2^30 bytes. Размер export LoRA включает classifier, LP содержит head; resume sizes отдельно.

Train/val fields обязательны для valid search и trained/final; test fields только для completed final, в остальных стадиях null, не 0. Не завершённые/численно невалидные attempts сохраняют доступную history и причину, без вымышленных итоговых метрик. Валидные низкие результаты сохраняем. COMPLETE для valid LR не требует predictions/norms; COMPLETE для final требует best+history+metadata+test predictions+metrics и norms для LoRA. Условия отбора в results.csv учитывают stage/status, не только marker.

Почему так: отсутствие test не превращается в ложный нулевой результат, а CSV можно заново построить из независимых артефактов.

## 13. Замеры и итоговая оценка

Основной memory показатель — peak allocated/reserved **всего training job с периодической val**; подписываем train+val, не выдаём за чистый backward peak. Trainer memory tracking отключён (skip_memory_metrics=True), наши peak stats сбрасываются один раз после подготовки модели до train, не во вложенном evaluate. Для test отдельный свежий процесс без optimizer и отдельные peaks. Это проще и однозначнее сложного вложенного измерения train/val по отдельности.

Время: фиксируем wall_train_job (включает val/checkpoint/sync), отдельно суммарные val/save/sync интервалы; train_loop_residual = job минус эти интервалы, с явно указанным включением чтения данных/логирования. Интервалы CUDA synchronize на границах; timing overhead одинаков для методов. Setup/download и test отдельно. Сравнение стоимости только при одинаковых GPU/math/batch/workers, условиях CPU/диска и warm-cache; иначе таблица фактических затрат с equipment, без строгого рейтинга методов. One process per run исключает пики от прежних моделей.

Test: model.eval()+torch.inference_mode(), общий AMP FP16, детерминированные transforms, фиксированный порядок, без shuffle. Argmax и true labels сразу переносятся на CPU; все logits/активации и изображения на GPU не накапливаем. NPZ содержит ID/labels/predictions, shapes/dtypes проверяются, checksum сохраняется. Повторный evaluate не затирает подтверждённый результат без отдельной явной операции. Norms из best считаем в FP32 по одной матрице под no_grad, classifier исключён. Статистика — [05-analysis.md](05-analysis.md).

Почему так: скорость не смешивается с Drive-копированием, память имеет однозначные границы, а для McNemar/bootstrap достаточно компактных предсказаний.

## 14. Приёмка до массовых запусков

Обязательные проверки перед LR search:

1. Manifest/revisions/labels и детерминированные val/transforms; чтение всех выбранных train/val.
2. Сборка LP/A/M/B: head hash, 48/24/72 модуля, ranks/scaling, whitelist, несколько optimizer steps и frozen weights.
3. Sample-weighted accumulation для полной и неполной группы, фактический scheduler/applied steps, диагностика AMP skips.
4. Короткий Both на Colab и самой маленькой участвующей локальной GPU: профиль помещается, измерены train/inference скорости на train/val, бюджет пересчитан.
5. Save→остановка→resume на границе условной эпохи; best reload, переносимые пути и правильный номер эпохи. Проверка на другой машине перед её допуском к resume.
6. Sync partial copy/retry/READY, отказ Drive и сохранение предыдущей версии; достаточный локальный/постоянный диск.
7. Evaluate/export из best без обучения на validation, schema/status/null fields, NPZ ID/order, отсутствие дублей при сборке CSV.
8. Контроль test gate: technical/lr_search не могут создать test loader, final без freeze заблокирован.

Это короткие инженерные проверки на поднаборах, не дополнительные полноценные обучения и не подбор эпох. После приёмки 24 запланированных runs запускаются без ручного изменения конфигураций. Размер GPU, права Drive, полный version lock, model/dataset SHA и фактическое время — остаются фактами для измерения/получения в bootstrap, а не невыбранными вариантами архитектуры.

Основания: [PEFT LoRA](https://huggingface.co/docs/peft/main/package_reference/lora), [ViT 4.57.1](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/vit/modeling_vit.py), [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer), [PEFT checkpoint](https://huggingface.co/docs/peft/main/developer_guides/checkpoint), [PyTorch reproducibility](https://docs.pytorch.org/docs/2.15/notes/randomness.html), [Colab FAQ](https://research.google.com/colaboratory/faq.html?hl=en). Конкретный API сверяем с выбранными закреплёнными версиями; main-документация не заменяет requirements lock.

## Состояние реализации: первый технический этап

Добавлены configs.py (PEFT LP/A/M/B), bootstrap.py, test_params.py (synthetic smoke), colab.ipynb, package_bootstrap.py, README и лёгкие проверки tests/test_configs.py. Текущий requirements.txt — закреплённый bootstrap-кандидат, не принятый полный GPU-lock.

Пройдены только явно выполненные проверки: арифметика/выбор модулей без ML-зависимостей, синтаксис Python/notebook и SHA-256 архива исходников. Фактические CPU/Colab PEFT forward/backward отмечаются после запуска, не по наличию кода. Создание manifest, train.py/Trainer, нормировка accumulation, полноценный save/resume/sync, schema/freeze и статистика остаются последующими этапами; массовые LR runs пока запрещены.
