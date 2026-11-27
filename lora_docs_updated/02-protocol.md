# Научный протокол food101-local-v2

Зафиксирован 27.11.2026 до итогового test. Активная серия: 10 эпох, шесть уникальных
конфигураций, 24 LR + 18 final = 42 новых обучения. Исторический пятиэпоховый v1
сохранён в [архиве](archive/v1-02-protocol.md); его результаты являются пилотными.
Команды — [09-v2-runs.md](09-v2-runs.md), основания — [10-v2-design.md](10-v2-design.md).

## Исследовательские вопросы

Основной: как размещение LoRA в Attention, MLP и обеих частях ViT влияет на
качество адаптации к Food-101 при одинаковом количестве обучаемых параметров,
общем максимальном бюджете обучения и одинаковой процедуре настройки?

Дополнительный: как меняется соотношение качества и затрат при одинаковом
ранге r16 каждой адаптируемой матрицы в этих трёх placements?

Дополнительно описываем скорость адаптации по val-кривым/updates, best epochs,
устойчивость по seed и фактические time/VRAM/export size. Равные параметры и
равный ранг не могут одновременно выполняться для наших трёх наборов targets.
Каждая серия отвечает на свой вопрос; rank16 Both имеет больший бюджет.

## Модель и параметры

Модель google/vit-base-patch16-224-in21k, revision
b4569560a39a0f1af58e3ddaf17facf20ab919b0. Processor той же revision.
12 блоков, hidden768, intermediate3072, heads12, patch16, image224, CLS classifier768→101.

Attention: Q/K/V/O во всех 12 блоках, 48 матриц768→768.
MLP: FC1/FC2 во всех 12 блоках, 24 матрицы768→3072/3072→768.
Both: все 72 матрицы. Точные имена/формы проверяются до запуска.

N_trainable = 77 669 + 12 × (6 144 × r_attn + 7 680 × r_mlp).

| ID | r Attention / MLP | LoRA params | С head | Серия |
| --- | --- | ---: | ---: | --- |
| LP | 0 / 0 | 0 | 77 669 | общий baseline |
| A-iso | 20 / 0 | 1 474 560 | 1 552 229 | iso |
| M-iso = M-16 | 0 / 16 | 1 474 560 | 1 552 229 | iso и rank16 |
| B-iso | 10 / 8 | 1 474 560 | 1 552 229 | iso |
| A-16 | 16 / 0 | 1 179 648 | 1 257 317 | rank16 |
| B-16 | 16 / 16 | 2 654 208 | 2 731 877 | rank16 |

M-16 — alias M-iso, не седьмая конфигурация и не независимый повтор.
B-iso распределяет LoRA-бюджет 50/50; другие распределения не исследуются.
Frozen: pretrained backbone, bias, LayerNorm, embeddings/CLS. Trainable: LoRA A/B
и одна новая classifier head с bias. LP обучает только head, без feature caching.
Head инициализируется одинаково внутри seed до PEFT: truncated normal mean0,
std=config.initializer_range, bounds[-2,2], bias0, отдельный RNG и проверка hash.
LoRA — штатный PEFT, B=0, bias=none, dropout0.1, modules_to_save=[classifier].
Alpha каждой матрицы равна её rank: scaling=alpha/r=1. rsLoRA/DoRA/quantization
не используются. Scaling1 не гарантирует одинаковых эффективных updates.

## Данные и случайности

ethz/food101, revision83488de741c1bd1ce27aa6a2b33e19c7bdf92ca9.
Прежний общий manifest hash8337fa066463bd272d5609998187f1f8cc730f3fa23c0eabdb7c555449e9c0cf.
Из официального train отобраны 200 на класс: 180 train / 20 val, всего18180/2020.
PCG64 seed42 и сохранённые ordered row IDs/labels/image SHA; разбиение не создаётся заново.
Остаток train не используется. Официальный validation25250 (250 на класс) — test.

Train: RGB → RandomResizedCrop224 scale(0.08,1) ratio(3/4,4/3), BILINEAR/antialias
→ HorizontalFlip0.5 → ToTensor FP32 → Normalize mean/std processor.
Val/test: RGB → Resize короткой стороны224 BILINEAR/antialias → CenterCrop224
→ ToTensor FP32 → та же Normalize, без повторной нормализации.

Инициализация head/adapters и данные имеют управляемые RNG. Порядок train и
аугментации зависят от experiment seed/epoch/sample ID через SHA256. Для одного
seed все методы получают одинаковые shuffle/crops/flip; одинаковые dropout masks
и побитовость разных GPU не обещаются. Workers0. Проверки изображений/labels/
пересечений/детерминированности сохраняются из v1.

## Общий бюджет обучения

10 эпох без ранней остановки. Effective batch512, группы35×512+260 на эпоху:
36 attempted optimizer updates/эпоху, 360/run. Ни padding, ни drop_last;
sample-weighted cross entropy по числу изображений accumulation-группы.

Microbatch16/accumulation32/eval32. Общий резерв при OOM до серии: 8/64, затем4/128;
eval32→16→8. При необходимости общий non-reentrant checkpointing для LoRA,
приёмка всех конфигураций и новый план до обучения. Не менять профиль отдельного метода/GPU.
FP32 weights/states, AMP FP16, SDPA, TF32 off, highest matmul precision,
cudnn.benchmark=False, workers0. Один процесс/одна GPU на run.

AdamW обычный adamw_torch, betas(0.9,0.999), eps1e-8, weight decay0.
Linear decay до0 за360 updates, warmup0, clipping1, smoothing0.
Внутри run общий LR head/LoRA. Фактические applied/skipped и scaler сохраняются.
Изолированный AMP skip допустим; scheduler следует applied updates. Десять
consecutive skips — technical_failed. Нефинитные forward loss/logits/weights — numerical_invalid.

Val2020 после каждой эпохи; best — максимум целого correct_count, tie ранняя эпоха.
Все десять эпох реально выполняются; сохранение раннего best защищает результат
от более позднего переобучения. Best epoch известна ретроспективно; стоимость
run не сокращается до неё. Десять — ограниченный общий бюджет, не гарантия сходимости.

## LR и final

Каждая из шести конфигураций × {1e-4,1e-3,1e-2,3e-2}, seed142, 10 эпох = 24 LR-пробы.
Все с исходного pretrained ViT/новой head/LoRA; test loader не создаётся.
3e-2 добавлен до новой серии, поскольку пилот выбрал верхнюю границу1e-2.
Сетка не расширяется автоматически. Низкая accuracy валидна; numerical_invalid
расходует кандидата и исключается из выбора. Все четыре невалидны — остановка
и диагностика, без произвольного fallback.

LR победителя — максимальный best val correct_count, tie меньший LR. Один LR
конфигурации фиксируется на final seeds42/43/44: 6×3=18 новых обучений.
Final не продолжает LR/LP weights. Best каждого final выбирается по val тем же правилом.

Технический сбой/OOM/ошибка кода не доказывают плохой LR: после диагностики
тот же task с начала в новой attempt, исходные файлы сохраняются. Не повторять
из-за качества. Numerical_invalid final блокирует полноту серии, не заменяется
другим LR/seed. Изменение scientific recipe/release требует нового протокола/плана.

Старый v1 (5 эпох/180 updates/3 LR/4 метода) — пилот. Его LR/checkpoints не
смешиваются с v2; linear schedule на10 эпох иной с первого шага.

## Идентичность, test и отчёт

study_protocol закрепляет code files/hashes, scientific recipe, модель/данные,
24 LR/18 final, назначения и общую M-iso. Поиск и final используют одну release.
Все 24 LR-исхода проверяются до final_runs/freeze. Freeze включает обе серии,
шесть выбранных LR и18 моделей. Test закрыт до freeze; по test решения не меняются.
Публичные test bytes в cache не являются использованием test labels/metrics.

Каждый result имеет immutable config/environment/history/updates/metrics/best
и checksums, marker последним. LR COMPLETE не означает final completion.
Final TRAINED подтверждает только обучение: test-поля null. Будущие evaluation
артефакты отдельны; training_status/metrics/best не переписываются.

Главная метрика — test top1 mean±sample std по трём seed. Парные эффекты и
все индивидуальные значения обязательны. Точная схема 18 McNemar/Holm, bootstrap,
норм и ограничений — [05-analysis.md](05-analysis.md). Стоимость только внутри
одинаковых GPU/profile/CPU/disk/cache; иначе фактическая hardware-таблица.

REF/QV-baseline, full fine-tuning, rank sweep, другие datasets/backbones и поиск
распределения бюджета Both не входят в эту серию. Их отсутствие ограничивает
выводы. Никакого ожидания accuracy0.96 из другого tutorial split.
