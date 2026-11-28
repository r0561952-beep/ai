# Ролевой адаптер LoRA для Qwen2.5-1.5B

Базовая модель: [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) (Apache-2.0), скачивается автоматически.
Адаптер: LoRA r=16, alpha=32, все линейные слои, fp16. Обучен `qwen_train.py` на 5000 русских ролевых сценах
(`prepare_rp.py`), формат ChatML, карточка персонажа — в system.

```python
from qwen_chat import QwenBot
bot = QwenBot("model/qwen_rp_adapter")
```

Сексуальные сцены с несовершеннолетними: данные очищены (`safety.py`), в чате такие запросы блокируются.
