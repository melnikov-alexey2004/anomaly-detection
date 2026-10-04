import dataclasses
import peft


@dataclasses.dataclass
class Config:
    train_batch_size = 64
    eval_batch_size = 64
