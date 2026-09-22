import dataclasses
import typing


@dataclasses.dataclass()
class Config:
    win_size: int
    step_size: int
    time_emb_len: int
    batch_size: int


def run(dataset: typing.Literal["bgl", "tbird", "spirit", "liberty"],
        config: Config):
    pass