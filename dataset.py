import torch
from torch.utils.data import Dataset, DataLoader, Sampler, IterableDataset
import os
import subprocess
import shutil
import warnings
import gzip
import typing
import datetime
import math


class Data:
    path_to_log_dir: str
    path_to_log: str
    cnt_ind: int

    def __init__(self, downloaded_url: str, cache_dir: str = os.path.expanduser('~/.dataset'),
                 dataset_dir: str = 'hdfs', extract_dir: str = 'archive_extracted',
                 repeat_download: bool = False,
                 custom_archive_path: typing.Optional[str] = None, remove_archive: bool = False,
                 archive_type=None, use_in_colab: bool = False, encoding: str = "latin-1",
                 errors: str = "replace", max_lines: float=math.inf,
                 window_size: int = 200,
                 step_size: int = 200,
                 ):
        # gzip_fp -> extract_dir/data_dir/data_dir.log
        # * -> extract_dir/data_dir/*

        assert window_size > 1
        assert step_size >= 1
        assert max_lines >= 0

        colab_content = "/content/"
        if use_in_colab:
            cache_dir = colab_content
            extract_dir = colab_content

        if os.path.exists(colab_content) and not use_in_colab:
            warnings.warn("use_in_colab = False но при этом существует /content/")

        self.max_lines = max_lines
        self.window_size = window_size
        self.step_size = step_size
        self.cache_dir = cache_dir
        self.dataset_dir = dataset_dir
        self.extract_dir = extract_dir
        self.custom_archive_path = custom_archive_path
        self.downloaded_url = downloaded_url
        self.use_in_colab = use_in_colab
        self.encoding = encoding
        self.errors = errors

        if not os.path.exists(cache_dir): raise FileExistsError
        if not repeat_download and os.path.exists(os.path.join(cache_dir, dataset_dir)):
            print('exist')
        else:
            if custom_archive_path is None:

                if os.path.exists(os.path.join(cache_dir, dataset_dir)):
                    warnings.warn("you use repeat download with downloaded archive, repeat_download=True")

                filepath = os.path.join(cache_dir, f'archive_{dataset_dir}')
                if os.path.exists(filepath) and not repeat_download:
                    print('already downloaded')
                else:
                    print('downloading...')
                    args = ['curl', '-o', filepath, '--retry-delay', '1', '--retry', str(int(100_000)),
                            '--retry-all-errors', '-L',
                            downloaded_url]
                    res = subprocess.run(args, capture_output=True, text=True)

                    print('stdout from curl', res.stdout)
                    print('stderr from curl', res.stderr)
                    print('download ends')
            else:
                filepath = custom_archive_path

                assert os.path.exists(filepath)

            print('archive', filepath, 'will extract')
            archive_path = filepath
            # поддерживаются те что втсречались в логхабе. не встречал  “bztar”, “xztar”, or “zstdtar”
            # поэтому их  поддержки нету
            # unpack_archive поддерживает след
            # “zip”, “tar”, “gztar”, “bztar”, “xztar”, or “zstdtar”
            # +gzip

            get_mime_type = lambda fp, p: \
            subprocess.run(['file', f'-{p}', fp], capture_output=True, text=True).stdout.split()[
                1].removesuffix(';').split('/')

            if archive_type is None:
                print("автоопределение типа архива")

                exact_tp = None

                pref, tp = get_mime_type(filepath, 'iz')
                if get_mime_type(filepath, 'i')[1] == "zip":
                    pref = "text"

                if pref == "text":
                    # not tar gz
                    pref2, tp2 = get_mime_type(filepath, 'i')
                    if pref2 == "application":
                        if tp2 == "gzip":
                            # gz
                            exact_tp = "GZIP"
                        elif tp2 == "zip":
                            exact_tp = "zip"
                        elif tp2 == "x-tar":
                            exact_tp = "tar"

                else:
                    if pref == "application" and tp == "x-tar":
                        # tar gz
                        exact_tp = "gztar"
            else:
                exact_tp = archive_type
                assert exact_tp in ["zip", "tar", "gztar", "GZIP"]
                print("тип архива статично указан (archive_type)")

            assert exact_tp is not None, f"{pref=} {tp=}"

            extract_dir = os.path.join(extract_dir, dataset_dir)
            os.makedirs(extract_dir, exist_ok=True)

            print(f"{exact_tp=}, {pref=}, {tp=}, {extract_dir=}")

            if exact_tp == "GZIP":
                # custom
                with gzip.open(filepath, 'rb') as f_in:
                    with open(extract_dir + f"/{dataset_dir}.log", 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                self.archive_type_is_gz = True
            elif exact_tp.isupper():
                raise NotImplementedError
            else:
                shutil.unpack_archive(filepath, extract_dir=extract_dir, format=exact_tp)

            print('extracted in', extract_dir)
            if remove_archive:
                # не удаляем из cache_Dir
                os.remove(archive_path)
                print('archive removed', )

    def get_time_content_label(self, raw_log: str) -> tuple[typing.Optional[datetime.datetime],
    str, int]:
        s = raw_log.split()
        label = 1
        try:
            if s[0] == "-": label = 0
            ts = int(s[1])
            dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
            cnt = ""
            if len(s) > self.cnt_ind:
                cnt = " ".join(s[self.cnt_ind:])
        except Exception:
            return None, "", 0

        return dt, cnt, label


import numpy as np
import typing
import math
import regex as re

patterns = [r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?',  # IP:PORT
            r'([0-9A-Fa-f]{2}:){11}[0-9A-Fa-f]{2}',  # Special MAC
            r'([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}',  # MAC
            r'[a-zA-Z0-9]*[:\.]*([/\\]+[^/\\\s\[\]]+)+[/\\]*',  # file path
            r'\b[0-9a-fA-F]{8}\b',
            r'\b[0-9a-fA-F]{10}\b',
            r'(\w+[\w\.]*)@(\w+[\w\.]*)\-(\w+[\w\.]*)',
            r'(\w+[\w\.]*)@(\w+[\w\.]*)',
            r'[a-zA-Z\.\:\-\_]*\d[a-zA-Z0-9\.\:\-\_]*',  # word
            ]
combined_pattern_str = '|'.join(patterns)

combined_pattern = re.compile(combined_pattern_str)
dots = re.compile(r'\.{3,}')

def replace_patterns(text):
    text = dots.sub(text, "..")
    text = combined_pattern.sub(text, "<*>")
    return text

import io

class SuperComputerDataset(Dataset):
    def __init__(
            self,
            source: Data,
            n: int,
            train_ratio: float = 0.3,
    ):
        assert n > 1
        assert 0.0 < train_ratio <= 1.0

        self.source = source
        if isinstance(source, Data):
            self.filepath = source.path_to_log
        else:
            raise ValueError

        self.max_lines = self.source.max_lines
        self.n = n
        self.window_size = source.window_size
        self.step_size = source.step_size
        self.train_ratio = train_ratio

        self.line_positions: list[int] = []  # позиции строк 0, n, 2n, ...
        self.labels_list: list[int] = []  # метка для каждой строки
        self.total_lines = 0
        self._num_samples: int = 0

        self._build_index()
        self.labels: np.ndarray[tuple[int], np.dtype[np.int8]] = np.asarray(self.labels_list, dtype=np.int8)
        self.file: typing.Optional[io.BufferedReader] = None

    def _build_index(self) -> None:
        line_idx = 0
        with open(self.filepath, "rb") as f:
            while True:
                pos = f.tell()
                raw = f.readline()

                if not raw or (line_idx >= self.max_lines):
                    break

                if line_idx % self.n == 0:
                    self.line_positions.append(pos)

                line = raw.decode("latin-1", errors="replace")
                self.labels_list.append(self.source.get_time_content_label(line)[2])


                line_idx += 1

        self.total_lines = line_idx

        if self.total_lines < self.window_size:
            total_samples = 0
        else:
            total_samples = (
                    (self.total_lines - self.window_size) // self.step_size + 1
            )

        self._num_samples = int(total_samples * self.train_ratio)

    def __getitem__(self, start_line: typing.Union[int, np.ndarray]) -> tuple[list, list, list, int]:
        if isinstance(start_line, np.ndarray):
            # 0d or scalar array
            start_line = start_line.item()

        if self.file is None:
            self.file = open(self.filepath, "rb")
        self.file = typing.cast(io.BufferedReader, self.file)

        if start_line < 0 or start_line >= self.total_lines:
            # последнее окно может быть не полным
            raise IndexError(start_line)

        anchor_idx = start_line // self.n
        anchor_line = anchor_idx * self.n
        skip_lines = start_line - anchor_line

        self.file.seek(self.line_positions[anchor_idx])
        for _ in range(skip_lines):
            if not self.file.readline():
                return [], [], [], 0
        window = []
        window_times = []
        window_raws = []
        for _ in range(self.window_size):
            raw = self.file.readline()
            if not raw:
                break
            raw_log = raw.decode("latin-1", errors="replace")
            dt, cnt, label = self.source.get_time_content_label(raw_log)
            cnt = replace_patterns(cnt)
            window_times.append(dt)
            window.append(cnt)
            window_raws.append(raw_log)

        window_label = max(self.labels[start_line: start_line + len(window)])
        return window, window_times, window_raws, window_label

    def __len__(self):
        return max(0, self.total_lines - self.window_size + 1)

import math
import typing
from torch.utils.data import IterableDataset

class EvaluationDataset(IterableDataset):
    def __init__(self, train_dataset: SuperComputerDataset):
        self.train_dataset = train_dataset
        self.window_size = train_dataset.window_size
        self.step_size = train_dataset.step_size
        self.total_lines = train_dataset.total_lines

        n_train = train_dataset._num_samples
        if n_train <= 0:
            self.start_line = 0
        else:
            last_train_start = (n_train - 1) * self.step_size
            self.start_line = min(last_train_start + self.window_size, self.total_lines)

        self.end_line = self.total_lines

    def __len__(self):
        last_possible_start = self.end_line - self.window_size
        if last_possible_start < self.start_line:
            return 0
        return (last_possible_start - self.start_line) // self.step_size + 1

    def __iter__(self):
        last_possible_start = self.end_line - self.window_size
        for start in range(self.start_line, last_possible_start + 1, self.step_size):
            yield self.train_dataset[start]

import datetime
import typing
import os


# gzip_fp -> extract_dir/data_dir/data_dir.log
# источник всех датасетов https://www.usenix.org/cfdr-data
# вместо логхаба
class BGL(Data):
    def __init__(self, window_size:int, step_size:int, max_lines:float=math.inf,
                 use_in_colab: bool = True):
        url = r'http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f.r43.cf2.rackcdn.com/hpc4/bgl2.gz'
        super().__init__(url, dataset_dir='bgl', use_in_colab=use_in_colab, window_size=window_size,
                         step_size=step_size, max_lines=max_lines)

        log_format = '<Label> <Id> <Date> <Code1> <Time> <Code2> <Component1> <Component2> <Level> <Content>'.split()
        self.cnt_ind = log_format.index("<Content>")
        self.path_to_log_dir = os.path.join(self.extract_dir, self.dataset_dir)
        self.path_to_log = os.path.join(self.path_to_log_dir, self.dataset_dir + ".log")


    def get_time_content_label(self, raw_log: str) -> tuple[typing.Optional[datetime.datetime],
    str, int]:
        s = raw_log.split()
        label = 1
        try:
            if s[0] == "-": label = 0
            t = s[4]
            dt = datetime.datetime.strptime(t, "%Y-%m-%d-%H.%M.%S.%f")
            cnt = ""
            if len(s) > self.cnt_ind:
                cnt = " ".join(s[self.cnt_ind:])
        except Exception:
            return None, "", 0

        return dt, cnt, label


class Tbird(Data):
    def __init__(self, window_size:int, step_size:int, max_lines:float=math.inf,
                 use_in_colab: bool = True):
        url = r'http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f.r43.cf2.rackcdn.com/hpc4/tbird2.gz'
        super().__init__(url, dataset_dir='tbird', use_in_colab=use_in_colab, window_size=window_size,
                         step_size=step_size, max_lines=max_lines)

        log_format = '<Label> <Id> <Date> <Admin> <Month> <Day> <Time> <AdminAddr> <Content>'.split()
        self.cnt_ind = log_format.index("<Content>")

        self.path_to_log_dir = os.path.join(self.extract_dir, self.dataset_dir)
        self.path_to_log = os.path.join(self.path_to_log_dir, self.dataset_dir + ".log")


class Spirit(Data):
    def __init__(self, window_size:int, step_size:int, max_lines:float=math.inf,
                 use_in_colab: bool = True):
        url = r'http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f.r43.cf2.rackcdn.com/hpc4/spirit2.gz'
        super().__init__(url, dataset_dir='spirit', use_in_colab=use_in_colab, window_size=window_size,
                         step_size=step_size, max_lines=max_lines)

        log_format = '<Label> <Id> <Date> <Admin> <Month> <Day> <Time> <AdminAddr> <Content>'.split()
        self.cnt_ind = log_format.index("<Content>")

        self.path_to_log_dir = os.path.join(self.extract_dir, self.dataset_dir)
        self.path_to_log = os.path.join(self.path_to_log_dir, self.dataset_dir + ".log")

class Liberty(Data):
    def __init__(self, window_size:int, step_size:int, max_lines:float=math.inf,
                 use_in_colab: bool = True):
        url = r'http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f.r43.cf2.rackcdn.com/hpc4/liberty2.gz'
        super().__init__(url, dataset_dir='liberty', use_in_colab=use_in_colab, window_size=window_size,
                         step_size=step_size, max_lines=max_lines)

        log_format = '<Label> <Id> <Date> <Admin> <Month> <Day> <Time> <AdminAddr> <Content>'.split()
        self.cnt_ind = log_format.index("<Content>")

        self.path_to_log_dir = os.path.join(self.extract_dir, self.dataset_dir)
        self.path_to_log = os.path.join(self.path_to_log_dir, self.dataset_dir + ".log")


import numpy as np

class BalancedSampler(Sampler):
    def __init__(self, dataset: SuperComputerDataset, target_ratio=0.3, max_samples=None, min_samples=1000,
                 seed: typing.Optional[int]=None):
        self.dataset = dataset
        self.target_ratio = target_ratio
        self.max_samples = max_samples
        self.min_samples = min_samples  # only if max_samples is None, min_samples can work

        self.W = dataset.window_size
        self.N = dataset.total_lines
        self.step_size = dataset.step_size
        labels = self.dataset.labels

        all_starts = np.arange(0, self.N - self.W + 1, self.step_size, dtype=np.int64)
        prefix = np.zeros(self.N + 1, dtype=np.int64)
        prefix[1:] = np.cumsum(labels)
        has_anom = (prefix[all_starts + self.W] - prefix[all_starts]) > 0
        self.normal_indices = all_starts[~has_anom]
        self.anomalous_indices = all_starts[has_anom]

        # self.normal_indices = np.where(self.labels == 0)[0]
        # self.anomalous_indices = np.where(self.labels == 1)[0]


        if len(self.anomalous_indices) <= len(self.normal_indices):
            self.minority_label, self.majority_label = "abnormal", "normal"
            self.minority_indices, self.majority_indices = self.anomalous_indices, self.normal_indices
        else:
            self.minority_label, self.majority_label = "normal", "abnormal"
            self.minority_indices, self.majority_indices = self.normal_indices, self.anomalous_indices

        print(f'sampler: a={len(self.anomalous_indices)}, n={len(self.normal_indices)}')
        t = len(self.anomalous_indices) + len(self.normal_indices)
        if t > 0: print(f'sampler: frac_a={len(self.anomalous_indices)/t*100:.2f}%, frac_n={len(self.normal_indices)/t*100:.2f}%')

        self.minority_count = max(int((self.target_ratio * len(self.majority_indices)) / (1 - self.target_ratio)), len(self.minority_indices))
        self.total_size = self.minority_count + len(self.majority_indices)

        if len(self.minority_indices) == 0:
            warnings.warn(f"нет ни одного окна с меткой {self.minority_label}")

        if len(self.majority_indices) == 0:
            warnings.warn(f"нет ни одного окна с меткой {self.majority_label}")

        if max_samples is not None:
            if max_samples > self.total_size:
                warnings.warn(f"max_samples > total, {max_samples=}, {self.total_size=}")
                warnings.warn(f"total осталось прежним")
                print(f"total c {self.total_size} остался как и был при {max_samples=}")

            else:
                # total >= max_samples
                # соханяем  нужную долю
                print(f"total c {self.total_size} урезали до: {max_samples}")
                self.total_size = max_samples
                self.minority_count = int(self.total_size * self.target_ratio)

        elif min_samples and self.total_size < min_samples:
            # если дополняем до нужного числа объектов то доля будет tar_rat
            print(f"min_samples: {self.total_size} увеличено до {self.min_samples}")
            self.total_size = min_samples
            self.minority_count = int(self.total_size * self.target_ratio)
            # если бы требовалось покрыть все минорные за одну эпоху
            # self.minority_count = min(
            #     self.total_size,
            #     max(int(self.total_size * self.target_ratio), len(self.minority_indices)),
            # )

        if len(self.minority_indices) == 0:
            # только мажоритарный класс
            warnings.warn("только мажоритарный класс")
            self.minority_count = 0
            self.total_size = len(self.majority_indices)
        if len(self.majority_indices) == 0:
            # только минорный класс
            warnings.warn("только минорный класс")
            self.minority_count = len(self.minority_indices)
            self.total_size = len(self.minority_indices)

        self.seed = seed
        self.rng = None

    def sample_count(self, array: np.ndarray, count: int) -> np.ndarray:
        if count < len(array):
            return self.rng.choice(array, count, replace=False)
        else:
            if len(array) == 0: return array
            reps, rem = divmod(count, len(array))
            reps_array = np.tile(array, reps)
            if rem:
                reps_array = np.concatenate([
                    reps_array,
                    self.rng.choice(array, rem, replace=False),
                ])
            self.rng.shuffle(reps_array)
            return reps_array


    def __iter__(self):

        wi = torch.utils.data.get_worker_info()
        base = self.seed if self.seed is not None else 0
        self.rng = np.random.default_rng(base + (wi.id if wi is not None else 0))

        oversampled_minority = self.sample_count(self.minority_indices, self.minority_count)
        oversampled_majority = self.sample_count( self.majority_indices, self.total_size - self.minority_count)
        combined = np.concatenate([oversampled_minority, oversampled_majority])
        self.rng.shuffle(combined)

        print(f'sampler: num {self.majority_label}={len(oversampled_majority)}, num {self.minority_label}={len(oversampled_minority)}')
        t = len(oversampled_majority) + len(oversampled_minority)
        if t > 0: print(f'sampler: frac {self.majority_label}={len(oversampled_majority)/t*100:.2f}, frac {self.minority_label}={len(oversampled_minority)/t*100:.2f}')
        return iter(combined)

    def __len__(self) -> int:
        return self.total_size


def make_collate_fn(encode_fn, jasper_batch):
    """
    encode_fn: list[str] -> Tensor [*, D]
    Окно (list[str]) кодируется чанками по jasper_batch строк,
    чтобы не улететь по памяти на длинных окнах.
    """
    def collate(batch):
        windows, times_list, labels = [], [], []
        for window, window_times, _raws, label in batch:
            if len(window) == 0:
                continue
            chunks = [window[i:i + jasper_batch]
                      for i in range(0, len(window), jasper_batch)]
            emb = torch.cat([encode_fn(c) for c in chunks], dim=0)  # [WS, D]
            windows.append(emb)
            times_list.append(window_times)
            labels.append(float(label))

        max_len = max(e.shape[0] for e in windows)
        D = windows[0].shape[1]

        padded = torch.zeros(len(windows), max_len, D, dtype=torch.float32)
        mask   = torch.zeros(len(windows), max_len, dtype=torch.long)
        for i, e in enumerate(windows):
            n = e.shape[0]
            padded[i, :n] = e
            mask[i, :n]   = 1

        return {
            "log_embs": padded,                            # [B, L, D]
            "attention_mask": mask,                        # [B, L]
            "times": times_list,                           # list[list[datetime]]
            "labels": torch.tensor(labels, dtype=torch.float32),
        }
    return collate