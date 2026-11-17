# dataset module

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
import hashlib
import pickle
import tempfile
from typing import Optional

# ============================================================
# Хелперы для кэша
# ============================================================

def _stable_hash(obj) -> str:
    import json
    s = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest()


def _file_fingerprint(path: str) -> dict:
    try:
        st = os.stat(path)
        return {"size": st.st_size, "mtime": int(st.st_mtime)}
    except OSError:
        return {"size": 0, "mtime": 0}


def _atomic_save(obj, path: str):
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".pt")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _safe_load(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[cache] load failed {path}: {e}")
        return None


# ============================================================
# Data (базовый класс с загрузкой/распаковкой)
# ============================================================

class Data:
    path_to_log_dir: str
    path_to_log: str
    cnt_ind: int

    def __init__(self, downloaded_url: str,
                 cache_dir: str = os.path.expanduser('~/.dataset'),
                 dataset_dir: str = 'hdfs',
                 extract_dir: str = 'archive_extracted',
                 repeat_download: bool = False,
                 custom_archive_path: typing.Optional[str] = None,
                 remove_archive: bool = False,
                 archive_type=None, use_in_colab: bool = False,
                 encoding: str = "latin-1", errors: str = "replace",
                 max_lines: float = math.inf,
                 start_line: int = 0,
                 end_line: typing.Optional[int] = None,
                 window_size: int = 200,
                 step_size: int = 200):
        assert window_size > 1
        assert step_size >= 1
        assert max_lines >= 0
        assert start_line >= 0
        assert end_line is None or end_line > start_line

        colab_content = "/content/"
        if use_in_colab:
            cache_dir = colab_content
            extract_dir = colab_content

        if os.path.exists(colab_content) and not use_in_colab:
            warnings.warn("use_in_colab = False но /content/ существует")

        self.max_lines = max_lines
        self.start_line = start_line
        self.end_line = end_line
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

        if not os.path.exists(cache_dir):
            raise FileExistsError

        if not repeat_download and os.path.exists(os.path.join(cache_dir, dataset_dir)):
            print('exist')
        else:
            if custom_archive_path is None:
                filepath = os.path.join(cache_dir, f'archive_{dataset_dir}')
                if os.path.exists(filepath) and not repeat_download:
                    print('already downloaded')
                else:
                    print('downloading...')
                    args = ['curl', '-o', filepath,
                            '--retry-delay', '1', '--retry', str(int(100_000)),
                            '--retry-all-errors', '-L', downloaded_url]
                    res = subprocess.run(args, capture_output=True, text=True)
                    print('stdout from curl', res.stdout)
                    print('stderr from curl', res.stderr)
                    print('download ends')
            else:
                filepath = custom_archive_path
                assert os.path.exists(filepath)

            print('archive', filepath, 'will extract')
            archive_path = filepath

            get_mime_type = lambda fp, p: \
            subprocess.run(['file', f'-{p}', fp],
                           capture_output=True, text=True).stdout.split()[
                1].removesuffix(';').split('/')

            if archive_type is None:
                print("автоопределение типа архива")
                exact_tp = None
                pref, tp = get_mime_type(filepath, 'iz')
                if get_mime_type(filepath, 'i')[1] == "zip":
                    pref = "text"
                if pref == "text":
                    pref2, tp2 = get_mime_type(filepath, 'i')
                    if pref2 == "application":
                        if tp2 == "gzip":
                            exact_tp = "GZIP"
                        elif tp2 == "zip":
                            exact_tp = "zip"
                        elif tp2 == "x-tar":
                            exact_tp = "tar"
                else:
                    if pref == "application" and tp == "x-tar":
                        exact_tp = "gztar"
            else:
                exact_tp = archive_type
                assert exact_tp in ["zip", "tar", "gztar", "GZIP"]
                print("тип архива статично указан (archive_type)")

            assert exact_tp is not None, f"{pref=} {tp=}"

            extract_dir = os.path.join(extract_dir, dataset_dir)
            os.makedirs(extract_dir, exist_ok=True)

            if exact_tp == "GZIP":
                with gzip.open(filepath, 'rb') as f_in:
                    with open(extract_dir + f"/{dataset_dir}.log", 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                self.archive_type_is_gz = True
            else:
                shutil.unpack_archive(filepath, extract_dir=extract_dir,
                                      format=exact_tp)

            print('extracted in', extract_dir)
            if remove_archive:
                os.remove(archive_path)
                print('archive removed')

    def get_time_content_label(self, raw_log: str):
        s = raw_log.split()
        label = 1
        try:
            if s[0] == "-":
                label = 0
            ts = int(s[1])
            dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
            cnt = ""
            if len(s) > self.cnt_ind:
                cnt = " ".join(s[self.cnt_ind:])
        except Exception:
            return None, "", 0
        return dt, cnt, label


# ============================================================
# Нормализация
# ============================================================

import numpy as np
import math
import regex as re


_patterns = [
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?',
    r'([0-9A-Fa-f]{2}:){11}[0-9A-Fa-f]{2}',
    r'([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}',
    r'[a-zA-Z0-9]*[:\.]*([/\\]+[^/\\\s\[\]]+)+[/\\]*',
    r'\b[0-9a-fA-F]{8}\b',
    r'\b[0-9a-fA-F]{10}\b',
    r'(\w+[\w\.]*)@(\w+[\w\.]*)\-(\w+[\w\.]*)',
    r'(\w+[\w\.]*)@(\w+[\w\.]*)',
]
_entity_pat = re.compile('|'.join(_patterns))
_camel_1 = re.compile(r'([a-z0-9])([A-Z])')
_camel_2 = re.compile(r'([A-Z]+)([A-Z][a-z])')
_non_alpha = re.compile(r'[^a-zA-Z]+')
_spaces = re.compile(r'\s+')
_digits = re.compile(r'\d+')


def normalize_text(s: str) -> str:
    s = _entity_pat.sub(' ', s)
    s = _camel_1.sub(r'\1_\2', s)
    s = _camel_2.sub(r'\1_\2', s)
    s = _digits.sub(r' N ', s)
    s = _non_alpha.sub(' ', s)
    s = s.lower()
    s = _spaces.sub(' ', s).strip()
    return s


import io


# ============================================================
# SuperComputerDataset
# ============================================================

class SuperComputerDataset(Dataset):
    def __init__(self, source: Data, n: int, train_ratio: float = 0.3,
                 cache_dir: typing.Optional[str] = None,
                 preprocess_fn: typing.Optional[typing.Callable[[str], str]] = None):

        self.preprocess_fn = preprocess_fn or normalize_text
        assert n > 1
        assert 0.0 < train_ratio <= 1.0

        self.source = source
        if not isinstance(source, Data):
            raise ValueError
        self.filepath = source.path_to_log

        self.max_lines = source.max_lines
        self.start_line = source.start_line
        self.end_line = source.end_line
        self.n = n
        self.window_size = source.window_size
        self.step_size = source.step_size
        self.train_ratio = train_ratio
        self.cache_dir = cache_dir

        self.line_positions: list[int] = []
        self.labels_list: list[int] = []
        self.total_lines = 0
        self._num_samples: int = 0

        fp_payload = {
            "filepath":   self.filepath,
            "file":       _file_fingerprint(self.filepath),
            "n":          n,
            "win_size":   self.window_size,
            "step_size":  self.step_size,
            "start_line": self.start_line,
            "end_line":   self.end_line,
            "max_lines":  str(self.max_lines),
            "train_ratio": self.train_ratio,
        }
        self._cache_fp = _stable_hash(fp_payload)
        cache_path = self._cache_path()

        loaded = False
        if cache_path:
            data = _safe_load(cache_path)
            if data is not None and data.get("fingerprint") == self._cache_fp:
                self.line_positions = data["line_positions"]
                self.labels_list    = data["labels_list"]
                self.total_lines    = data["total_lines"]
                self._num_samples   = data["_num_samples"]
                print(f"[data] cache HIT: {cache_path}")
                loaded = True

        if not loaded:
            print(f"[data] building index...")
            self._build_index()
            if cache_path:
                _atomic_save({
                    "fingerprint":    self._cache_fp,
                    "line_positions": self.line_positions,
                    "labels_list":    self.labels_list,
                    "total_lines":    self.total_lines,
                    "_num_samples":   self._num_samples,
                }, cache_path)
                print(f"[data] cache SAVED: {cache_path}")

        self.labels = np.asarray(self.labels_list, dtype=np.int8)
        self.file: typing.Optional[io.BufferedReader] = None

    def _cache_path(self) -> typing.Optional[str]:
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, f"dataset_{self._cache_fp}.pkl")

    def _build_index(self) -> None:
        max_read = None if math.isinf(self.max_lines) else int(self.max_lines)

        line_idx = 0
        with open(self.filepath, "rb") as f:
            for _ in range(self.start_line):
                if not f.readline():
                    break

            while True:
                if self.end_line is not None and \
                        (self.start_line + line_idx) >= self.end_line:
                    break
                if max_read is not None and line_idx >= max_read:
                    break

                pos = f.tell()
                raw = f.readline()
                if not raw:
                    break

                if line_idx % self.n == 0:
                    self.line_positions.append(pos)

                line = raw.decode("latin-1", errors="replace")
                self.labels_list.append(
                    self.source.get_time_content_label(line)[2])
                line_idx += 1

        self.total_lines = line_idx
        end_display = self.start_line + self.total_lines
        print(f"[data] read [{self.start_line}, {end_display}) = "
              f"{self.total_lines} lines "
              f"(max_lines={self.max_lines}, end_line={self.end_line})")

        if self.total_lines < self.window_size:
            total_samples = 0
        else:
            total_samples = (self.total_lines - self.window_size) \
                // self.step_size + 1

        self._num_samples = int(total_samples * self.train_ratio)

    def __getitem__(self, start_line):
        if isinstance(start_line, np.ndarray):
            start_line = start_line.item()

        if self.file is None:
            self.file = open(self.filepath, "rb")
        self.file = typing.cast(io.BufferedReader, self.file)

        if start_line < 0 or start_line >= self.total_lines:
            raise IndexError(start_line)

        anchor_idx = start_line // self.n
        anchor_line = anchor_idx * self.n
        skip_lines = start_line - anchor_line

        self.file.seek(self.line_positions[anchor_idx])
        for _ in range(skip_lines):
            if not self.file.readline():
                return [], [], [], 0

        window, window_times, window_raws = [], [], []
        for _ in range(self.window_size):
            raw = self.file.readline()
            if not raw:
                break
            raw_log = raw.decode("latin-1", errors="replace")
            dt, cnt, label = self.source.get_time_content_label(raw_log)
            cnt = self.preprocess_fn(cnt).strip()
            window_times.append(dt)
            window.append(cnt)
            window_raws.append(raw_log)

        window_label = max(self.labels[start_line: start_line + len(window)])
        return window, window_times, window_raws, window_label

    def __len__(self):
        return max(0, self.total_lines - self.window_size + 1)


# ============================================================
# EvaluationDataset
# ============================================================

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
            self.start_line = min(last_train_start + self.window_size,
                                  self.total_lines)

        self.end_line = self.total_lines

    def __len__(self):
        last_possible_start = self.end_line - self.window_size
        if last_possible_start < self.start_line:
            return 0
        return (last_possible_start - self.start_line) // self.step_size + 1

    def __iter__(self):
        last_possible_start = self.end_line - self.window_size
        for start in range(self.start_line, last_possible_start + 1,
                           self.step_size):
            yield self.train_dataset[start]


# ============================================================
# Конкретные датасеты
# ============================================================

class BGL(Data):
    def __init__(self, window_size: int, step_size: int,
                 max_lines: float = math.inf,
                 start_line: int = 0,
                 end_line: typing.Optional[int] = None,
                 use_in_colab: bool = True):
        url = (r'http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f'
               r'.r43.cf2.rackcdn.com/hpc4/bgl2.gz')
        super().__init__(url, dataset_dir='bgl', use_in_colab=use_in_colab,
                         window_size=window_size, step_size=step_size,
                         max_lines=max_lines, start_line=start_line,
                         end_line=end_line)
        log_format = ('<Label> <Id> <Date> <Code1> <Time> <Code2> '
                      '<Component1> <Component2> <Level> <Content>').split()
        self.cnt_ind = log_format.index("<Content>")
        self.path_to_log_dir = os.path.join(self.extract_dir,
                                            self.dataset_dir)
        self.path_to_log = os.path.join(self.path_to_log_dir,
                                        self.dataset_dir + ".log")

    def get_time_content_label(self, raw_log: str):
        s = raw_log.split()
        label = 1
        try:
            if s[0] == "-":
                label = 0
            t = s[4]
            dt = datetime.datetime.strptime(t, "%Y-%m-%d-%H.%M.%S.%f")
            cnt = ""
            if len(s) > self.cnt_ind:
                cnt = " ".join(s[self.cnt_ind:])
        except Exception:
            return None, "", 0
        return dt, cnt, label


class Tbird(Data):
    def __init__(self, window_size: int, step_size: int,
                 max_lines: float = math.inf,
                 start_line: int = 0,
                 end_line: typing.Optional[int] = None,
                 use_in_colab: bool = True):
        url = (r'http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f'
               r'.r43.cf2.rackcdn.com/hpc4/tbird2.gz')
        super().__init__(url, dataset_dir='tbird', use_in_colab=use_in_colab,
                         window_size=window_size, step_size=step_size,
                         max_lines=max_lines, start_line=start_line,
                         end_line=end_line)
        log_format = ('<Label> <Id> <Date> <Admin> <Month> <Day> <Time> '
                      '<AdminAddr> <Content>').split()
        self.cnt_ind = log_format.index("<Content>")
        self.path_to_log_dir = os.path.join(self.extract_dir,
                                            self.dataset_dir)
        self.path_to_log = os.path.join(self.path_to_log_dir,
                                        self.dataset_dir + ".log")

    def get_time_content_label(self, raw_log: str):
        res = super().get_time_content_label(raw_log)
        if res[0] is not None:
            return res
        s = raw_log.split()
        label = 1
        try:
            if s[0] == "-":
                label = 0
            date_str, time_str = s[2], s[6]
            dt = datetime.datetime.strptime(
                f"{date_str} {time_str}", "%Y.%m.%d %H:%M:%S")
            cnt = " ".join(s[self.cnt_ind:]) if len(s) > self.cnt_ind else ""
        except Exception:
            return None, "", 0
        return dt, cnt, label


class Spirit(Data):
    def __init__(self, window_size: int, step_size: int,
                 max_lines: float = math.inf,
                 start_line: int = 0,
                 end_line: typing.Optional[int] = None,
                 use_in_colab: bool = True):
        url = (r'http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f'
               r'.r43.cf2.rackcdn.com/hpc4/spirit2.gz')
        super().__init__(url, dataset_dir='spirit', use_in_colab=use_in_colab,
                         window_size=window_size, step_size=step_size,
                         max_lines=max_lines, start_line=start_line,
                         end_line=end_line)
        log_format = ('<Label> <Id> <Date> <Admin> <Month> <Day> <Time> '
                      '<AdminAddr> <Content>').split()
        self.cnt_ind = log_format.index("<Content>")
        self.path_to_log_dir = os.path.join(self.extract_dir,
                                            self.dataset_dir)
        self.path_to_log = os.path.join(self.path_to_log_dir,
                                        self.dataset_dir + ".log")

    def get_time_content_label(self, raw_log: str):
        res = super().get_time_content_label(raw_log)
        if res[0] is not None:
            return res
        s = raw_log.split()
        label = 1
        try:
            if s[0] == "-":
                label = 0
            date_str, time_str = s[2], s[6]
            dt = datetime.datetime.strptime(
                f"{date_str} {time_str}", "%Y.%m.%d %H:%M:%S")
            cnt = " ".join(s[self.cnt_ind:]) if len(s) > self.cnt_ind else ""
        except Exception:
            return None, "", 0
        return dt, cnt, label


class Liberty(Data):
    def __init__(self, window_size: int, step_size: int,
                 max_lines: float = math.inf,
                 start_line: int = 0,
                 end_line: typing.Optional[int] = None,
                 use_in_colab: bool = True):
        url = (r'http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f'
               r'.r43.cf2.rackcdn.com/hpc4/liberty2.gz')
        super().__init__(url, dataset_dir='liberty',
                         use_in_colab=use_in_colab,
                         window_size=window_size, step_size=step_size,
                         max_lines=max_lines, start_line=start_line,
                         end_line=end_line)
        log_format = ('<Label> <Id> <Date> <Admin> <Month> <Day> <Time> '
                      '<AdminAddr> <Content>').split()
        self.cnt_ind = log_format.index("<Content>")
        self.path_to_log_dir = os.path.join(self.extract_dir,
                                            self.dataset_dir)
        self.path_to_log = os.path.join(self.path_to_log_dir,
                                        self.dataset_dir + ".log")

    def get_time_content_label(self, raw_log: str):
        res = super().get_time_content_label(raw_log)
        if res[0] is not None:
            return res
        s = raw_log.split()
        label = 1
        try:
            if s[0] == "-":
                label = 0
            date_str, time_str = s[2], s[6]
            dt = datetime.datetime.strptime(
                f"{date_str} {time_str}", "%Y.%m.%d %H:%M:%S")
            cnt = " ".join(s[self.cnt_ind:]) if len(s) > self.cnt_ind else ""
        except Exception:
            return None, "", 0
        return dt, cnt, label


class BalancedSampler(Sampler):
    """
    Сэмплер ТОЛЬКО по [valid_start, valid_end) стартов окон.
    Осэмплирует minority до target_ratio, но никогда не превышает
    max_oversample_factor × n_min. Никогда не трогает индексы вне диапазона.
    """

    def __init__(
        self,
        dataset,
        valid_range: tuple,          # (start, end) — старты окон, ОБЯЗАТЕЛЬНО
        target_ratio: float = 0.5,
        max_samples: Optional[int] = None,
        max_oversample_factor: float = 10.0,
        min_minority_for_balance: int = 50,
        seed: int = 42,
        use_oversampling: bool = True,
    ):
        assert 0.0 < target_ratio < 1.0
        assert max_oversample_factor >= 1.0

        N, W, step = dataset.total_lines, dataset.window_size, dataset.step_size
        vmin = max(0, int(valid_range[0]))
        vmax = min(N - W + 1, int(valid_range[1]))

        self.seed = seed
        self.valid_range = (vmin, vmax)
        self.indices_, self.stats = self._build(
            dataset, vmin, vmax, target_ratio,
            max_samples, max_oversample_factor,
            min_minority_for_balance, use_oversampling)

    @staticmethod
    def _build(dataset, vmin, vmax, target_ratio,
               max_samples, max_oversample_factor,
               min_minority_for_balance, use_oversampling):
        N, W, step = dataset.total_lines, dataset.window_size, dataset.step_size
        starts = np.arange(vmin, vmax, step, dtype=np.int64)
        if len(starts) == 0:
            return np.array([], dtype=np.int64), {"reason": "empty_range"}

        labels = dataset.labels
        prefix = np.zeros(N + 1, dtype=np.int64)
        prefix[1:] = np.cumsum(labels)
        has_anom = (prefix[starts + W] - prefix[starts]) > 0

        anom = starts[has_anom]
        norm = starts[~has_anom]

        if len(anom) <= len(norm):
            minority, majority, min_label = anom, norm, 1
        else:
            minority, majority, min_label = norm, anom, 0

        n_min, n_maj = len(minority), len(majority)

        rng = np.random.default_rng(42)

        def _pass_through(reason):
            idx = starts.copy()
            rng.shuffle(idx)
            return idx, {
                "reason": reason, "range": (vmin, vmax),
                "n_min_natural": n_min, "n_maj_natural": n_maj,
                "total": len(idx), "minority_label": min_label,
            }

        if not use_oversampling:
            return _pass_through("oversampling_disabled")
        if n_min == 0:
            return _pass_through("no_minority")
        if n_min < min_minority_for_balance:
            return _pass_through(f"minority_too_rare({n_min}<{min_minority_for_balance})")

        # Желаемое число minority
        target_min = int(target_ratio * n_maj / (1.0 - target_ratio))
        # Никогда не undersample minority
        target_min = max(n_min, target_min)
        # Жёсткий кап
        cap = int(n_min * max_oversample_factor)
        target_min = min(target_min, cap)

        # Верхняя граница по max_samples
        if max_samples is not None and target_min + n_maj > max_samples:
            desired_min = int(max_samples * target_ratio)
            desired_min = min(desired_min, cap)                    # ← кап соблюдён
            desired_min = max(desired_min, min(n_min, max_samples))# ← не undersample
            desired_maj = max_samples - desired_min
            if desired_maj > n_maj:                               # не хватило majority
                desired_min = max(min(cap, max_samples - n_maj), desired_min)
                desired_maj = n_maj
            target_min, n_maj_target = desired_min, desired_maj
        else:
            n_maj_target = n_maj

        # Сборка: minority — с replacement при необходимости,
        # majority — без (subsample если надо)
        if target_min <= n_min:
            min_out = rng.choice(minority, target_min, replace=False)
        else:
            extra = target_min - n_min
            min_out = np.concatenate(
                [minority, rng.choice(minority, extra, replace=True)])

        if n_maj_target <= n_maj:
            maj_out = rng.choice(majority, n_maj_target, replace=False)
        else:  # теоретически не должно происходить
            maj_out = majority

        idx = np.concatenate([min_out, maj_out])
        rng.shuffle(idx)

        return idx, {
            "reason": "balanced",
            "range": (vmin, vmax),
            "n_min_natural": n_min, "n_maj_natural": n_maj,
            "n_min_target": len(min_out), "n_maj_target": len(maj_out),
            "actual_minority_ratio": len(min_out) / len(idx),
            "minority_label": min_label,
            "total": len(idx),
        }

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (wi.id if wi else 0))
        idx = self.indices_.copy()
        rng.shuffle(idx)
        return iter(idx.tolist())

    def __len__(self):
        return len(self.indices_)

# ============================================================
# Preprocessing registry
# ============================================================

_REPLACE_PATTERNS = [
    r'True', r'true', r'False', r'false',
    r'\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|'
    r'thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|'
    r'thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b',
    r'\b(Mon|Monday|Tue|Tuesday|Wed|Wednesday|Thu|Thursday|Fri|Friday|Sat|Saturday|Sun|Sunday)\b',
    r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
    r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
    r'\s+(\d{1,2})\s+\b',
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?',
    r'([0-9A-Fa-f]{2}:){11}[0-9A-Fa-f]{2}',
    r'([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}',
    r'[a-zA-Z0-9]*[:\.]*([/\\]+[^/\\\s\[\]]+)+[/\\]*',
    r'\b[0-9a-fA-F]{8}\b', r'\b[0-9a-fA-F]{10}\b',
    r'(\w+[\w\.]*)@(\w+[\w\.]*)\-(\w+[\w\.]*)',
    r'(\w+[\w\.]*)@(\w+[\w\.]*)',
    r'[a-zA-Z\.\:\-\_]*\d[a-zA-Z0-9\.\:\-\_]*',
]
_REPLACE_COMBINED = '|'.join(_REPLACE_PATTERNS)
_RE_MULTIDOT = re.compile(r'[\.]{3,}')


def replace_patterns(text: str) -> str:
    text = _RE_MULTIDOT.sub('.. ', text)
    return re.sub(_REPLACE_COMBINED, '<*>', text)


def _preproc_none(s):      return s.strip()
def _preproc_normalize(s): return normalize_text(s).strip()
def _preproc_replace(s):   return replace_patterns(s).strip()
def _preproc_both(s):      return normalize_text(replace_patterns(s)).strip()


PREPROCESSORS: typing.Dict[str, typing.Callable[[str], str]] = {
    "none":      _preproc_none,
    "normalize": _preproc_normalize,   # ваш старый preprocessing
    "replace":   _preproc_replace,     # LogLLM-style
    "both":      _preproc_both,
}


def get_preprocess_fn(mode: str) -> typing.Callable[[str], str]:
    if mode not in PREPROCESSORS:
        raise ValueError(
            f"Unknown preprocess mode={mode!r}. "
            f"Available: {list(PREPROCESSORS.keys())}")
    return PREPROCESSORS[mode]