#!/usr/bin/env python3
"""받아쓰기 전에 원본 음성을 손보는 단계들.

네 단계(16kHz 단일 채널 변환 / 잡음 억제 / 긴 무음 잘라내기 / 소리 크기 고르게 펴기)를
각각 켜고 끌 수 있다. 무음을 잘라내도 남긴 구간을 함께 들고 다니므로,
받아쓴 시각을 원본 음성의 시각으로 되돌릴 수 있다.

numpy 와 PyAV 만 쓰므로 ffmpeg 같은 별도 프로그램을 설치할 필요가 없다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from faster_whisper.audio import decode_audio

SAMPLE_RATE = 16000
_EPS = 1e-10


@dataclass
class Prepared:
    """손본 음성과, 원본 시각으로 되돌리는 데 필요한 정보."""

    audio: np.ndarray
    sample_rate: int
    original_duration: float
    kept: list[tuple[float, float]] = field(default_factory=list)
    stages: list[str] = field(default_factory=list)
    silence_threshold_db: float | None = None

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate

    @property
    def removed(self) -> float:
        return max(0.0, self.original_duration - self.duration)

    def to_original_time(self, t: float) -> float:
        """손본 음성의 시각을 원본 음성의 시각으로 되돌린다."""
        if not self.kept:
            return t
        offset = 0.0
        for start, end in self.kept:
            span = end - start
            if t <= offset + span:
                return start + (t - offset)
            offset += span
        return self.kept[-1][1]


def load_16k_mono(path: str) -> np.ndarray:
    """PyAV 로 음성을 읽어 16kHz 단일 채널 float32 로 바꾼다."""
    return np.asarray(decode_audio(path, sampling_rate=SAMPLE_RATE), dtype=np.float32)


def _frame_starts(n_samples: int, frame: int, hop: int) -> np.ndarray:
    if n_samples < frame:
        return np.zeros(1, dtype=np.int64)
    return np.arange(0, n_samples - frame + 1, hop, dtype=np.int64)


def _frame_rms(x: np.ndarray, starts: np.ndarray, frame: int) -> np.ndarray:
    """프레임별 RMS. 누적합을 써서 큰 행렬을 만들지 않는다."""
    power = np.concatenate(([0.0], np.cumsum(np.square(x, dtype=np.float64))))
    ends = np.minimum(starts + frame, len(x))
    return np.sqrt((power[ends] - power[starts]) / np.maximum(ends - starts, 1))


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """True 가 이어지는 구간들의 (시작, 끝) 목록."""
    changes = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def suppress_noise(
    x: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    strength: float = 1.5,
    floor_db: float = -15.0,
    n_fft: int = 512,
) -> np.ndarray:
    """가장 조용한 구간에서 잡음의 모양을 배워 주파수마다 빼낸다(스펙트럼 차감)."""
    hop = n_fft // 4
    if len(x) < n_fft * 4:
        return x

    window = np.hanning(n_fft + 1)[:n_fft].astype(np.float64)
    bins = np.arange(n_fft)
    xp = np.pad(x.astype(np.float64), (n_fft, n_fft * 2))
    starts = _frame_starts(len(xp), n_fft, hop)

    # 조용한 프레임 10% 를 잡음 표본으로 삼는다.
    rms = _frame_rms(xp, starts, n_fft)
    quiet = starts[rms <= np.percentile(rms, 10)]
    if len(quiet) == 0:
        return x

    noise = np.zeros(n_fft // 2 + 1)
    for block in np.array_split(quiet, max(1, len(quiet) // 512)):
        frames = xp[block[:, None] + bins] * window
        noise += np.abs(np.fft.rfft(frames, axis=1)).sum(axis=0)
    noise /= len(quiet)

    floor = 10 ** (floor_db / 20)
    out = np.zeros(len(xp))
    weight = np.zeros(len(xp))
    square = window ** 2
    for block in np.array_split(starts, max(1, len(starts) // 512)):
        frames = xp[block[:, None] + bins] * window
        spec = np.fft.rfft(frames, axis=1)
        mag = np.abs(spec)
        # 잡음만큼 깎되, 너무 깎아 소리가 뭉개지지 않게 바닥을 둔다.
        scale = np.maximum(mag - strength * noise, floor * mag) / np.maximum(mag, _EPS)
        restored = np.fft.irfft(spec * scale, n=n_fft, axis=1) * window
        for i, start in enumerate(block):
            out[start:start + n_fft] += restored[i]
            weight[start:start + n_fft] += square

    y = out / np.maximum(weight, _EPS)
    return y[n_fft:n_fft + len(x)].astype(np.float32)


def trim_long_silence(
    x: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    max_silence: float = 0.8,
    keep_silence: float = 0.2,
    threshold_db: float | None = None,
) -> tuple[np.ndarray, list[tuple[float, float]], float]:
    """정해진 길이보다 긴 무음만 잘라낸다. 남긴 구간(원본 시각)도 함께 돌려준다."""
    frame = max(1, int(0.02 * sample_rate))
    hop = max(1, frame // 2)
    starts = _frame_starts(len(x), frame, hop)
    rms_db = 20 * np.log10(np.maximum(_frame_rms(x, starts, frame), _EPS))

    if threshold_db is None:
        # 가장 큰 소리에서 35dB 아래를 무음으로 본다.
        threshold_db = max(float(np.percentile(rms_db, 95)) - 35.0, -55.0)

    speech = rms_db > threshold_db
    whole = [(0.0, len(x) / sample_rate)]
    if not speech.any():
        return x, whole, threshold_db

    pad = int(keep_silence * sample_rate)
    gap = int(max_silence * sample_rate)
    intervals: list[list[int]] = []
    for first, last in _true_runs(speech):
        low = max(0, int(starts[first]) - pad)
        high = min(len(x), int(starts[last - 1]) + frame + pad)
        # 짧은 무음은 그대로 둔다. 긴 무음일 때만 구간이 끊긴다.
        if intervals and low - intervals[-1][1] < gap:
            intervals[-1][1] = high
        else:
            intervals.append([low, high])

    y = np.concatenate([x[low:high] for low, high in intervals])
    kept = [(low / sample_rate, high / sample_rate) for low, high in intervals]
    return y.astype(np.float32), kept, threshold_db


def even_out_loudness(
    x: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    target_dbfs: float = -20.0,
    max_gain_db: float = 18.0,
    window_sec: float = 0.4,
) -> np.ndarray:
    """구간마다 크기를 재어 큰 데는 낮추고 작은 데는 올려 고르게 편다."""
    frame = max(1, int(window_sec * sample_rate))
    hop = max(1, frame // 2)
    starts = _frame_starts(len(x), frame, hop)
    rms = _frame_rms(x, starts, frame)

    target = 10 ** (target_dbfs / 20)
    quiet_floor = 10 ** (-50.0 / 20)  # 이보다 조용하면 잡음으로 보고 끌어올리지 않는다
    limit = 10 ** (max_gain_db / 20)
    gain = np.clip(target / np.maximum(rms, quiet_floor), 1 / limit, limit)

    if len(gain) >= 3:  # 소리가 출렁이지 않게 이웃끼리 다듬는다
        padded = np.pad(gain, 1, mode="edge")
        gain = (padded[:-2] + padded[1:-1] + padded[2:]) / 3

    centers = starts + frame / 2
    y = x * np.interp(np.arange(len(x)), centers, gain).astype(np.float32)

    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 0.99:  # 찌그러지지 않게 꼭대기를 눌러 둔다
        y *= 0.99 / peak
    return y.astype(np.float32)


def prepare(
    path: str,
    *,
    denoise: bool = False,
    trim_silence: bool = False,
    normalize: bool = False,
    denoise_strength: float = 1.5,
    max_silence: float = 0.8,
    keep_silence: float = 0.2,
    silence_db: float | None = None,
    target_dbfs: float = -20.0,
) -> Prepared:
    """켜진 단계만 차례로 적용한다. 순서: 잡음 억제 → 무음 잘라내기 → 크기 고르게."""
    audio = load_16k_mono(path)
    result = Prepared(
        audio=audio,
        sample_rate=SAMPLE_RATE,
        original_duration=len(audio) / SAMPLE_RATE,
        stages=["16k-mono"],
    )

    if denoise:
        result.audio = suppress_noise(result.audio, SAMPLE_RATE, strength=denoise_strength)
        result.stages.append("denoise")

    if trim_silence:
        result.audio, result.kept, used_db = trim_long_silence(
            result.audio,
            SAMPLE_RATE,
            max_silence=max_silence,
            keep_silence=keep_silence,
            threshold_db=silence_db,
        )
        result.silence_threshold_db = used_db
        result.stages.append("trim-silence")

    if normalize:
        result.audio = even_out_loudness(result.audio, SAMPLE_RATE, target_dbfs=target_dbfs)
        result.stages.append("normalize")

    return result
