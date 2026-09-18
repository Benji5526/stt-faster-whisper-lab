#!/usr/bin/env python3
"""손으로 받아 적은 정답과 대조해 글자 단위 오류율(CER)을 센다.

- 띄어쓰기와 문장부호는 빼고 센다.
- 바꾸기(치환)는 한 번으로 센다. 빠짐(삭제), 끼어듦(삽입)도 각각 한 번이다.
- CER = (바꾸기 + 빠짐 + 끼어듦) / 정답 글자 수
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import numpy as np

# 정답과 인식 결과가 너무 길면 대조표가 메모리를 많이 먹는다.
MAX_CELLS = 9_000_000

_MATCH, _SUB, _DELETE, _INSERT = 0, 1, 2, 3
_LABEL = {_SUB: "바꿈", _DELETE: "빠짐", _INSERT: "끼어듦"}


@dataclass
class Error:
    kind: str
    ref_text: str
    hyp_text: str
    ref_at: int
    hyp_at: int


@dataclass
class Report:
    ref: str
    hyp: str
    substitutions: int
    deletions: int
    insertions: int
    errors: list[Error]

    @property
    def total(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def cer(self) -> float:
        return self.total / len(self.ref) if self.ref else 0.0


def normalize(text: str) -> str:
    """띄어쓰기와 문장부호를 빼고, 영문은 소문자로 맞춘다."""
    kept = []
    for ch in text:
        if ch.isspace():
            continue
        if unicodedata.category(ch)[0] in "PS":  # 문장부호와 기호
            continue
        kept.append(ch.lower())
    return "".join(kept)


def _distance_table(ref: str, hyp: str) -> np.ndarray:
    """편집 거리표. 한 줄씩 numpy 로 채운다."""
    n, m = len(ref), len(hyp)
    table = np.zeros((n + 1, m + 1), dtype=np.int32)
    table[0] = np.arange(m + 1)
    columns = np.arange(m + 1, dtype=np.int32)
    hyp_codes = np.frombuffer(hyp.encode("utf-32-le"), dtype=np.uint32) if m else np.zeros(0, np.uint32)

    for i, ch in enumerate(ref, start=1):
        previous = table[i - 1]
        row = np.empty(m + 1, dtype=np.int32)
        row[0] = i
        if m:
            code = np.frombuffer(ch.encode("utf-32-le"), dtype=np.uint32)[0]
            # 바꾸기(대각선)와 빠짐(위쪽)을 먼저 본다.
            row[1:] = np.minimum(previous[:-1] + (hyp_codes != code), previous[1:] + 1)
        # 끼어듦(왼쪽)은 값이 1씩 늘어나므로 누적 최솟값으로 한 번에 반영한다.
        row = np.minimum.accumulate(row - columns) + columns
        table[i] = row
    return table


def _backtrace(table: np.ndarray, ref: str, hyp: str) -> list[tuple[int, int, int]]:
    """어디서 어떤 편집이 있었는지 뒤에서부터 되짚는다."""
    ops: list[tuple[int, int, int]] = []
    i, j = len(ref), len(hyp)
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            same = ref[i - 1] == hyp[j - 1]
            if table[i][j] == table[i - 1][j - 1] + (0 if same else 1):
                ops.append((_MATCH if same else _SUB, i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i > 0 and table[i][j] == table[i - 1][j] + 1:
            ops.append((_DELETE, i - 1, j))
            i -= 1
            continue
        ops.append((_INSERT, i, j - 1))
        j -= 1
    ops.reverse()
    return ops


def _group(ops: list[tuple[int, int, int]], ref: str, hyp: str) -> list[Error]:
    """붙어 있는 틀린 자리는 하나로 묶어 보여 준다."""
    errors: list[Error] = []
    index = 0
    while index < len(ops):
        if ops[index][0] == _MATCH:
            index += 1
            continue
        start = index
        while index < len(ops) and ops[index][0] != _MATCH:
            index += 1
        chunk = ops[start:index]

        ref_span = [pos for kind, pos, _ in chunk if kind in (_SUB, _DELETE)]
        hyp_span = [pos for kind, _, pos in chunk if kind in (_SUB, _INSERT)]
        kinds = {kind for kind, _, _ in chunk}
        kind = _LABEL[chunk[0][0]] if len(kinds) == 1 else "섞임"

        errors.append(
            Error(
                kind=kind,
                ref_text=ref[ref_span[0]:ref_span[-1] + 1] if ref_span else "",
                hyp_text=hyp[hyp_span[0]:hyp_span[-1] + 1] if hyp_span else "",
                ref_at=ref_span[0] if ref_span else chunk[0][1],
                hyp_at=hyp_span[0] if hyp_span else chunk[0][2],
            )
        )
    return errors


def compare(reference: str, hypothesis: str) -> Report:
    """정답과 받아쓴 글을 대조해 오류율과 틀린 자리를 낸다."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    if (len(ref) + 1) * (len(hyp) + 1) > MAX_CELLS:
        raise ValueError(
            f"글이 너무 길어 대조표를 만들 수 없습니다 "
            f"(정답 {len(ref)}자 x 인식 {len(hyp)}자). 파일을 나눠서 대조해 주세요."
        )

    table = _distance_table(ref, hyp)
    ops = _backtrace(table, ref, hyp)
    counts = {_SUB: 0, _DELETE: 0, _INSERT: 0}
    for kind, _, _ in ops:
        if kind in counts:
            counts[kind] += 1

    return Report(
        ref=ref,
        hyp=hyp,
        substitutions=counts[_SUB],
        deletions=counts[_DELETE],
        insertions=counts[_INSERT],
        errors=_group(ops, ref, hyp),
    )


def _shorten(text: str, limit: int = 30) -> str:
    """틀린 자리가 길면 가운데를 줄여 보여 준다. 세는 데는 영향이 없다."""
    if len(text) <= limit:
        return text
    return f"{text[:limit * 2 // 3]}…{text[-(limit // 3):]}({len(text)}자)"


def format_report(report: Report, context: int = 8, max_errors: int = 50) -> list[str]:
    """사람이 읽을 수 있게 줄 목록으로 만든다."""
    lines = [
        f"정답 글자 수 : {len(report.ref)} (띄어쓰기, 문장부호 뺀 수)",
        f"인식 글자 수 : {len(report.hyp)}",
        f"바꿈 {report.substitutions} / 빠짐 {report.deletions} / 끼어듦 {report.insertions}"
        f"  = 모두 {report.total}",
        f"글자 오류율(CER) : {report.cer:.2%}",
    ]
    if not report.errors:
        lines.append("틀린 자리가 없습니다.")
        return lines

    shown = report.errors if max_errors <= 0 else report.errors[:max_errors]
    lines.append("")
    lines.append(f"틀린 자리 {len(report.errors)}군데" + ("" if len(shown) == len(report.errors) else f" (앞 {len(shown)}군데만 보임)"))
    for number, error in enumerate(shown, start=1):
        ref_before = report.ref[max(0, error.ref_at - context):error.ref_at]
        ref_after = report.ref[error.ref_at + len(error.ref_text):][:context]
        hyp_before = report.hyp[max(0, error.hyp_at - context):error.hyp_at]
        hyp_after = report.hyp[error.hyp_at + len(error.hyp_text):][:context]
        ref_text, hyp_text = _shorten(error.ref_text), _shorten(error.hyp_text)
        lines.append(f"{number:>3}) {error.kind}  '{ref_text}' -> '{hyp_text}'")
        lines.append(f"     정답: ...{ref_before}[{ref_text}]{ref_after}...")
        lines.append(f"     인식: ...{hyp_before}[{hyp_text}]{hyp_after}...")
    return lines
