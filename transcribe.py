#!/usr/bin/env python3
"""음성 손보기 -> 한국어 받아쓰기 -> 정답 대조까지 한 번에 하는 스크립트.

보기:
    # 아무것도 손대지 않은 기준 실행
    python transcribe.py testvoice.ogg --no-resample

    # 16kHz 단일 채널 변환만 (기본값)
    python transcribe.py testvoice.ogg

    # 거기에 잡음 억제 하나만 더해서 비교
    python transcribe.py testvoice.ogg --denoise

    # 정답 파일과 대조해 글자 오류율까지
    python transcribe.py testvoice.ogg --ref testvoice_answer.txt

한 번에 하나만 바꿔 돌리고, 실행마다 runs.tsv 에 한 줄씩 쌓이는 설정과
결과 파일 이름을 견주면 어떤 단계가 도움이 되었는지 알 수 있다.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from faster_whisper import WhisperModel

import audio_prep
import cer

LOG_COLUMNS = [
    "실행시각", "음성파일", "모델", "언어", "고유명사", "16k모노", "잡음억제",
    "무음절삭", "크기평탄화", "beam", "VAD", "조각수", "정답파일", "CER", "결과파일",
]


def cuda_usable() -> bool:
    """GPU 가 있고, 거기에 필요한 라이브러리까지 실제로 불러와지는지 본다."""
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() <= 0:
            return False
    except Exception:
        return False

    # GPU 가 보여도 cuBLAS 가 없으면 받아쓰는 도중에 터진다. 미리 걸러 낸다.
    if sys.platform == "win32":
        import ctypes

        for dll in ("cublas64_12.dll", "cublas64_11.dll"):
            try:
                ctypes.WinDLL(dll)
                return True
            except OSError:
                continue
        return False
    return True


def pick_device(requested: str) -> tuple[str, str]:
    """사용할 장치와 연산 타입을 고른다. (device, compute_type)"""
    if requested == "auto":
        requested = "cuda" if cuda_usable() else "cpu"
    # GPU 면 float16, CPU 면 int8 이 속도/메모리 균형이 좋다.
    return requested, "float16" if requested == "cuda" else "int8"


def format_timestamp(seconds: float) -> str:
    """초를 HH:MM:SS.mmm 형태로 바꾼다."""
    ms = int(round(seconds * 1000))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def read_text(path: Path) -> str:
    """한글 문서가 utf-8 이 아닐 수도 있어 흔한 인코딩을 차례로 시도한다."""
    for encoding in ("utf-8", "utf-8-sig", "cp949"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"글자 인코딩을 알 수 없습니다: {path}")


def collect_terms(args: argparse.Namespace) -> list[str]:
    """자주 나오는 고유명사 목록을 모은다."""
    terms: list[str] = []
    if args.terms:
        terms.extend(part.strip() for part in args.terms.split(","))
    if args.terms_file:
        for raw in read_text(args.terms_file.expanduser()).splitlines():
            line = raw.split("#", 1)[0].strip()
            if line:
                terms.append(line)

    unique: dict[str, None] = {}
    for term in terms:
        if term:
            unique.setdefault(term, None)
    return list(unique)


def config_tag(args: argparse.Namespace, resample: bool) -> str:
    """켠 단계를 파일 이름에 넣을 짧은 표시로 만든다. R=16k모노 D=잡음 T=무음 N=크기."""
    switches = (("R", resample), ("D", args.denoise), ("T", args.trim_silence), ("N", args.normalize))
    return "".join(letter for letter, on in switches if on) or "raw"


def build_output_path(audio: Path, args: argparse.Namespace, resample: bool, outdir: Path) -> Path:
    """모델 이름, 켠 단계, 실행 시각이 드러나는 겹치지 않는 경로를 만든다."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{audio.stem}_{args.model}_{config_tag(args, resample)}_{stamp}"
    path = outdir / f"{base}.txt"
    counter = 2  # 같은 초에 두 번 돌린 경우까지 대비한다
    while path.exists():
        path = outdir / f"{base}_{counter}.txt"
        counter += 1
    return path


def append_log(path: Path, row: dict[str, object]) -> None:
    """실행마다 설정과 결과 파일 이름을 한 줄로 남긴다."""
    is_new = not path.exists()
    with path.open("a", encoding="utf-8") as fp:
        if is_new:
            fp.write("\t".join(LOG_COLUMNS) + "\n")
        fp.write("\t".join(str(row.get(column, "")) for column in LOG_COLUMNS) + "\n")


def transcribe(args, audio, out_path, device, compute_type, prepared, terms):
    """받아쓰면서 화면과 파일에 같이 남긴다. (조각 수, 받아쓴 글) 을 돌려준다."""
    model = WhisperModel(args.model, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        prepared.audio if prepared else str(audio),
        language=None if args.language == "auto" else args.language,
        beam_size=args.beam_size,
        vad_filter=not args.no_vad,
        hotwords=", ".join(terms) if terms else None,
    )

    print(f"인식된 언어: {info.language} (확신도 {info.language_probability:.2f})")
    print("-" * 78, flush=True)

    stages = ", ".join(prepared.stages) if prepared else "없음(원본 그대로)"
    spoken: list[str] = []
    with out_path.open("w", encoding="utf-8") as fp:
        fp.write(f"# 원본 파일  : {audio}\n")
        fp.write(f"# 모델       : {args.model} ({device}, {compute_type})\n")
        fp.write(f"# 언어       : {info.language} (확신도 {info.language_probability:.2f})\n")
        fp.write(f"# 손본 단계  : {stages}\n")
        if prepared and prepared.kept:
            fp.write(f"# 잘라낸 무음: {prepared.removed:.1f}초 / 원본 {prepared.original_duration:.1f}초\n")
        fp.write(f"# 고유명사   : {', '.join(terms) if terms else '없음'}\n")
        fp.write(f"# 받아쓴 시각: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        fp.write("# 시각은 모두 원본 음성 기준입니다.\n")
        fp.write("-" * 78 + "\n")

        # 조각이 나오는 대로 바로 적으므로 중간에 멈춰도 그때까지는 남는다.
        for segment in segments:
            # 무음을 잘라냈어도 원본에서 되짚을 수 있게 시각을 되돌린다.
            start = prepared.to_original_time(segment.start) if prepared else segment.start
            end = prepared.to_original_time(segment.end) if prepared else segment.end
            text = segment.text.strip()
            line = f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}"
            print(line, flush=True)
            fp.write(line + "\n")
            fp.flush()
            spoken.append(text)

    return len(spoken), " ".join(spoken)


def compare_with_reference(args, out_path: Path, spoken_text: str) -> str:
    """정답 파일과 대조해 오류율과 틀린 자리를 화면과 파일에 남긴다."""
    report = cer.compare(read_text(args.ref.expanduser()), spoken_text)
    lines = cer.format_report(report, context=args.context, max_errors=args.max_errors)

    print()
    print("=" * 78)
    print(f"정답 파일과 대조: {args.ref}")
    print("=" * 78)
    for line in lines:
        print(line)

    with out_path.open("a", encoding="utf-8") as fp:
        fp.write("\n" + "=" * 78 + "\n")
        fp.write(f"# 정답 파일: {args.ref}\n")
        fp.write("# 띄어쓰기와 문장부호는 빼고 셌고, 바꾸기는 한 번으로 셌습니다.\n")
        fp.write("=" * 78 + "\n")
        fp.write("\n".join(lines) + "\n")

    return f"{report.cer:.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="음성을 손본 뒤 한국어로 받아쓰고, 정답과 대조해 글자 오류율을 낸다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio", type=Path, help="받아쓸 음성 파일 경로")

    stage = parser.add_argument_group("음성 손보기 (각각 켜고 끌 수 있음)")
    stage.add_argument("--resample", action=argparse.BooleanOptionalAction, default=True,
                       help="16kHz 단일 채널로 바꾼다")
    stage.add_argument("--denoise", action=argparse.BooleanOptionalAction, default=False,
                       help="잡음을 억제한다")
    stage.add_argument("--trim-silence", action=argparse.BooleanOptionalAction, default=False,
                       help="긴 무음을 잘라낸다")
    stage.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False,
                       help="소리 크기를 고르게 편다")
    stage.add_argument("--denoise-strength", type=float, default=1.5, help="잡음을 깎는 세기")
    stage.add_argument("--max-silence", type=float, default=0.8, help="이보다 긴 무음만 잘라낸다(초)")
    stage.add_argument("--keep-silence", type=float, default=0.2, help="말 앞뒤로 남길 여유(초)")
    stage.add_argument("--silence-db", type=float, default=None, help="무음 기준(dB). 비우면 자동")
    stage.add_argument("--target-dbfs", type=float, default=-20.0, help="맞출 소리 크기(dBFS)")

    stt = parser.add_argument_group("받아쓰기")
    stt.add_argument("--model", default="small", help="모델 크기 (tiny/base/small/medium/large-v3)")
    stt.add_argument("--language", default="ko", help="언어. auto 면 자동으로 알아낸다")
    stt.add_argument("--terms", default=None, help="자주 나오는 고유명사, 쉼표로 구분")
    stt.add_argument("--terms-file", type=Path, default=None, help="고유명사를 한 줄에 하나씩 적은 파일")
    stt.add_argument("--beam-size", type=int, default=5, help="빔 서치 크기")
    stt.add_argument("--no-vad", action="store_true", help="무음 구간 걸러내기(VAD)를 끈다")
    stt.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="연산 장치")

    check = parser.add_argument_group("정답 대조")
    check.add_argument("--ref", type=Path, default=None, help="손으로 받아 적은 정답 파일")
    check.add_argument("--context", type=int, default=8, help="틀린 자리 앞뒤로 보여 줄 글자 수")
    check.add_argument("--max-errors", type=int, default=50, help="보여 줄 틀린 자리 수. 0 이면 전부")

    out = parser.add_argument_group("결과")
    out.add_argument("--outdir", type=Path, default=None, help="결과 폴더. 비우면 음성 파일 옆")
    out.add_argument("--log", type=Path, default=None, help="실행 기록 파일. 비우면 결과 폴더의 runs.tsv")
    return parser.parse_args()


def main() -> int:
    for stream in (sys.stdout, sys.stderr):  # 한글이 콘솔에서 깨지지 않게 한다
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    args = parse_args()

    audio = args.audio.expanduser()
    if not audio.is_file():
        print(f"[오류] 음성 파일을 찾을 수 없습니다: {audio}", file=sys.stderr)
        return 1
    if args.ref and not args.ref.expanduser().is_file():
        print(f"[오류] 정답 파일을 찾을 수 없습니다: {args.ref}", file=sys.stderr)
        return 1

    # 나머지 손보기 단계는 16kHz 단일 채널 신호 위에서 돈다.
    needs_signal = args.denoise or args.trim_silence or args.normalize
    resample = args.resample or needs_signal
    if needs_signal and not args.resample:
        print("[알림] 다른 손보기 단계를 켜서 16kHz 단일 채널 변환도 함께 켭니다.")

    outdir = (args.outdir.expanduser() if args.outdir else audio.parent).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = build_output_path(audio, args, resample, outdir)
    log_path = args.log.expanduser() if args.log else outdir / "runs.tsv"

    device, compute_type = pick_device(args.device)
    terms = collect_terms(args)

    print(f"음성 파일 : {audio}")
    print(f"모델      : {args.model} ({device}, {compute_type}), 언어 {args.language}")
    print(f"손보기    : 16k모노={resample} 잡음억제={args.denoise} "
          f"무음절삭={args.trim_silence} 크기평탄화={args.normalize}")
    print(f"고유명사  : {', '.join(terms) if terms else '없음'}")
    print(f"결과 파일 : {out_path}")
    print()

    prepared = None
    if resample:
        prepared = audio_prep.prepare(
            str(audio),
            denoise=args.denoise,
            trim_silence=args.trim_silence,
            normalize=args.normalize,
            denoise_strength=args.denoise_strength,
            max_silence=args.max_silence,
            keep_silence=args.keep_silence,
            silence_db=args.silence_db,
            target_dbfs=args.target_dbfs,
        )
        print(f"손본 결과 : {prepared.original_duration:.1f}초 -> {prepared.duration:.1f}초"
              f" ({', '.join(prepared.stages)})")
        if prepared.silence_threshold_db is not None:
            print(f"무음 기준 : {prepared.silence_threshold_db:.1f} dB")

    print("모델을 불러오는 중입니다. 처음이면 내려받느라 시간이 걸립니다...\n", flush=True)
    try:
        count, spoken_text = transcribe(args, audio, out_path, device, compute_type, prepared, terms)
    except RuntimeError as err:
        # 장치를 직접 고른 게 아닐 때만, GPU 가 말을 안 들으면 CPU 로 한 번 더 해 본다.
        if args.device != "auto" or device != "cuda":
            raise
        print(f"\n[알림] GPU 로 받아쓰지 못했습니다: {err}")
        print("[알림] CPU 로 다시 시도합니다.\n", flush=True)
        out_path.unlink(missing_ok=True)
        device, compute_type = "cpu", "int8"
        count, spoken_text = transcribe(args, audio, out_path, device, compute_type, prepared, terms)

    print("-" * 78)
    print(f"조각 {count}개를 받아썼습니다." if count
          else "받아쓸 말소리를 찾지 못했습니다. --no-vad 로 다시 해 보세요.")

    cer_value = compare_with_reference(args, out_path, spoken_text) if args.ref else ""

    append_log(log_path, {
        "실행시각": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "음성파일": audio.name,
        "모델": args.model,
        "언어": args.language,
        "고유명사": ";".join(terms),
        "16k모노": int(resample),
        "잡음억제": int(args.denoise),
        "무음절삭": int(args.trim_silence),
        "크기평탄화": int(args.normalize),
        "beam": args.beam_size,
        "VAD": int(not args.no_vad),
        "조각수": count,
        "정답파일": args.ref.name if args.ref else "",
        "CER": cer_value,
        "결과파일": out_path.name,
    })

    print()
    print(f"결과를 저장했습니다: {out_path}")
    print(f"실행 기록을 남겼습니다: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
