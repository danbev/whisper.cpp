#!/usr/bin/env python3
import argparse
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import List


DEFAULT_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"


def eprint(*args):
    print(*args, file=sys.stderr)


def format_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{amount:.1f} GB"


def path_is_inside(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def download_file(url: str, destination: Path) -> None:
    if destination.exists():
        eprint(f"using existing archive: {destination}")
        return

    eprint(f"downloading {url}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        total_header = response.headers.get("Content-Length")
        total = int(total_header) if total_header else None
        downloaded = 0

        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break

            output.write(chunk)
            downloaded += len(chunk)

            if total:
                percent = downloaded * 100.0 / total
                progress = f"{format_bytes(downloaded)} / {format_bytes(total)} ({percent:.1f}%)"
            else:
                progress = format_bytes(downloaded)

            print(f"\rdownloaded {progress}", end="", file=sys.stderr, flush=True)

    print(file=sys.stderr)
    eprint(f"downloaded archive: {destination}")


def extract_archive(archive: Path, work_dir: Path) -> None:
    extracted = work_dir / "LibriSpeech" / "test-clean"
    if extracted.exists():
        eprint(f"using existing dataset: {extracted}")
        return

    eprint(f"extracting {archive}")
    root = work_dir.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (work_dir / member.name).resolve()
            if not path_is_inside(target, root):
                raise RuntimeError(f"archive member escapes destination: {member.name}")
        try:
            tar.extractall(work_dir, filter="data")
        except TypeError:
            tar.extractall(work_dir)


def find_flac_files(dataset: Path) -> List[Path]:
    files: List[Path] = []
    for speaker_dir in sorted(path for path in dataset.iterdir() if path.is_dir()):
        for chapter_dir in sorted(path for path in speaker_dir.iterdir() if path.is_dir()):
            files.extend(sorted(chapter_dir.glob("*.flac")))
    return files


def hypothesis_path(audio: Path) -> Path:
    return Path(str(audio) + ".txt")


def run_parakeet_batch(args, files: List[Path]) -> int:
    file_list = args.work_dir / ".parakeet-files.txt"
    file_list.write_text("".join(f"{audio}\n" for audio in files), encoding="utf-8")

    command = [
        str(args.cli),
        "--output-txt",
        "--model",
        str(args.model),
        "--file-list",
        str(file_list),
    ]

    if args.no_prints:
        command.append("--no-prints")

    if args.threads is not None:
        command.extend(["--threads", str(args.threads)])

    command.extend(args.extra_args)

    try:
        return subprocess.run(command, cwd=args.repo_root).returncode
    finally:
        file_list.unlink(missing_ok=True)


def transcribe(args, files: List[Path]) -> None:
    pending = []
    for audio in files:
        hyp = hypothesis_path(audio)
        if args.force or not hyp.exists():
            pending.append(audio)

    eprint(f"audio files: {len(files)}")
    eprint(f"pending transcriptions: {len(pending)}")

    if not pending:
        return

    eprint(f"run: {len(pending)} file(s)")
    ret = run_parakeet_batch(args, pending)
    if ret != 0:
        raise RuntimeError(f"parakeet-cli failed with exit code {ret}")

    missing = [audio for audio in pending if not hypothesis_path(audio).exists()]
    if missing:
        first = missing[0]
        raise RuntimeError(f"missing hypothesis after transcription: {first}")


def score(args) -> int:
    command = [str(args.python), "eval.py"]
    completed = subprocess.run(
        command,
        cwd=args.work_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.stdout:
        print(completed.stdout, end="")

    if completed.returncode != 0:
        return completed.returncode

    if args.result_file:
        args.result_file.write_text(completed.stdout, encoding="utf-8")
        eprint(f"wrote result: {args.result_file}")

    return 0


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent

    parser = argparse.ArgumentParser(
        description="Run parakeet-cli on LibriSpeech test-clean and compute WER."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--work-dir", type=Path, default=script_dir)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=script_dir / "LibriSpeech" / "test-clean",
        help="Directory containing LibriSpeech .flac files.",
    )
    parser.add_argument(
        "--cli",
        type=Path,
        required=True,
        help="Path to parakeet-cli.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=repo_root / "models" / "ggml-parakeet-tdt-0.6b-v3.bin",
        help="Path to converted Parakeet ggml model.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--no-prints", action="store_true", help="Pass --no-prints to parakeet-cli.")
    parser.add_argument("--force", action="store_true", help="Regenerate existing .flac.txt files.")
    parser.add_argument("--download", action="store_true", help="Download and extract test-clean if needed.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--result-file", type=Path, default=None)
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments passed to parakeet-cli after '--'.",
    )

    args = parser.parse_args()

    if args.extra_args and args.extra_args[0] == "--":
        args.extra_args = args.extra_args[1:]

    args.repo_root = args.repo_root.resolve()
    args.work_dir = args.work_dir.resolve()
    args.dataset = args.dataset.resolve()
    args.cli = args.cli.resolve()
    args.model = args.model.resolve()
    if args.result_file is not None:
        args.result_file = args.result_file.resolve()

    return args


def main() -> int:
    args = parse_args()

    if args.download:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        archive = args.work_dir / Path(args.url).name
        download_file(args.url, archive)
        extract_archive(archive, args.work_dir)

    if not args.cli.exists():
        eprint(f"error: parakeet-cli not found: {args.cli}")
        return 2

    if not args.model.exists():
        eprint(f"error: model not found: {args.model}")
        return 2

    if not args.dataset.exists():
        eprint(f"error: dataset not found: {args.dataset}")
        eprint("hint: pass --download or extract test-clean under tests/librispeech-parakeet/LibriSpeech")
        return 2

    eprint(f"repo root: {args.repo_root}")
    eprint(f"work dir: {args.work_dir}")
    eprint(f"dataset: {args.dataset}")
    eprint(f"cli: {args.cli}")
    eprint(f"model: {args.model}")

    files = find_flac_files(args.dataset)
    if not files:
        eprint(f"error: no .flac files found under {args.dataset}")
        return 2

    try:
        transcribe(args, files)
    except RuntimeError as exc:
        eprint(f"error: {exc}")
        return 1

    return score(args)


if __name__ == "__main__":
    raise SystemExit(main())
