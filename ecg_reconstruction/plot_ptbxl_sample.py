"""PTB-XL 12-lead ECG 샘플을 내려받아 PNG 그래프로 저장한다.

기본 샘플은 PTB-XL 데이터셋의 00001_lr(100 Hz)이다. 이 파일은 약 10초,
12개 lead로 구성되어 있어 처음 ECG 파형을 확인하기에 적합하다.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import wfdb


# PTB-XL의 저해상도(100 Hz) 첫 번째 레코드 경로이다.
DATASET_VERSION = "1.0.3"
RECORD_RELATIVE_PATH = Path("records100") / "00000" / "00001_lr"
PHYSIONET_BASE_URL = f"https://physionet.org/files/ptb-xl/{DATASET_VERSION}"


def download_sample(data_directory: Path) -> Path:
    """샘플의 .hea/.dat 파일이 없을 때만 내려받고 레코드 기본 경로를 반환한다."""
    record_base = data_directory / RECORD_RELATIVE_PATH
    record_base.parent.mkdir(parents=True, exist_ok=True)

    for extension in (".hea", ".dat"):
        destination = record_base.with_suffix(extension)
        if destination.exists():
            print(f"이미 있음: {destination}")
            continue

        source_url = f"{PHYSIONET_BASE_URL}/{RECORD_RELATIVE_PATH.as_posix()}{extension}"
        print(f"다운로드 중: {source_url}")
        try:
            urlretrieve(source_url, destination)
        except URLError as error:
            raise RuntimeError(
                "PTB-XL 샘플을 내려받지 못했습니다. 인터넷 연결을 확인한 뒤 다시 실행하세요."
            ) from error

    return record_base


def load_ecg(record_base: Path):
    """WFDB 레코드를 읽고, 12개의 ECG 신호가 맞는지 확인한다."""
    record = wfdb.rdrecord(str(record_base))
    if record.p_signal is None:
        raise ValueError("읽어 온 레코드에 디지털 ECG 신호가 없습니다.")
    if record.p_signal.shape[1] != 12:
        raise ValueError(f"12개 lead가 필요하지만 {record.p_signal.shape[1]}개를 받았습니다.")
    return record


def save_12_lead_plot(record, output_path: Path) -> None:
    """각 lead를 한 행씩 배치한 12행 그래프를 PNG로 저장한다."""
    signal = record.p_signal
    sampling_rate = record.fs
    time_seconds = range(len(signal))
    time_seconds = [sample / sampling_rate for sample in time_seconds]

    figure, axes = plt.subplots(12, 1, figsize=(15, 18), sharex=True)
    figure.suptitle("PTB-XL sample 00001_lr: 12-lead ECG", fontsize=16)

    for lead_index, axis in enumerate(axes):
        axis.plot(time_seconds, signal[:, lead_index], color="tab:blue", linewidth=0.8)
        axis.set_ylabel(f"{record.sig_name[lead_index]}\n(mV)", rotation=0, labelpad=28, va="center")
        axis.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (seconds)")
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="PTB-XL 12-lead ECG sample plotter")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data") / "ptb-xl",
        help="PTB-XL sample files will be stored here (default: data/ptb-xl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output") / "ptbxl_00001_12lead.png",
        help="PNG output path (default: output/ptbxl_00001_12lead.png)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing PNG output file.",
    )
    arguments = parser.parse_args()

    if arguments.output.exists() and not arguments.overwrite:
        raise FileExistsError(
            f"Output already exists: {arguments.output}. "
            "Check the file, then use --overwrite only if replacement is intended."
        )

    record_base = download_sample(arguments.data_dir)
    ecg_record = load_ecg(record_base)
    save_12_lead_plot(ecg_record, arguments.output)

    print(f"Lead names: {', '.join(ecg_record.sig_name)}")
    print(f"Sampling rate: {ecg_record.fs} Hz")
    print(f"PNG saved: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
