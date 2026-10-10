"""Train ECGDilatedModel against an explicitly selected same-seed new baseline."""

from multiseed_common import train_run, training_parser


def parse_args(argv=None):
    return training_parser("candidate", argv)


def main():
    train_run(parse_args(), "candidate")


if __name__ == "__main__":
    main()
