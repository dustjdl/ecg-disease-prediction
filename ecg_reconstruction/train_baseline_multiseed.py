"""Train ECGModel with an explicit seed, retaining the fixed baseline protocol."""

from multiseed_common import train_run, training_parser


def parse_args(argv=None):
    return training_parser("baseline", argv)


def main():
    train_run(parse_args(), "baseline")


if __name__ == "__main__":
    main()
