from __future__ import annotations

import argparse
from typing import Optional, List


def main() -> None:
    ap = argparse.ArgumentParser(prog="trading-bot")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("--config", required=True)
    p_ingest.add_argument("--days", type=int, default=None)
    p_ingest.add_argument("--tf", default=None)
    p_ingest.add_argument("--symbol", default=None)

    # build dataset
    p_ds = sub.add_parser("build-dataset")
    p_ds.add_argument("--config", required=True)
    p_ds.add_argument("--symbol", default=None)
    p_ds.add_argument("--force-features", action="store_true")

    # train
    p_train = sub.add_parser("train")
    p_train.add_argument("--config", required=True)

    # backtest
    p_bt = sub.add_parser("backtest")
    p_bt.add_argument("--config", required=True)

    # pipeline
    p_pipe = sub.add_parser("pipeline")
    p_pipe.add_argument("--ingest-config", required=True)
    p_pipe.add_argument("--dataset-config", required=True)
    p_pipe.add_argument("--train-config", required=True)
    p_pipe.add_argument("--backtest-config", default=None)

    p_pipe.add_argument("--symbol", default=None)
    p_pipe.add_argument("--days", type=int, default=None)
    p_pipe.add_argument("--tf", default=None)
    p_pipe.add_argument("--force-features", action="store_true")

    args = ap.parse_args()

    if args.cmd == "ingest":
        from src.cli import ingest as ingest_mod

        argv: List[str] = ["--config", args.config]
        if args.days is not None:
            argv += ["--days", str(args.days)]
        if args.tf is not None:
            argv += ["--tf", args.tf]
        if args.symbol is not None:
            argv += ["--symbol", args.symbol]
        ingest_mod.main(argv)
        return

    if args.cmd == "build-dataset":
        from src.cli import build_dataset as build_ds_mod

        argv = ["--config", args.config]
        if args.symbol is not None:
            argv += ["--symbol", args.symbol]
        if args.force_features:
            argv += ["--force-features"]
        build_ds_mod.main(argv)
        return

    if args.cmd == "train":
        from src.cli import train_model as train_mod

        train_mod.main(["--config", args.config])
        return

    if args.cmd == "backtest":
        from src.cli import backtest_model as bt_mod

        bt_mod.main(["--config", args.config])
        return

    if args.cmd == "pipeline":
        from src.cli import ingest as ingest_mod
        from src.cli import build_dataset as build_ds_mod
        from src.cli import train_model as train_mod

        ingest_argv = ["--config", args.ingest_config]
        if args.days is not None:
            ingest_argv += ["--days", str(args.days)]
        if args.tf is not None:
            ingest_argv += ["--tf", args.tf]
        if args.symbol is not None:
            ingest_argv += ["--symbol", args.symbol]
        ingest_mod.main(ingest_argv)

        ds_argv = ["--config", args.dataset_config]
        if args.symbol is not None:
            ds_argv += ["--symbol", args.symbol]
        if args.force_features:
            ds_argv += ["--force-features"]
        build_ds_mod.main(ds_argv)

        train_mod.main(["--config", args.train_config])

        if args.backtest_config:
            from src.cli import backtest_model as bt_mod

            bt_mod.main(["--config", args.backtest_config])
        return


if __name__ == "__main__":
    main()
