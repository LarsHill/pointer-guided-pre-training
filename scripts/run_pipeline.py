import argparse
import datetime
import logging
import multiprocessing
import os

import yaml
from fluidml import Flow, TaskSpec, configure_logging

from llm import project_path
from llm.fluid_helper import MyLocalFileStore, TaskResource, get_balanced_devices
from llm.train import Mode, evaluate, train

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default=None,
        type=str,
        help="Mode to differentiate between training and validation (train) and training and testing (test)",
    )
    parser.add_argument(
        "--config",
        default="train_cfg.yml",
        type=str,
        help="config file name",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        type=str,
        help="Path to config",
    )
    parser.add_argument(
        "--cuda-ids",
        default=None,
        type=int,
        nargs="+",
        help="GPU ids, e.g. `--cuda-ids 0 1`",
    )
    parser.add_argument("--use-cuda", action="store_true", help="Use cuda.")
    parser.add_argument("--warm-start", action="store_true", help="Tries to warm start training.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of multiprocessing workers.",
    )
    parser.add_argument(
        "--force",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Task or tasks to force execute. '+' registers successor tasks also for force execution."
            "E.g. --force ModelTraining+"
        ),
    )
    parser.add_argument(
        "--cfg-expansion",
        type=str,
        default="product",
        choices=["product", "zip"],
        help="Method to expand config for grid search",
    )
    parser.add_argument("--log-to-tmux", action="store_true", help="Log to several tmux panes.")
    parser.add_argument("--project-name", type=str, default=None, help="Name of project.")
    parser.add_argument("--run-name", type=str, default=None, help="Name of run.")
    parser.add_argument("--debug", action="store_true", help="Activate debug mode.")

    return parser.parse_args()


def main():
    start = datetime.datetime.now()

    args = parse_args()

    run_name = "debug" if args.debug else args.run_name

    # load config from path if provided
    if args.config_path is not None:
        config_path = args.config_path
    else:
        config_path = os.path.join(project_path, "scripts", args.config)
    config = yaml.safe_load(open(config_path, "r"))
    if "_anchors" in config:
        del config["_anchors"]

    project_name = args.project_name if args.project_name is not None else config.get("project_name")

    # get base directory to store experiment results
    out_dir = config.pop("out_dir")
    if args.debug:
        out_dir = os.path.join(out_dir, "debug")

    # get run mode and add to train config
    mode = args.mode if args.mode is not None else config.pop("mode")
    if mode is None:
        mode = "train"
    config["train"]["mode"] = mode

    # configure logging
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    configure_logging(level="INFO")

    # Init all tasks
    training = TaskSpec(
        task=train,
        config=config["train"],
        additional_kwargs={"warm_start": args.warm_start, "debug": args.debug},
        expand=args.cfg_expansion,
    )

    # all tasks
    tasks = [
        training,
    ]

    if mode == Mode.TEST:
        evaluation = TaskSpec(
            task=evaluate,
            config=config["eval"],
            reduce=True,
        )
        tasks.append(evaluation)
        # dependencies between tasks
        evaluation.requires(training)

    # create local file storage used for versioning
    results_store = MyLocalFileStore(base_dir=out_dir)

    # get available and balanced devices (gpus or cpu)
    # list of resources is distributed among workers if num_workers > 1
    # e.g. to manage that each worker has dedicated access to specific gpus
    # else the first resource is assigned to all tasks
    devices = get_balanced_devices(count=args.num_workers, use_cuda=args.use_cuda, cuda_ids=args.cuda_ids)
    resources = [TaskResource(device=devices[i]) for i in range(args.num_workers)]

    # create flow (expanded task graph)
    flow = Flow(tasks=tasks, config_group_prefix="@", config_ignore_prefix="_")
    # run linearly without swarm if num_workers is set to 1
    # note resources are now assigned equally to all tasks (e.g. device info)
    # else run graph in parallel using multiprocessing
    # create list of resources which is distributed among workers
    # e.g. to manage that each worker has dedicated access to specific gpus
    flow.run(
        num_workers=args.num_workers,
        resources=resources,
        log_to_tmux=args.log_to_tmux,
        force=args.force,
        results_store=results_store,
        project_name=project_name,
        run_name=run_name,
    )

    end = datetime.datetime.now()
    logger.info(f"{end - start}")


if __name__ == "__main__":
    main()
