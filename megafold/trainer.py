from __future__ import annotations

import glob
import os
import pprint
import traceback
from contextlib import contextmanager
from functools import partial
from importlib.metadata import version
from pathlib import Path

import torch
import torchinfo
from adam_atan2_pytorch.foreach import AdamAtan2
from beartype.typing import Any, Callable, List, Literal, Set
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.ops.lion import DeepSpeedCPULion
from ema_pytorch import EMA
from lightning import Fabric
from lightning.fabric.strategies import DDPStrategy, DeepSpeedStrategy
from lightning.pytorch.utilities.memory import garbage_collection_cuda
from lion_pytorch.foreach import Lion
from pydantic import BaseModel
from torch.nn import Module
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader as OrigDataLoader
from torch.utils.data import Dataset, Sampler
from torchmetrics.aggregation import MeanMetric
from tqdm import tqdm
from wrapt_timeout_decorator import timeout

from megafold.data import mmcif_writing
from megafold.inputs import (
    BatchedAtomInput,
    collate_inputs_to_batched_atom_input,
    compose_calls,
)
if os.environ.get("AF3_OPTIMIZATIONS_MODE") == "baseline": 
    from megafold.baseline import ComputeConfidenceScore, ComputeModelSelectionScore, Sample
elif os.environ.get("AF3_OPTIMIZATIONS_MODE") == "layernormlinear":
    from megafold.megafold_layernormlinear import ComputeConfidenceScore, ComputeModelSelectionScore, Sample
else:
    from megafold.model.megafold import ComputeConfidenceScore, ComputeModelSelectionScore, Sample
    
from megafold.tensor_typing import package_available, should_typecheck, typecheck
from megafold.utils.model_utils import at_most_one_of, divisible_by
from megafold.utils.trainer_utils import (
    CycleIterator,
    capture_hparams,
    choose_logger,
    generate_id,
    get_default_supported_precision,
    get_logger_experiment_id,
    parse_devices,
    parse_dtype,
)
from megafold.utils.utils import default, exists, not_exists
import time 
from deepspeed.utils.timer import SynchronizedWallClockTimer



# constants

PHASES = Literal["train", "val", "test"]
FORWARD_MAX_SECONDS_PER_INPUT = 120
SAMPLING_MAX_SECONDS_PER_INPUT = 300


# helpers


@contextmanager
def to_device_and_back(module: Module, device: torch.device):
    """Move module to device and back after context."""
    orig_device = next(module.parameters()).device
    need_move_device = orig_device != device

    if need_move_device:
        module.to(device)

    yield

    if need_move_device:
        module.to(orig_device)


# dataloader and collation fn


@typecheck
def DataLoader(
    *args,
    atoms_per_window: int | None = None,
    map_input_fn: Callable | None = None,
    transform_to_atom_inputs: bool = True,
    **kwargs,
) -> OrigDataLoader:
    """DataLoader with collation function."""
    collate_fn = partial(
        collate_inputs_to_batched_atom_input,
        atoms_per_window=atoms_per_window,
        transform_to_atom_inputs=transform_to_atom_inputs,
    )

    if exists(map_input_fn):
        collate_fn = partial(collate_fn, map_input_fn=map_input_fn)

    return OrigDataLoader(*args, collate_fn=collate_fn, **kwargs)


# default scheduler used in paper w/ warmup


def default_lambda_lr_fn(
    steps: int,
    warmup_steps: int = 3000,
    decay_every_n_steps: int = 5e4,
    decay_rate: float = 0.95,
    disabled: bool = False,
) -> float:
    """Default lambda function for scheduler."""
    if disabled:
        return 1.0

    # warmup for `warmup_steps` steps

    if steps < warmup_steps:
        return steps / warmup_steps

    # decay `decay_rate` every `decay_every_n_steps` steps

    steps -= warmup_steps
    return decay_rate ** (steps / decay_every_n_steps)


# main class


class Trainer:
    """Section 5.4."""

    @typecheck
    def __init__(
        self,
        model: BaseModel,  # NOTE: must be a `MegaFoldConfig` instance
        *,
        dataset: Dataset,
        num_train_steps: int,
        global_batch_size: int,
        num_valid_steps: int | None = None,
        num_test_steps: int | None = None,
        devices: int | str = "auto",
        num_nodes: int = 1,
        seed: int = 42,
        grad_accum_every: int = 1,
        clear_cuda_cache_every: int = 1,
        confidence_head_interval: int = 10,
        offload_optimizer: bool = False,
        allgather_bucket_size: int = 200_000_000,
        reduce_bucket_size: int = 200_000_000,
        samples_on_cpu: bool = False,
        map_dataset_input_fn: Callable | None = None,
        valid_dataset: Dataset | None = None,
        valid_every: int = 1000,
        test_dataset: Dataset | None = None,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        ema_decay: float = 0.999,
        lr: float = 1e-3,
        default_adam_kwargs: dict = dict(
            betas=(0.9, 0.95),
            eps=1e-8,
        ),
        clip_grad_norm: float = 10.0,
        default_lambda_lr: Callable = default_lambda_lr_fn,
        train_sampler: Sampler | None = None,
        fabric: Fabric | None = None,
        profile: bool = False,
        profiler_kwargs: dict = dict(),
        logger_name: Literal["wandb", "tensorboard", "csv"] | None = None,
        logger_kwargs: dict = dict(),
        train_log_interval: int = 1,
        accelerator: Literal["cpu", "gpu", "tpu", "mps", "auto"] = "auto",
        strategy: Literal["auto", "ddp", "deepspeed"] = "ddp",
        strategy_stage: int = 0,
        checkpoint_prefix: str = "megafold.ckpt.",
        checkpoint_every: int = 25,
        checkpoint_folder: str = "./checkpoints",
        overwrite_checkpoints: bool = False,
        fabric_kwargs: dict = dict(),
        precision: Literal["bf16-true", "bf16-mixed", "32-true", None] = None,
        use_ema: bool = True,
        ema_kwargs: dict = dict(use_foreach=True),
        ema_on_cpu: bool = False,
        use_adam_atan2: bool = False,
        use_lion: bool = False,
        use_torch_compile: bool = False,
        is_fine_tuning: bool = False,
        summarize_model: bool = True,
        num_samples_per_example: int = 5,
        visualize_train_samples_every_n_steps: int = 0,
        visualize_valid_samples_every_n_steps: int = 0,
        visualize_test_samples_every_n_steps: int = 0,
        watch_model: Literal["gradients", "parameters", "all"] | None = None,
        watch_model_freq: int = 1,
        dl_kwargs: dict = dict(),
    ):
        super().__init__()

        # precision

        self.dtype = parse_dtype(precision or get_default_supported_precision(training=True))

        # strategy

        devices = parse_devices(devices)
        self.devices = devices

        if strategy == "ddp":
            # if necessary, address potential DDP activation checkpointing issue: https://discuss.pytorch.org/t/ddp-and-gradient-checkpointing/132244
            strategy = DDPStrategy(find_unused_parameters=False, static_graph=False)
        elif strategy == "deepspeed":
            ds_config = DeepSpeedStrategy(
                zero_optimization=False,
                stage=strategy_stage,
                offload_optimizer=offload_optimizer,
                allgather_bucket_size=allgather_bucket_size,
                reduce_bucket_size=reduce_bucket_size,
            ).config

            # override certain default config values
            ds_config["gradient_clipping"] = clip_grad_norm
            ds_config["train_micro_batch_size_per_gpu"] = 1
            ds_config["gradient_accumulation_steps"] = grad_accum_every

            strategy = DeepSpeedStrategy(config=ds_config)
        elif strategy != "auto":
            raise ValueError(f"Unknown strategy: {strategy}")

        self.using_deepspeed_strategy = isinstance(strategy, DeepSpeedStrategy)

        # logger

        loggers = None

        if exists(logger_name):
            loggers = [choose_logger(logger_name, **logger_kwargs)]

        self.train_log_interval = train_log_interval

        # fabric

        if not_exists(fabric):
            fabric = Fabric(
                accelerator=accelerator,
                devices=devices,
                num_nodes=num_nodes,
                strategy=strategy,
                # NOTE: we use 32-bit precision by default to avoid weight casting issues with DeepSpeed
                precision=precision or "32-true",
                loggers=loggers,
                **fabric_kwargs,
            )

        self.fabric = fabric
        self.fabric.launch()

        # dataset arguments

        dataset_ = dataset.datasets[0] if isinstance(dataset, ConcatDataset) else dataset
        cropping_config = getattr(dataset_, "cropping_config", {})
        self.crop_size = cropping_config.get("n_res", int(1e6))

        # hyperparameters

        hparams = capture_hparams()

        self.fabric.print(pprint.pformat(hparams))
        if logger_name in ("tensorboard", "wandb"):
            self.fabric.logger.log_hyperparams(hparams)

        # checkpointing logic

        self.checkpoint_prefix = checkpoint_prefix
        self.checkpoint_every = checkpoint_every
        self.overwrite_checkpoints = overwrite_checkpoints
        self.checkpoint_folder = Path(checkpoint_folder)

        self.checkpoint_folder.mkdir(exist_ok=True, parents=True)
        assert (
            self.checkpoint_folder.is_dir()
        ), f"Checkpoint folder {self.checkpoint_folder} does not exist."

        # random seed

        latest_step = self.get_latest_step_from_checkpoint_folder()

        # NOTE: we use same seed for every process to init model the same way;
        # we also add the latest step to the seed to ensure that the dataloaders
        # are initialized uniquely if we are resuming training from a checkpoint,
        # e.g., since the PDBDataset is a map-style dataset that internally samples
        # PDB IDs in an iterable (state-less) manner using the WeightedPDBSampler;
        # this is designed as such to ensure that map-style distillation datasets
        # are directly compatible with the PDBDataset via simple concatenation.
        self.print(f"Seeding everything with seed {seed + latest_step}.")
        self.fabric.seed_everything(seed + latest_step)

        # PAE-specific loss adjustment

        if latest_step < 5000:
            # NOTE: we do this to prevent the PAE weights
            # (which importantly are used for sample ranking)
            # from getting stuck in local minima early in training
            # when the model is poor at denoising structures
            model.train_pae = False

        # efficient model instantiation

        with self.fabric.init_module():
            model = model.create_instance()  # NOTE: parameters are placed on the meta-device

        # exponential moving average (EMA)

        self.ema_model = None
        self.has_ema = use_ema

        if self.has_ema:
            self.ema_model = EMA(
                model,
                beta=ema_decay,
                update_every=checkpoint_every,
                inv_gamma=1.0,
                power=1.0,
                include_online_model=False,
                allow_different_devices=True,
                coerce_dtype=True,
                **ema_kwargs,
            )

            self.ema_device = "cpu" if ema_on_cpu else self.device
            self.ema_model.to(self.ema_device)

        # maybe torch compile

        if use_torch_compile:
            assert (
                not should_typecheck
            ), "Does not work well with jaxtyping + beartype, please invoke your training script with the environment flag `TYPECHECK=False` - ex. `TYPECHECK=False python train_megafold.py`"
            model = torch.compile(model)

        # reseed everything (since for some reason model initialization resets `torch.initial_seed()`)

        self.fabric.seed_everything(seed + latest_step)

        # if map dataset function given, curry into DataLoader

        DataLoader_ = partial(DataLoader, atoms_per_window=model.atoms_per_window)

        if exists(map_dataset_input_fn):
            DataLoader_ = partial(DataLoader_, map_input_fn=map_dataset_input_fn)

        # maybe distillation dataset training

        train_dl_kwargs = dict()

        if exists(train_sampler):
            train_dl_kwargs.update(sampler=train_sampler)
        else:
            train_dl_kwargs.update(shuffle=True, drop_last=True)

        # train dataloader

        self.global_batch_size = global_batch_size
        self.num_nodes = num_nodes

        self.dataloader = DataLoader_(
            dataset, batch_size=self.batch_size(), **dl_kwargs, **train_dl_kwargs
        )
        dataloaders = [self.dataloader]

        # validation dataloader on the EMA model

        self.valid_every = valid_every

        self.needs_valid = exists(valid_dataset)

        if self.needs_valid:
            self.valid_dataloader = DataLoader_(
                valid_dataset, batch_size=self.batch_size(), **dl_kwargs
            )
            dataloaders.append(self.valid_dataloader)

        # testing dataloader on EMA model

        self.needs_test = exists(test_dataset)

        if self.needs_test:
            self.test_dataloader = DataLoader_(
                test_dataset, batch_size=self.batch_size(), **dl_kwargs
            )
            dataloaders.append(self.test_dataloader)

        # training, validation, and test steps

        self.num_train_steps = num_train_steps
        self.num_valid_steps = num_valid_steps
        self.num_test_steps = num_test_steps

        # optimizer

        if not_exists(optimizer):
            optimizer_klass = (
                partial(DeepSpeedCPUAdam, adamw_mode=False)
                if self.using_deepspeed_strategy and offload_optimizer
                else Adam
            )

            assert at_most_one_of(use_adam_atan2, use_lion)

            if use_adam_atan2:
                default_adam_kwargs.pop("eps", None)
                optimizer_klass = AdamAtan2
                assert not (self.using_deepspeed_strategy and offload_optimizer), (
                    "AdamAtan2 is not supported with DeepSpeed optimizer offloading. "
                    "Please set `use_adam_atan2=False` or `offload_optimizer=False`."
                )
            elif use_lion:
                default_adam_kwargs.pop("eps", None)
                optimizer_klass = (
                    DeepSpeedCPULion
                    if self.using_deepspeed_strategy and offload_optimizer
                    else Lion
                )

            optimizer = optimizer_klass(model.parameters(), lr=lr, **default_adam_kwargs)

        elif (
            self.using_deepspeed_strategy
            and offload_optimizer
            and not (
                isinstance(optimizer, DeepSpeedCPUAdam) or isinstance(optimizer, DeepSpeedCPULion)
            )
        ):
            raise ValueError(
                "When using DeepSpeed optimizer offloading, the optimizer must be an instance of DeepSpeedCPUAdam or DeepSpeedCPULion."
            )

        # scheduler

        if not_exists(scheduler):
            scheduler = LambdaLR(optimizer, lr_lambda=default_lambda_lr)

        # fabric setup for model and optimizer
        #print("\n\n\nNumber of trainable params:",  sum(p.numel() for p in model.parameters() if p.requires_grad))
        #print("\n\n\nNumber of params:",  sum(p.numel() for p in model.parameters()))
        #print("\n\n\n")
        #for name, p in model.named_parameters():
         #   print(name)

        model, optimizer = self.fabric.setup(model, optimizer)

        self.model, self.model_optimizer, self.scheduler = model, optimizer, scheduler

        if self.is_main and summarize_model:
            torchinfo.summary(model)

        # dataloaders

        dataloaders = self.fabric.setup_dataloaders(*dataloaders)

        self.dataloader = dataloaders # [0]

        if self.needs_valid:
            self.valid_dataloader = dataloaders[1]

        if self.needs_test:
            self.test_dataloader = dataloaders[-1]

        # maximum norm gradient clipping

        self.clip_grad_norm = clip_grad_norm

        # gradient accumulation

        self.grad_accum_every = grad_accum_every

        # CUDA memory clearing

        self.clear_cuda_cache_every = clear_cuda_cache_every

        # steps

        self.steps = 0

        # confidence head interval

        self.confidence_head_interval = confidence_head_interval

        # path caching for the last loaded model, if any

        self.train_id = get_logger_experiment_id(self.fabric.loggers)

        self.last_loaded_train_id = None
        self.model_loaded_from_path: Path | None = None

        # model selection

        self.is_fine_tuning = is_fine_tuning
        self.num_samples_per_example = num_samples_per_example

        self.compute_model_selection_score = ComputeModelSelectionScore(
            is_fine_tuning=is_fine_tuning
        )

        self.best_model_selection_step = -1
        self.best_model_selection_score = -float("inf")
        self.best_top_ranked_lddt = -float("inf")

        self.samples_on_cpu = samples_on_cpu

        # visualization parameters

        self.visualize_train_samples_every_n_steps = visualize_train_samples_every_n_steps
        self.visualize_valid_samples_every_n_steps = visualize_valid_samples_every_n_steps
        self.visualize_test_samples_every_n_steps = visualize_test_samples_every_n_steps

        if logger_name == "wandb" and exists(watch_model):
            assert package_available(
                "wandb"
            ), "Please install and use the `wandb` package to log model gradients/parameters."

            self.fabric.logger.experiment.watch(model, log=watch_model, log_freq=watch_model_freq)

        # profiler

        self.profile = profile

        if self.profile:
            assert "log_dir" in profiler_kwargs, "Please provide a `log_dir` for the profiler."

            self.profiler_log_dir = profiler_kwargs["log_dir"]

            self.profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.profiler_log_dir),
                profile_memory=True,
                with_stack=True,
            )

    @property
    def device(self) -> torch.device:
        """Get device."""
        return self.fabric.device

    @property
    def is_main(self) -> bool:
        """Check if main rank."""
        return self.fabric.global_rank == 0

    def generate_train_id(self):
        """Generate a unique training id."""
        if exists(self.train_id):
            return

        self.train_id = generate_id()

    @property
    def train_id_with_prev(self) -> str:
        """Get train id with previous train id."""
        if not_exists(self.last_loaded_train_id):
            return self.train_id

        ckpt_num = str(self.model_loaded_from_path).split(".")[-2]

        return f"{self.last_loaded_train_id}.{ckpt_num}-{self.train_id}"

    # saving and loading

    def save_checkpoint(self):
        """Save checkpoint."""
        assert exists(self.train_id_with_prev), "Train ID not generated."

        # formulate checkpoint path and save

        os.makedirs(self.checkpoint_folder, exist_ok=True)

        checkpoint_path = (
            self.checkpoint_folder
            / f"({self.train_id_with_prev})_{self.checkpoint_prefix}{self.steps}.pt"
        )

        self.save(checkpoint_path, overwrite=self.overwrite_checkpoints)

    def save(self, path: str | Path, overwrite=False, prefix: str | None = None):
        """Save model and optimizer states."""
        self.wait()

        if isinstance(path, str):
            path = Path(path)

        assert not path.is_dir() and (overwrite or not path.exists()), (
            f"Checkpoint file {path} already exists. "
            "Please set `overwrite=True` to overwrite the file."
        )

        path.parent.mkdir(exist_ok=True, parents=True)

        package = dict(
            version=self.model.state_dict_with_init_args["version"],
            init_args_and_kwargs=self.model.state_dict_with_init_args["init_args_and_kwargs"],
            model=self.model,
            ema_model=self.ema_model.state_dict() if self.has_ema else None,
            model_optimizer=self.model_optimizer,
            scheduler=self.scheduler,
            steps=self.steps,
            id=self.train_id,
            best_model_selection_step=self.best_model_selection_step,
            best_model_selection_score=self.best_model_selection_score,
            best_top_ranked_lddt=self.best_top_ranked_lddt,
        )

        self.print(f"Saving checkpoint to {str(path)}")
        self.fabric.save(path, package)

        self.wait()

    def get_latest_step_from_checkpoint_folder(
        self, prefix=None, excluded_prefixes: Set[str] | None = {"collated_"}
    ) -> int:
        """Get latest step from checkpoint folder."""
        path = self.checkpoint_folder

        if isinstance(path, str):
            path = Path(path)

        assert path.is_dir(), f"Checkpoint folder {path} does not exist."

        prefix = default(prefix, self.checkpoint_prefix)

        model_paths = [
            p
            for p in path.glob(f"**/*_{prefix}*.pt")
            if not any(p.name.startswith(e) for e in excluded_prefixes)
        ]

        if not model_paths:
            self.print(f"WARNING: No files found in directory {path}. Skipping seed loading.")
            return 0

        model_paths = sorted(model_paths, key=lambda p: int(str(p).split(".")[-2]))
        latest_step = int(str(model_paths[-1]).split(".")[-2])

        return latest_step

    def load_from_checkpoint_folder(self, **kwargs):
        """Load from checkpoint folder."""
        self.load(path=self.checkpoint_folder, **kwargs)

    def load(
        self,
        path: str | Path,
        strict=True,
        prefix=None,
        only_model=False,
        reset_steps=False,
        load_best_model=False,
        excluded_prefixes: Set[str] | None = {"collated_"},
    ):
        """Load model and optimizer states."""
        if isinstance(path, str):
            path = Path(path)

        if not path.exists():
            self.print(
                f"WARNING: {str(path)} cannot be found. Skipping checkpoint loading from this folder."
            )
            return

        # if the path is a directory, then automatically load latest checkpoint

        if path.is_dir():
            prefix = default(prefix, self.checkpoint_prefix)

            model_paths = [
                p
                for p in path.glob(f"**/*_{prefix}*.pt")
                if not any(p.name.startswith(e) for e in excluded_prefixes)
            ]

            if not model_paths:
                self.print(
                    f"WARNING: No files found in directory {path}. Skipping checkpoint loading."
                )
                return

            if load_best_model:
                paths = [
                    p
                    for p in model_paths
                    if int(str(p).split(".")[-2]) == self.best_model_selection_step
                ]
                assert (
                    paths
                ), f"No best model at step {self.best_model_selection_step} with model selection score {self.best_model_selection_score:.6f} found at {path}."

                path = paths[0]
                self.print(
                    f"Best model found at step {self.best_model_selection_step} with model selection score {self.best_model_selection_score:.6f}."
                )
            else:
                model_paths = sorted(model_paths, key=lambda p: int(str(p).split(".")[-2]))
                path = model_paths[-1]

        # load model from path

        package = dict(
            version=self.model.state_dict_with_init_args["version"],
            init_args_and_kwargs=self.model.state_dict_with_init_args["init_args_and_kwargs"],
            model=self.model,
            ema_model=self.ema_model.state_dict() if self.has_ema else None,
            model_optimizer=self.model_optimizer,
            scheduler=self.scheduler,
            steps=self.steps,
            id=self.train_id,
            best_model_selection_step=self.best_model_selection_step,
            best_model_selection_score=self.best_model_selection_score,
            best_top_ranked_lddt=self.best_top_ranked_lddt,
        )

        self.print(f"Loading checkpoint from {path}")
        self.fabric.load(path, package, strict=strict)

        # load EMA model weights

        if self.has_ema:
            # NOTE: `strict=False` to allow loading of EMA model weights even if PLM/NLM model weights are not present
            self.ema_model.load_state_dict(package["ema_model"], strict=False)

        # ensure that the model is loaded from the same version

        self.model._version = package["version"]
        self.model._args_and_kwargs = package["init_args_and_kwargs"]

        package_version = package["version"]
        current_version = version("megafold")

        if package_version != current_version:
            self.print(
                f"WARNING: Loading a saved model from version {package_version}, but you are on version {current_version}."
            )

        # for eventually saving entire training history in filename

        self.model_loaded_from_path = path
        self.last_loaded_train_id = package["id"]

        if only_model:
            return

        # install remaining metadata

        if reset_steps:
            self.steps = 0
        else:
            self.steps = package.get("steps", 0)

        self.best_model_selection_step = package.get("best_model_selection_step", -1)
        self.best_model_selection_score = package.get("best_model_selection_score", -float("inf"))
        self.best_top_ranked_lddt = package.get("best_top_ranked_lddt", -float("inf"))

    # shortcut methods

    def wait(self):
        """Wait for all ranks to sync."""
        self.fabric.barrier()

    def print(self, *args, **kwargs):
        """Print to stdout."""
        self.fabric.print(*args, **kwargs)

    def log(self, name: str, value: Any):
        """Log dictionary."""
        self.fabric.log(name, value, step=self.steps)

    def log_dict(self, **log_data):
        """Log dictionary."""
        self.fabric.log_dict(log_data, step=self.steps)

    def batch_size(self) -> int:
        """Number of samples between optimizer steps per data-parallel rank."""
        batch_size = self.global_batch_size // (self.devices * self.num_nodes)
        assert batch_size > 0, "Effective batch size must be greater than 0."
        return batch_size

    # MSA caching

    def cache_msas(self, split: Literal["train", "val", "test"]):
        """Cache MSAs for a given dataset split."""
        dataloader = self.dataloader

        if split == "val":
            dataloader = self.valid_dataloader
            assert self.needs_valid, "Validation dataloader not available."
        elif split == "test":
            dataloader = self.test_dataloader
            assert self.needs_test, "Test dataloader not available."

        for _ in tqdm(dataloader, desc=f"Caching MSAs for {split} split..."):
            pass

        self.print(f"Finished caching MSAs for {split} split.")

    # input caching

    def cache_inputs(self, split: Literal["train", "val", "test"]):
        """Cache input features for a given dataset split."""
        dataloader = self.dataloader

        if split == "val":
            dataloader = self.valid_dataloader
            assert self.needs_valid, "Validation dataloader not available."
        elif split == "test":
            dataloader = self.test_dataloader
            assert self.needs_test, "Test dataloader not available."

        for _ in tqdm(dataloader, desc=f"Caching inputs for {split} split..."):
            pass

        self.print(f"Finished caching inputs for {split} split.")

    # sampling and visualization

    @typecheck
    @torch.inference_mode()
    def visualize(
        self,
        sampled_atom_pos: Float["b m 3"],  # type: ignore
        atom_mask: Bool["b m"],  # type: ignore
        filepaths: List[str],
        batch_idx: int,
        phase: PHASES,
        sample_idx: int = 1,
        filename_suffixes: List[str] | None = None,
        b_factors: Float["b m"] | None = None,  # type: ignore
        allow_atom_mask_mismatch: bool = False,
        verbose: bool = False,
    ) -> None:
        """Visualize samples pre-generated for the examples in a batch.

        :param sampled_atom_pos: The sampled atom positions for the batch.
        :param atom_mask: The atom mask for the batch.
        :param filepaths: The filepaths of the input examples.
        :param batch_idx: The index of the current batch.
        :param phase: The phase of the current step.
        :param sample_idx: The index of the sample to visualize.
        :param filename_suffixes: The suffixes to append to the filenames.
        :param b_factors: The B-factors or equivalent mmCIF field values to list for each atom.
        :param allow_atom_mask_mismatch: Whether to allow the atom mask to mismatch the atom
            positions.
        :param verbose: Whether to print verbose output.
        """
        if verbose:
            self.print(f"Visualizing {phase} samples...")

        samples_output_dir = os.path.join(self.checkpoint_folder, f"{phase}_samples")
        os.makedirs(samples_output_dir, exist_ok=True)

        batch_size = len(atom_mask)

        for b in range(batch_size):
            input_filepath = filepaths[b]
            file_id = os.path.splitext(os.path.basename(input_filepath))[0]
            filename_suffix = filename_suffixes[b] if exists(filename_suffixes) else ""

            output_filepath = os.path.join(
                samples_output_dir,
                os.path.basename(input_filepath).replace(
                    ".cif",
                    f"-sampled-step-{self.steps}-batch-{batch_idx}-example-{b}-sample-{sample_idx}{filename_suffix}.cif",
                ),
            )

            example_atom_mask = atom_mask[b]
            sampled_atom_positions = sampled_atom_pos[b][example_atom_mask].float().cpu().numpy()
            example_b_factors = (
                b_factors[b][example_atom_mask].float().cpu().numpy()
                if exists(b_factors)
                else None
            )

            mmcif_writing.write_mmcif_from_filepath_and_id(
                input_filepath=input_filepath,
                output_filepath=output_filepath,
                file_id=file_id,
                gapless_poly_seq=True,
                insert_orig_atom_names=True,
                insert_megafold_mmcif_metadata=True,
                sampled_atom_positions=sampled_atom_positions,
                b_factors=example_b_factors,
                allow_atom_mask_mismatch=allow_atom_mask_mismatch,
            )

    @typecheck
    @torch.no_grad()
    def sample_and_visualize(
        self,
        model: Module,
        batch: BatchedAtomInput,
        batch_idx: int,
        phase: PHASES,
        sample_idx: int = 1,
        filename_suffixes: List[str] | None = None,
        allow_atom_mask_mismatch: bool = False,
        verbose: bool = False,
    ) -> None:
        """Visualize samples generated for the examples in the input batch.

        :param model: The model to use for sampling.
        :param batch: A batch of `AtomInput` data.
        :param batch_idx: The index of the current batch.
        :param phase: The phase of the current step.
        :param sample_idx: The index of the sample to visualize.
        :param filename_suffixes: The suffixes to append to the filenames.
        :param allow_atom_mask_mismatch: Whether to allow the atom mask to mismatch the atom positions.
        :param verbose: Whether to print verbose output.
        """
        if verbose:
            self.print(f"Sampling and visualizing {phase} samples...")

        batch_sampled_atom_pos = timeout(
            dec_timeout=SAMPLING_MAX_SECONDS_PER_INPUT,
            use_signals=True,
            timeout_exception=BaseException,
        )(model.__call__)(
            **batch.dict(),
            dtype=self.dtype,
            return_loss=False,
            num_sample_steps=200,
            num_recycling_steps=4,
            verbose=verbose,
        )

        samples_output_dir = os.path.join(self.checkpoint_folder, f"{phase}_samples")
        os.makedirs(samples_output_dir, exist_ok=True)

        for example_idx, sampled_atom_pos in enumerate(batch_sampled_atom_pos):
            input_filepath = batch.filepath[example_idx]
            file_id = os.path.splitext(os.path.basename(input_filepath))[0]
            filename_suffix = filename_suffixes[example_idx] if exists(filename_suffixes) else ""

            output_filepath = os.path.join(
                samples_output_dir,
                os.path.basename(input_filepath).replace(
                    ".cif",
                    f"-sampled-step-{self.steps}-batch-{batch_idx}-example-{example_idx}-sample-{sample_idx}{filename_suffix}.cif",
                ),
            )

            atom_mask = ~batch.missing_atom_mask[example_idx]
            sampled_atom_positions = sampled_atom_pos[atom_mask].cpu().numpy()

            mmcif_writing.write_mmcif_from_filepath_and_id(
                input_filepath=input_filepath,
                output_filepath=output_filepath,
                file_id=file_id,
                gapless_poly_seq=True,
                insert_orig_atom_names=True,
                insert_megafold_mmcif_metadata=True,
                sampled_atom_positions=sampled_atom_positions,
                allow_atom_mask_mismatch=allow_atom_mask_mismatch,
            )

    # main train forwards

    def __call__(self, verbose: Literal["", "standard", "extra"] = "extra"):
        """Train model."""
        self.generate_train_id()

        # cycle through dataloader

        dl = CycleIterator(self.dataloader)

        # set up metric accumulation

        self.wait()

        # maybe start profiling

        if self.profile:
            self.print("Starting profiler...")
            self.profiler.start()

        # prepare model selection buffers on the correct device

        samples_device = "cpu" if self.samples_on_cpu else self.device

        self.compute_model_selection_score.dist_breaks = (
            self.compute_model_selection_score.dist_breaks.to(samples_device)
        )
        self.compute_model_selection_score.lddt_thresholds = (
            self.compute_model_selection_score.lddt_thresholds.to(samples_device)
        )

        self.compute_model_selection_score.compute_confidence_score.pae_breaks = (
            self.compute_model_selection_score.compute_confidence_score.pae_breaks.to(
                samples_device
            )
        )
        self.compute_model_selection_score.compute_confidence_score.pde_breaks = (
            self.compute_model_selection_score.compute_confidence_score.pde_breaks.to(
                samples_device
            )
        )

        # prepare optimizer gradient clearing procedure

        zero_grad = (
            self.model_optimizer.zero_grad
            if hasattr(self.model_optimizer, "zero_grad")
            else (
                compose_calls(
                    self.model_optimizer.clear_lp_grads, self.model_optimizer.clear_hp_grads
                )
            )
        )

        # while less than required number of training steps

        grad_accum_iter = 0
        prevTime = None
        timeSoFar = [] 
        lossSoFar = [] 

        while self.steps < self.num_train_steps:
            if self.steps == 121: # CAP: only train for 121 steps
                break 
            self.model.train()

            grad_accum_iter += 1
            is_accumulating = grad_accum_iter < self.grad_accum_every

            # fetch training batch

            if verbose:
                self.print(
                    f"Step {self.steps}, Accum {grad_accum_iter} | Fetching training batch..."
                )

            # track time 
            if prevTime is not None:
                diff = time.time() - prevTime
                print(f"Time taken for training step {self.steps}: {diff}")
                timeSoFar.append(diff)
                print(f"Time over the steps: {timeSoFar}")
                print(f"Loss over the steps: {lossSoFar}")
                print("Memory usage: " + SynchronizedWallClockTimer.memory_usage())
            prevTime = time.time()

            train_batch = next(dl)
            input = train_batch.dict()
            print("\n Sequence length: ", input["is_molecule_types"].shape[1])
           
            # maybe profile

            if self.profile:
                self.profiler.step()

                if self.steps >= 1 + 1 + 3:
                    break

            # forward pass

            with self.fabric.no_backward_sync(
                self.model, enabled=is_accumulating and not self.using_deepspeed_strategy
            ):
                loss_breakdown = None

                try:
                    if verbose == "extra":
                        self.print(f"Step {self.steps}, Accum {grad_accum_iter} | Forward pass...")

                    loss, loss_breakdown = timeout(
                        dec_timeout=FORWARD_MAX_SECONDS_PER_INPUT,
                        use_signals=True,
                        timeout_exception=BaseException,
                    )(self.model.__call__)(
                        **train_batch.dict(),
                        dtype=self.dtype,
                        return_loss_breakdown=True,
                        call_confidence_head=self.steps % self.confidence_head_interval == 0,
                        # verbose=verbose == "extra",
                    )

                except BaseException as e:
                    self.print(
                        f"Step {self.steps}, Accum {grad_accum_iter} | Skipping training batch due to forward base exception: {e}, {traceback.format_exc()}"
                    )
                    loss = torch.tensor([torch.nan], device=self.device)

                    if "out of memory" in str(e):
                        self.print(
                            f"Step {self.steps}, Accum {grad_accum_iter} | Failing on training batch forward due to GPU being out of memory."
                        )

                except Exception as e:
                    self.print(
                        f"Step {self.steps}, Accum {grad_accum_iter} | Skipping training batch due to forward exception: {e}, {traceback.format_exc()}"
                    )
                    loss = torch.tensor([torch.nan], device=self.device)

                    if "out of memory" in str(e):
                        self.print(
                            f"Step {self.steps}, Accum {grad_accum_iter} | Failing on training batch forward due to GPU being out of memory."
                        )

                # skip step if any device fails its forward pass (e.g., by running out of memory)

                self.wait()
                losses = self.fabric.all_gather(loss)

                if torch.isnan(losses).any() or torch.isinf(losses).any():
                    self.print(
                        f"Step {self.steps}, Accum {grad_accum_iter} | Skipping training batch due to invalid (e.g., NaN or inf) loss."
                    )

                    # clean up the computational graph using a cool-down period

                    self.wait()
                    zero_grad()

                    del train_batch, loss, losses
                    if exists(loss_breakdown):
                        del loss_breakdown

                    garbage_collection_cuda()

                    grad_accum_iter = 0
                    is_accumulating = grad_accum_iter < self.grad_accum_every

                    self.wait()
                    continue

                # backward pass

                try:
                    if verbose == "extra":
                        self.print(
                            f"Step {self.steps}, Accum {grad_accum_iter} | Backward pass..."
                        )
                    # NOTE: DeepSpeed handles gradient accumulation internally
                    self.fabric.backward(
                        loss / (1.0 if self.using_deepspeed_strategy else self.grad_accum_every)
                    )
                except Exception as e:
                    self.print(
                        f"Step {self.steps}, Accum {grad_accum_iter} | Failing on training batch backward due to exception: {e}, {traceback.format_exc()}"
                    )
                    raise e

            # proceed only after accumulating all gradients

            if not is_accumulating:
                # loss metrics

                for k, v in loss_breakdown._asdict().items():
                    # lazily create breakdown metrics
                    if not hasattr(self, f"mean_loss_breakdown_{k}"):
                        setattr(
                            self,
                            f"mean_loss_breakdown_{k}",
                            MeanMetric(sync_on_compute=True).to(self.device),
                        )
                    mean_train_metric = getattr(self, f"mean_loss_breakdown_{k}")
                    mean_train_metric.update(v.detach() if torch.is_tensor(v) else v)

                # gradient clipping

                if self.clip_grad_norm > 0 and not self.using_deepspeed_strategy:
                    if verbose == "extra":
                        self.print(
                            f"Step {self.steps} | Clipping gradients to a maximum norm of {self.clip_grad_norm}..."
                        )
                    self.fabric.clip_gradients(
                        self.model, self.model_optimizer, max_norm=self.clip_grad_norm
                    )

                # optimizer step

                if verbose == "extra":
                    self.print(f"Step {self.steps} | Optimization...")

                self.model_optimizer.step()

                # update exponential moving average

                if verbose == "extra":
                    self.print(f"Step {self.steps} | EMA update...")

                # NOTE: it is assumed that for non-parameter-sharding training strategies
                # such as DeepSpeed ZeRO Stage 2, the model parameters at this point on each
                # device are identical for the current EMA weight update, such that the
                # rank zero EMA weights can subsequently be treated as global EMA weights

                if self.has_ema:
                    self.ema_model.update()

                # zero gradients

                if not isinstance(self.fabric.strategy, DeepSpeedStrategy):
                    # NOTE: DeepSpeed handles gradient zeroing internally

                    if verbose == "extra":
                        self.print(f"Step {self.steps} | Zeroing gradients...")

                    zero_grad()

                # update scheduler

                if verbose == "extra":
                    self.print(f"Step {self.steps} | Scheduler update...")

                self.scheduler.step()

                # increment steps

                self.steps += 1
                grad_accum_iter = 0

                # visualize samples

                seq_len = train_batch.molecule_atom_lens.shape[-1]
                filepaths_available = hasattr(train_batch, "filepath") and exists(
                    train_batch.filepath
                )
                visualize_samples = (
                    # NOTE: we cannot visualize cropped examples, since the sampled atom positions
                    # would then not be of the same shape as the original atom positions
                    filepaths_available
                    and self.visualize_train_samples_every_n_steps > 0
                    and self.steps % self.visualize_train_samples_every_n_steps == 0
                    and seq_len < self.crop_size
                )

                if visualize_samples:
                    if verbose == "extra":
                        self.print(f"Step {self.steps} | Sample visualization...")

                    eval_model = default(self.ema_model, self.model)

                    with torch.no_grad(), to_device_and_back(eval_model, self.device):
                        eval_model.eval()

                        try:
                            self.sample_and_visualize(
                                eval_model,
                                train_batch,
                                self.steps,
                                phase="train",
                                # verbose=verbose in ("standard", "extra"),
                            )

                        except BaseException as e:
                            self.print(
                                f"Step {self.steps} | Skipping sample visualization due to base exception: {e}, {traceback.format_exc()}"
                            )
                            garbage_collection_cuda()

                        except Exception as e:
                            self.print(
                                f"Step {self.steps} | Skipping sample visualization due to exception: {e}, {traceback.format_exc()}"
                            )
                            garbage_collection_cuda()

                # log

                if self.steps % self.train_log_interval == 0:
                    if verbose == "extra":
                        self.print(f"Step {self.steps} | Logging...")

                    metrics = {
                        "step": self.steps,
                        "learning_rate": self.scheduler.get_last_lr()[0],
                    }

                    for k in loss_breakdown._asdict():
                        # NOTE: these are expensive device-to-host synchronizations
                        mean_train_metric = getattr(self, f"mean_loss_breakdown_{k}")
                        metrics[f"train/{k}"] = mean_train_metric.compute().item()

                    self.print(
                        f"Step {metrics['step']} |"
                        f" Train loss: {metrics['train/total_loss']:.6f} (step)"
                    )
                    lossSoFar.append(metrics['train/total_loss'])
                    

                    self.log_dict(**metrics)

                # maybe validate with EMA model

                force_save_best_checkpoint = False

                if self.needs_valid and divisible_by(self.steps, self.valid_every):
                    if verbose == "extra":
                        self.print(f"Step {self.steps} | Validating...")

                    # set up metric accumulation

                    mean_model_selection_score = MeanMetric(sync_on_compute=True).to(self.device)
                    mean_top_ranked_lddt = MeanMetric(sync_on_compute=True).to(self.device)

                    self.wait()

                    if verbose:
                        self.print("Validating...")

                    eval_model = default(self.ema_model, self.model)

                    with torch.no_grad(), to_device_and_back(eval_model, self.device):
                        eval_model.eval()

                        for valid_batch_idx, valid_batch in enumerate(self.valid_dataloader):
                            if (
                                exists(self.num_valid_steps)
                                and valid_batch_idx >= self.num_valid_steps
                            ):
                                self.print(
                                    f"Step {self.steps} |"
                                    f" Stopping validation early after seeing {self.num_valid_steps} val batches."
                                )
                                del valid_batch
                                garbage_collection_cuda()
                                break

                            if verbose == "extra":
                                self.print(
                                    f"Step {self.steps} | Running val step {valid_batch_idx}..."
                                )

                            # generate multiple samples per example in each batch

                            valid_samples: List[Sample] = []

                            try:
                                for _ in range(self.num_samples_per_example):
                                    valid_sampled_atom_pos, valid_logits = timeout(
                                        dec_timeout=SAMPLING_MAX_SECONDS_PER_INPUT,
                                        use_signals=True,
                                        timeout_exception=BaseException,
                                    )(eval_model.__call__)(
                                        **valid_batch.dict(),
                                        dtype=self.dtype,
                                        return_loss=False,
                                        return_confidence_head_logits=True,
                                        return_distogram_head_logits=True,
                                        num_sample_steps=200,
                                        num_recycling_steps=4,
                                        # verbose=verbose == "extra",
                                    )
                                    valid_plddt = ComputeConfidenceScore.compute_plddt(
                                        valid_logits.plddt.to(samples_device)
                                    )
                                    valid_samples.append(
                                        (
                                            valid_sampled_atom_pos.to(samples_device),
                                            valid_logits.pde.to(samples_device),
                                            valid_plddt.to(samples_device),
                                            valid_logits.distance.to(samples_device),
                                        )
                                    )

                            except BaseException as e:
                                self.print(
                                    f"Step {self.steps} |"
                                    f" Skipping validation step {valid_batch_idx} due to base exception: {e}, {traceback.format_exc()}"
                                )
                                mean_model_selection_score.update(
                                    torch.tensor([torch.nan], device=self.device)
                                )
                                mean_top_ranked_lddt.update(
                                    torch.tensor([torch.nan], device=self.device)
                                )

                                del valid_batch
                                garbage_collection_cuda()

                            except Exception as e:
                                self.print(
                                    f"Step {self.steps} |"
                                    f" Skipping validation step {valid_batch_idx} due to exception: {e}, {traceback.format_exc()}"
                                )
                                mean_model_selection_score.update(
                                    torch.tensor([torch.nan], device=self.device)
                                )
                                mean_top_ranked_lddt.update(
                                    torch.tensor([torch.nan], device=self.device)
                                )

                                del valid_batch
                                garbage_collection_cuda()

                            # NOTE: we must wait until all ranks are synchronized each validation step
                            # before we decide which ranks can compute valid scores and visualize samples
                            self.wait()
                            if len(valid_samples) != self.num_samples_per_example:
                                continue

                            valid_score_details = self.compute_model_selection_score.compute_model_selection_score(
                                valid_batch,
                                valid_samples,
                                is_fine_tuning=self.is_fine_tuning,
                                return_details=True,
                                # NOTE: the AF3 supplement (Section 5.7) suggests that DM did not compute validation RASA for unresolved regions
                                compute_rasa=False,
                                device=samples_device,
                            )

                            valid_top_sample = valid_score_details.scored_samples[
                                valid_score_details.best_gpde_index
                            ]
                            (
                                valid_top_sample_idx,
                                valid_top_batch_sampled_atom_pos,
                                valid_top_sample_plddt,
                                valid_top_model_selection_score,
                                _,
                            ) = valid_top_sample

                            # compute the unweighted lDDT score

                            valid_unweighted_score_details = (
                                self.compute_model_selection_score.compute_model_selection_score(
                                    valid_batch,
                                    valid_samples,
                                    is_fine_tuning=self.is_fine_tuning,
                                    return_details=True,
                                    return_unweighted_scores=True,
                                    compute_rasa=False,
                                    device=samples_device,
                                )
                            )

                            valid_unweighted_top_sample = (
                                valid_unweighted_score_details.scored_samples[
                                    valid_unweighted_score_details.best_gpde_index
                                ]
                            )
                            valid_top_ranked_lddt = valid_unweighted_top_sample[3]

                            mean_model_selection_score.update(
                                valid_score_details.score.mean().detach()
                            )
                            mean_top_ranked_lddt.update(valid_top_ranked_lddt.mean().detach())

                            # visualize (top) samples

                            seq_len = valid_batch.molecule_atom_lens.shape[-1]
                            filepaths_available = hasattr(valid_batch, "filepath") and exists(
                                valid_batch.filepath
                            )
                            visualize_samples = (
                                # NOTE: we cannot visualize cropped examples, since the sampled atom positions
                                # would then not be of the same shape as the original atom positions
                                filepaths_available
                                and self.visualize_valid_samples_every_n_steps > 0
                                and self.steps % self.visualize_valid_samples_every_n_steps == 0
                                and seq_len < self.crop_size
                            )

                            if visualize_samples:
                                assert exists(
                                    valid_top_batch_sampled_atom_pos
                                ), "The top sampled validation atom positions must be provided to visualize them."
                                filename_suffixes = [
                                    f"-score-{score:.4f}"
                                    for score in valid_top_model_selection_score.tolist()
                                ]
                                filepaths = (
                                    list(valid_batch.filepath)
                                    if hasattr(valid_batch, "filepath")
                                    and exists(valid_batch.filepath)
                                    else None
                                )
                                if exists(filepaths):
                                    self.visualize(
                                        sampled_atom_pos=valid_top_batch_sampled_atom_pos,
                                        atom_mask=~valid_batch.missing_atom_mask,
                                        filepaths=filepaths,
                                        batch_idx=valid_batch_idx,
                                        phase="val",
                                        sample_idx=valid_top_sample_idx,
                                        filename_suffixes=filename_suffixes,
                                        b_factors=valid_top_sample_plddt,
                                        # verbose=verbose in ("standard", "extra"),
                                    )

                    # log

                    valid_model_selection_score = (
                        mean_model_selection_score.compute().item()
                    )  # NOTE: expensive device-to-host synchronization
                    valid_top_ranked_lddt = (
                        mean_top_ranked_lddt.compute().item()
                    )  # NOTE: expensive device-to-host synchronization

                    valid_metrics = {
                        "val/model_selection_score": valid_model_selection_score,
                        "val/top_ranked_lddt": valid_top_ranked_lddt,
                    }

                    self.print(
                        f"Step {self.steps} |"
                        f" Val model selection score: {valid_metrics['val/model_selection_score']:.6f} (epoch),",
                        f" Val top ranked lDDT: {valid_metrics['val/top_ranked_lddt']:.6f} (epoch)",
                    )

                    self.log_dict(**valid_metrics)

                    self.wait()

                    # track best model selection score

                    if (
                        valid_metrics["val/model_selection_score"]
                        > self.best_model_selection_score
                    ):
                        if verbose:
                            self.print(
                                f"Step {self.steps} |"
                                f" New best val model selection score: {valid_metrics['val/model_selection_score']:.6f} (epoch),",
                                f" New best val top ranked lDDT: {valid_metrics['val/top_ranked_lddt']:.6f} (epoch)",
                            )

                        self.best_model_selection_step = self.steps
                        self.best_model_selection_score = valid_metrics[
                            "val/model_selection_score"
                        ]
                        self.best_top_ranked_lddt = valid_metrics["val/top_ranked_lddt"]

                        force_save_best_checkpoint = True

                # maybe save a checkpoint

                if force_save_best_checkpoint or divisible_by(self.steps, self.checkpoint_every):
                    if verbose == "extra":
                        self.print(
                            f"Step {self.steps} | Saving a{' new best ' if force_save_best_checkpoint else ' '}checkpoint..."
                        )
                    self.save_checkpoint()

                # clear CUDA cache

                if (
                    self.clear_cuda_cache_every > 0
                    and self.steps % self.clear_cuda_cache_every == 0
                ):
                    if verbose == "extra":
                        self.print(f"Step {self.steps} | Clearing CUDA cache...")
                    torch.cuda.empty_cache()

        # maybe finish profiling

        if self.profile:
            self.print("Stopping profiler...")
            self.profiler.stop()

        # maybe test

        if self.needs_test:
            self.wait()

            self.load_from_checkpoint_folder(load_best_model=True)

            # set up metric accumulation

            mean_model_selection_score = MeanMetric(sync_on_compute=True).to(self.device)
            mean_top_ranked_lddt = MeanMetric(sync_on_compute=True).to(self.device)

            self.wait()

            if verbose:
                self.print("Testing...")

            eval_model = default(self.ema_model, self.model)

            with torch.no_grad(), to_device_and_back(eval_model, self.device):
                eval_model.eval()

                for test_batch_idx, test_batch in enumerate(self.test_dataloader):
                    if exists(self.num_test_steps) and test_batch_idx >= self.num_test_steps:
                        self.print(
                            f"Step {self.steps} |"
                            f" Stopping testing early after seeing {self.num_test_steps} test batches."
                        )
                        del test_batch
                        garbage_collection_cuda()
                        break

                    if verbose == "extra":
                        self.print(f"Step {self.steps} | Running test step {test_batch_idx}...")

                    # generate multiple samples per example in each batch

                    test_samples: List[Sample] = []

                    try:
                        for _ in range(self.num_samples_per_example):
                            test_sampled_atom_pos, test_logits = timeout(
                                dec_timeout=SAMPLING_MAX_SECONDS_PER_INPUT,
                                use_signals=True,
                                timeout_exception=BaseException,
                            )(eval_model.__call__)(
                                **test_batch.dict(),
                                dtype=self.dtype,
                                return_loss=False,
                                return_confidence_head_logits=True,
                                return_distogram_head_logits=True,
                                num_sample_steps=200,
                                num_recycling_steps=4,
                                # verbose=verbose == "extra",
                            )
                            test_plddt = ComputeConfidenceScore.compute_plddt(
                                test_logits.plddt.to(samples_device)
                            )
                            test_samples.append(
                                (
                                    test_sampled_atom_pos.to(samples_device),
                                    test_logits.pde.to(samples_device),
                                    test_plddt.to(samples_device),
                                    test_logits.distance.to(samples_device),
                                )
                            )

                    except BaseException as e:
                        self.print(
                            f"Step {self.steps} |"
                            f" Skipping test step {test_batch_idx} due to base exception: {e}, {traceback.format_exc()}"
                        )
                        mean_model_selection_score.update(
                            torch.tensor([torch.nan], device=self.device)
                        )
                        mean_top_ranked_lddt.update(torch.tensor([torch.nan], device=self.device))

                        del test_batch
                        garbage_collection_cuda()

                    except Exception as e:
                        self.print(
                            f"Step {self.steps} |"
                            f" Skipping test step {test_batch_idx} due to exception: {e}, {traceback.format_exc()}"
                        )
                        mean_model_selection_score.update(
                            torch.tensor([torch.nan], device=self.device)
                        )
                        mean_top_ranked_lddt.update(torch.tensor([torch.nan], device=self.device))

                        del test_batch
                        garbage_collection_cuda()

                    # NOTE: we must wait until all ranks are synchronized each test step
                    # before we decide which ranks can compute valid scores and visualize samples
                    self.wait()
                    if len(test_samples) != self.num_samples_per_example:
                        continue

                    test_score_details = self.compute_model_selection_score.compute_model_selection_score(
                        test_batch,
                        test_samples,
                        is_fine_tuning=self.is_fine_tuning,
                        return_details=True,
                        return_unweighted_scores=False,
                        # NOTE: the AF3 supplement (Section 5.7) suggests that DM computed RASA only for the test set's unresolved regions
                        # NOTE: cannot find where to get the unresolved chain IDs and residue masks from to match the AF3 supplement
                        compute_rasa=True,
                        unresolved_cid=None,
                        unresolved_residue_mask=None,
                        device=samples_device,
                    )

                    test_top_sample = test_score_details.scored_samples[
                        test_score_details.best_gpde_index
                    ]
                    (
                        test_top_sample_idx,
                        test_top_batch_sampled_atom_pos,
                        test_top_sample_plddt,
                        test_top_model_selection_score,
                        _,
                    ) = test_top_sample

                    # compute the unweighted lDDT score

                    test_unweighted_score_details = (
                        self.compute_model_selection_score.compute_model_selection_score(
                            test_batch,
                            test_samples,
                            is_fine_tuning=self.is_fine_tuning,
                            return_details=True,
                            return_unweighted_scores=True,
                            compute_rasa=False,
                            device=samples_device,
                        )
                    )

                    test_unweighted_top_sample = test_unweighted_score_details.scored_samples[
                        test_unweighted_score_details.best_gpde_index
                    ]
                    test_top_ranked_lddt = test_unweighted_top_sample[3]

                    mean_model_selection_score.update(test_score_details.score.mean().detach())
                    mean_top_ranked_lddt.update(test_top_ranked_lddt.mean().detach())

                    # visualize (top) samples

                    seq_len = test_batch.molecule_atom_lens.shape[-1]
                    filepaths_available = hasattr(test_batch, "filepath") and exists(
                        test_batch.filepath
                    )
                    visualize_samples = (
                        # NOTE: we cannot visualize cropped examples, since the sampled atom positions
                        # would then not be of the same shape as the original atom positions
                        filepaths_available
                        and self.visualize_test_samples_every_n_steps > 0
                        and self.steps % self.visualize_test_samples_every_n_steps == 0
                        and seq_len < self.crop_size
                    )

                    if visualize_samples:
                        assert exists(
                            test_top_batch_sampled_atom_pos
                        ), "The top sampled test atom positions must be provided to visualize them."
                        filename_suffixes = [
                            f"-score-{score:.4f}"
                            for score in test_top_model_selection_score.tolist()
                        ]
                        filepaths = (
                            list(test_batch.filepath)
                            if hasattr(test_batch, "filepath") and exists(test_batch.filepath)
                            else None
                        )
                        if exists(filepaths):
                            self.visualize(
                                sampled_atom_pos=test_top_batch_sampled_atom_pos,
                                atom_mask=~test_batch.missing_atom_mask,
                                filepaths=filepaths,
                                batch_idx=test_batch_idx,
                                phase="test",
                                sample_idx=test_top_sample_idx,
                                filename_suffixes=filename_suffixes,
                                b_factors=test_top_sample_plddt,
                                # verbose=verbose in ("standard", "extra"),
                            )

            # log

            test_model_selection_score = (
                mean_model_selection_score.compute().item()
            )  # NOTE: expensive device-to-host synchronization
            test_top_ranked_lddt = (
                mean_top_ranked_lddt.compute().item()
            )  # NOTE: expensive device-to-host synchronization

            test_metrics = {
                "test/model_selection_score": test_model_selection_score,
                "test/top_ranked_lddt": test_top_ranked_lddt,
            }

            self.print(
                f"Step {self.steps} |"
                f" Test model selection score: {test_metrics['test/model_selection_score']:.6f} (epoch),",
                f" Test top ranked lDDT: {test_metrics['test/top_ranked_lddt']:.6f} (epoch)",
            )

            self.log_dict(**test_metrics)

        self.wait()

        # maybe log profiler artifacts

        if self.profile and self.is_main:
            assert package_available(
                "wandb"
            ), "Please install and use the `wandb` package to log profiler artifacts."
            import wandb

            profile_art = wandb.Artifact("trace", type="profile")

            trace_files = list(glob.glob(os.path.join(self.profiler_log_dir, "*.pt.trace.json")))
            assert trace_files, "No trace files found."

            profile_art.add_file(trace_files[0], "trace.pt.trace.json")
            self.fabric.logger.experiment.log_artifact(profile_art)

            self.print("Profiler artifacts logged.")

        print("Training complete.")

