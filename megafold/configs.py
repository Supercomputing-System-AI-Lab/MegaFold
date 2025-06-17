from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path

import polars as pl
import yaml
from beartype.typing import Any, Callable, Dict, List, Literal, Tuple
from pydantic import BaseModel, ConfigDict, model_validator
from pydantic.types import DirectoryPath, FilePath
from torch.utils.data import ConcatDataset, WeightedRandomSampler

from megafold.data.weighted_pdb_sampler import WeightedPDBSampler
from megafold.inputs import (
    CONSTRAINTS,
    AtomDataset,
    PDBDataset,
    PDBDistillationDataset,
    pdb_dataset_to_atom_inputs,
)
from megafold.nlm import NLMEmbedding
import os 
if os.environ.get("AF3_OPTIMIZATIONS_MODE") == "baseline": 
    from megafold.baseline import MegaFold
elif os.environ.get("AF3_OPTIMIZATIONS_MODE") == "layernormlinear":
    from megafold.megafold_layernormlinear import MegaFold
else:
    from megafold.model.megafold import MegaFold
from megafold.plm import PLMEmbedding
from megafold.tensor_typing import typecheck
from megafold.trainer import Dataset, Fabric, LRScheduler, Optimizer, Trainer
from megafold.utils.utils import exists, not_exists

# functions


@typecheck
def safe_deep_get(
    d: dict,
    dotpath: (
        str | List[str]
    ),  # dotpath notation, so accessing {'a': {'b'': {'c': 1}}} would be "a.b.c"
    default=None,
) -> Any:
    """Safely get a value from a nested dictionary using dotpath notation, returning a default
    value if the key is not found."""
    if isinstance(dotpath, str):
        dotpath = dotpath.split(".")

    for key in dotpath:
        if not isinstance(d, dict) or key not in d:
            return default

        d = d[key]

    return d


@typecheck
def yaml_config_path_to_dict(path: str | Path) -> dict:
    """Parse a yaml config file at the given path and return the dictionary representation."""
    if isinstance(path, str):
        path = Path(path)

    assert path.is_file(), f"Cannot find {str(path)}."

    with open(str(path), "r") as f:
        maybe_config_dict = yaml.safe_load(f)

    assert exists(maybe_config_dict), f"Unable to parse yaml config at {str(path)}."
    assert isinstance(maybe_config_dict, dict), "YAML config file is not a dictionary."

    return maybe_config_dict


# base pydantic classes for constructing megafold and trainer from config files


class BaseModelWithExtra(BaseModel):
    """Base class for Pydantic models that allows for extra fields."""

    model_config = ConfigDict(
        extra="allow",
        use_enum_values=True,
    )


class MegaFoldConfig(BaseModelWithExtra):
    """Pydantic model for the MegaFold configuration."""

    dim_atom_inputs: int
    dim_template_feats: int
    dim_template_model: int
    atoms_per_window: int
    dim_atom: int
    dim_atompair_inputs: int
    dim_atompair: int
    dim_input_embedder_token: int
    dim_single: int
    dim_pairwise: int
    dim_token: int
    ignore_index: int = -1
    num_dist_bins: int | None
    num_plddt_bins: int
    num_pde_bins: int
    num_pae_bins: int
    sigma_data: int | float
    diffusion_num_augmentations: int = 48
    loss_confidence_weight: int | float
    loss_distogram_weight: int | float
    loss_diffusion_weight: int | float
    prior_type: Literal["diffusion"]
    multi_chain_permutation_alignment: bool
    atom_permutation_alignment: bool
    use_optimized_evo: Literal["deepspeed", "triton"] | None = None
    globally_enable_autocasting: bool = True
    use_tempo_layernorm: bool
    plm_embeddings: PLMEmbedding | Tuple[PLMEmbedding, ...] | None = None
    nlm_embeddings: NLMEmbedding | Tuple[NLMEmbedding, ...] | None = None
    plm_kwargs: dict | Tuple[dict, ...] | None = None
    nlm_kwargs: dict | Tuple[dict, ...] | None = None
    constraints: List[CONSTRAINTS] | None = None

    @classmethod
    @typecheck
    def from_yaml_file(cls, path: str | Path, dotpath: str | List[str] = []) -> MegaFoldConfig:
        """Create an instance of MegaFoldConfig from a yaml file at the given path."""
        config_dict = yaml_config_path_to_dict(path)
        config_dict = safe_deep_get(config_dict, dotpath)
        assert exists(config_dict), f'Config not found at path {".".join(dotpath)}.'

        return cls(**config_dict)

    def create_instance(self) -> MegaFold:
        """Create an instance of MegaFold from the configuration."""
        megafold = MegaFold(**self.model_dump())
        return megafold

    @classmethod
    def create_instance_from_yaml_file(
        cls, path: str | Path, dotpath: str | List[str] = []
    ) -> MegaFold:
        """Create an instance of MegaFold from a yaml file at the given path."""
        megafold_config = cls.from_yaml_file(path, dotpath)
        return megafold_config.create_instance()


class WeightedPDBSamplerConfig(BaseModelWithExtra):
    """Pydantic model for the WeightedPDBSampler configuration."""

    chain_mapping_paths: List[FilePath]
    interface_mapping_path: FilePath

    def create_instance(
        self,
        batch_size: int,
        pdb_ids_to_skip: List[str] | None = None,
        pdb_ids_to_keep: List[str] | None = None,
    ) -> WeightedPDBSampler:
        """Create an instance of WeightedPDBSampler from the configuration."""
        return WeightedPDBSampler(
            **{
                "batch_size": batch_size,
                "pdb_ids_to_skip": pdb_ids_to_skip,
                "pdb_ids_to_keep": pdb_ids_to_keep,
                **self.model_dump(),
            }
        )


class DatasetConfig(BaseModelWithExtra):
    """Pydantic model for the Dataset configuration."""

    dataset_type: Literal["pdb", "atom"] = "pdb"
    train_folder: DirectoryPath | None = None
    valid_folder: DirectoryPath | None = None
    test_folder: DirectoryPath | None = None
    convert_pdb_to_atom: bool = False
    pdb_to_atom_kwargs: dict = dict()
    train_weighted_sampler: WeightedPDBSamplerConfig | None = None
    valid_weighted_sampler: WeightedPDBSamplerConfig | None = None
    test_weighted_sampler: WeightedPDBSamplerConfig | None = None
    pdb_distillation: bool = False
    pdb_distillation_only: bool = False
    distillation_kwargs: dict = dict()
    overfitting_train_examples: bool = (
        False  # NOTE: if true, overfit to the training dataset by treating it also as the validation and test datasets
    )
    sample_only_pdb_ids: List[str] | None = (
        None  # if specified, a subset of  PDB IDs to sample from the training, validation, or testing sets
    )
    filter_out_pdb_ids: List[str] | None = (
        None  # if specified, a subset of PDB IDs to filter out from the training, validation, or testing sets
    )
    kwargs: dict = dict()
    train_kwargs: dict = dict()
    valid_kwargs: dict = dict()
    test_kwargs: dict = dict()
    dl_kwargs: dict = dict()


class TrainerConfig(BaseModelWithExtra):
    """Pydantic model for the Trainer configuration."""

    model: MegaFoldConfig | None = None
    num_train_steps: int
    global_batch_size: int
    grad_accum_every: int
    valid_every: int
    ema_decay: float
    lr: float
    clip_grad_norm: int | float
    accelerator: str
    strategy: str
    strategy_stage: int
    checkpoint_prefix: str
    checkpoint_every: int
    checkpoint_folder: str
    overwrite_checkpoints: bool
    offload_optimizer: bool = False
    allgather_bucket_size: int = 200_000_000
    reduce_bucket_size: int = 200_000_000
    num_valid_steps: int | None = None
    num_test_steps: int | None = None
    samples_on_cpu: bool = False
    profile: bool = False
    profiler_kwargs: dict = dict(
        log_dir="./profiler_logs",
    )
    diffusion_num_augmentations: int = 48
    dataset_config: DatasetConfig | None = None
    logger_name: Literal["csv", "tensorboard", "wandb"] | None = (None,)
    logger_kwargs: dict = dict(
        out_dir=".",
        name="auto",
    )

    @classmethod
    @typecheck
    def from_yaml_file(cls, path: str | Path, dotpath: str | List[str] = []) -> TrainerConfig:
        """Create an instance of TrainerConfig from a yaml file at the given path."""
        config_dict = yaml_config_path_to_dict(path)
        config_dict = safe_deep_get(config_dict, dotpath)
        assert exists(config_dict), f'Config not found at path {".".join(dotpath)}.'

        return cls(**config_dict)

    def create_instance(
        self,
        dataset: Dataset | None = None,
        model: MegaFoldConfig | None = None,
        fabric: Fabric | None = None,
        test_dataset: Dataset | None = None,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        valid_dataset: Dataset | None = None,
        map_dataset_input_fn: Callable | None = None,
    ) -> Trainer:
        """Create an instance of Trainer from the configuration."""
        trainer_kwargs = self.model_dump(
            exclude={
                "dataset_config",
            }
        )

        assert exists(self.model) ^ exists(
            model
        ), "Either model is available on the trainer config, or passed in when creating the instance, but not both or neither."

        #print("from inside configs.py: ", len(dataset))
        # handle model

        if exists(self.model):
            megafold = self.model
        else:
            megafold = model

        # handle dataset

        if exists(dataset):
            trainer_kwargs.update(dataset=dataset)

        if exists(valid_dataset):
            trainer_kwargs.update(valid_dataset=dataset)

        if exists(test_dataset):
            trainer_kwargs.update(test_dataset=dataset)

        if exists(self.dataset_config):
            dataset_config = self.dataset_config

            dataset_type = dataset_config.dataset_type
            dataset_kwargs = dataset_config.kwargs
            distillation_kwargs = dataset_config.distillation_kwargs

            trainer_kwargs.update(dl_kwargs=dataset_config.dl_kwargs)

            convert_pdb_to_atom = dataset_config.convert_pdb_to_atom
            pdb_to_atom_kwargs = dataset_config.pdb_to_atom_kwargs

            if convert_pdb_to_atom:
                assert (
                    dataset_type == "pdb"
                ), "Must be `pdb` dataset_type if `convert_pdb_to_atom` is set to True."

            if dataset_type == "pdb":
                dataset_klass = PDBDataset
                distillation_dataset_klass = PDBDistillationDataset
            elif dataset_type == "atom":
                dataset_klass = distillation_dataset_klass = AtomDataset
            else:
                raise ValueError(f"Unhandled dataset_type {dataset_type}.")

            # subset dataset examples

            sample_only_pdb_ids = (
                # sample only specific PDB IDs as requested
                set(dataset_config.sample_only_pdb_ids)
                if exists(dataset_config.sample_only_pdb_ids)
                else None
            )
            sample_only_pdb_ids_list = (
                list(sample_only_pdb_ids) if exists(sample_only_pdb_ids) else None
            )

            filter_out_pdb_ids = (
                # filter out specific PDB IDs as requested
                set(dataset_config.filter_out_pdb_ids)
                if exists(dataset_config.filter_out_pdb_ids)
                else None
            )
            filter_out_pdb_ids_list = (
                list(filter_out_pdb_ids) if exists(filter_out_pdb_ids) else None
            )

            # create dataset for train, valid, and test

            for trainer_kwarg_key, config_key in (
                ("dataset", "train_folder"),
                ("valid_dataset", "valid_folder"),
                ("test_dataset", "test_folder"),
            ):
                split = config_key.split("_")[0]

                sample_only_pdb_ids_ = (
                    copy.deepcopy(sample_only_pdb_ids) if exists(sample_only_pdb_ids) else None
                )
                sample_only_pdb_ids_list_ = (
                    copy.deepcopy(sample_only_pdb_ids_list)
                    if exists(sample_only_pdb_ids_list)
                    else None
                )

                filter_out_pdb_ids_ = (
                    copy.deepcopy(filter_out_pdb_ids) if exists(filter_out_pdb_ids) else None
                )
                filter_out_pdb_ids_list_ = (
                    copy.deepcopy(filter_out_pdb_ids_list)
                    if exists(filter_out_pdb_ids_list)
                    else None
                )

                # handle overfitting

                orig_split = split

                if dataset_config.overfitting_train_examples:
                    split = "train"
                    config_key = f"{split}_folder"

                # retrieve the folder

                folder = getattr(dataset_config, config_key, None)

                if not_exists(folder):
                    continue

                assert trainer_kwarg_key not in trainer_kwargs

                # prefilter mmCIF files based on metadata if provided

                dataset_specific_kwargs = getattr(dataset_config, f"{split}_kwargs")

                mmcif_metadata_filepath = dataset_kwargs.get("mmcif_metadata_filepath")

                if exists(mmcif_metadata_filepath):
                    mmcif_metadata_df = pl.read_csv(mmcif_metadata_filepath)

                    if "min_length" in dataset_kwargs and exists(dataset_kwargs["min_length"]):
                        mmcif_metadata_df = mmcif_metadata_df.filter(
                            mmcif_metadata_df["num_tokens"] >= dataset_kwargs["min_length"]
                        )

                    if "max_length" in dataset_kwargs and exists(dataset_kwargs["max_length"]):
                        mmcif_metadata_df = mmcif_metadata_df.filter(
                            mmcif_metadata_df["num_tokens"] <= dataset_kwargs["max_length"]
                        )

                    if "max_num_atoms" in dataset_kwargs and exists(
                        dataset_kwargs["max_num_atoms"]
                    ):
                        # NOTE: this serves simply as a cropping-centric heuristic to filter out mmCIF files that are too large
                        mmcif_metadata_df = mmcif_metadata_df.filter(
                            (
                                (mmcif_metadata_df["num_atoms"] / mmcif_metadata_df["num_tokens"])
                                * dataset_kwargs["crop_size"]
                            )
                            <= dataset_kwargs["max_num_atoms"]
                        )

                    if "filter_for_alphabetic_chain_orderings" in dataset_kwargs and exists(
                        dataset_kwargs["filter_for_alphabetic_chain_orderings"]
                    ):
                        if dataset_kwargs["filter_for_alphabetic_chain_orderings"]:
                            # NOTE: due to a bug present during mmCIF and MSA preprocessing (affecting ~11% of the PDBDataset's complexes),
                            # this serves as a stopgap measure to filter out (bugged) complexes with non-alphabetic chain orderings
                            mmcif_metadata_df = mmcif_metadata_df.filter(
                                mmcif_metadata_df["chain_ids"].map_elements(
                                    lambda chain_ids: chain_ids.split("-")
                                    == sorted(chain_ids.split("-")),
                                    return_dtype=bool,
                                )
                            )

                    if "cutoff_date" in dataset_specific_kwargs and exists(
                        dataset_specific_kwargs["cutoff_date"]
                    ):
                        mmcif_metadata_df = mmcif_metadata_df.filter(
                            mmcif_metadata_df["release_date"]
                            <= dataset_specific_kwargs["cutoff_date"]
                        )

                    sample_only_pdb_ids_ = (
                        sample_only_pdb_ids_.intersection(
                            set(mmcif_metadata_df["file_id"].to_list())
                        )
                        if exists(sample_only_pdb_ids_)
                        else set(mmcif_metadata_df["file_id"].to_list())
                    )
                    sample_only_pdb_ids_list_ = list(sample_only_pdb_ids_)
                    assert (
                        len(sample_only_pdb_ids_) > 0
                    ), "No PDB IDs found after filtering with mmCIF metadata."

                # handle weighted pdb sampling

                sampler = None
                weighted_sampler_config = getattr(dataset_config, f"{split}_weighted_sampler")

                if exists(weighted_sampler_config):
                    sampler = weighted_sampler_config.create_instance(
                        batch_size=1,
                        pdb_ids_to_skip=filter_out_pdb_ids_list_,
                        pdb_ids_to_keep=sample_only_pdb_ids_list_,
                    )

                # instantiate dataset

                dataset = dataset_klass(
                    folder,
                    sampler=sampler,
                    filter_out_pdb_ids=filter_out_pdb_ids_,
                    sample_only_pdb_ids=sample_only_pdb_ids_,
                    **dataset_kwargs,
                    **dataset_specific_kwargs,
                )
                #print("from configs.py: ", dataset, folder, sampler, filter_out_pdb_ids_, sample_only_pdb_ids_)
                if convert_pdb_to_atom:
                    dataset = pdb_dataset_to_atom_inputs(
                        dataset, return_atom_dataset=True, **pdb_to_atom_kwargs
                    )

                # handle distillation dataset training

                if dataset_config.pdb_distillation and orig_split == "train":
                    assert exists(distillation_kwargs), (
                        "When `pdb_distillation=True`, `distillation_kwargs` must be provided "
                        "to instantiate the distillation dataset."
                    )
                    assert Path(
                        distillation_kwargs["uniprot_to_pdb_id_mapping_filepath"]
                    ).exists(), (
                        "When `pdb_distillation=True`, a `uniprot_to_pdb_id_mapping_filepath` "
                        "must be provided to map UniProt IDs to PDB IDs for distillation."
                    )
                    assert "folder" in distillation_kwargs and exists(
                        distillation_kwargs["folder"]
                    ), (
                        "When `pdb_distillation=True`, a valid `folder` must be provided "
                        "to instantiate the distillation dataset."
                    )
                    assert "distillation_template_mmcif_dir" in distillation_kwargs and exists(
                        distillation_kwargs["distillation_template_mmcif_dir"]
                    ), (
                        "When `pdb_distillation=True`, a valid `distillation_template_mmcif_dir` must be provided "
                        "to instantiate the distillation dataset."
                    )
                    assert (
                        "sampling_weight" in distillation_kwargs
                        and exists(distillation_kwargs["sampling_weight"])
                        and 0 < distillation_kwargs["sampling_weight"] < 1.0
                    ), (
                        "When `pdb_distillation=True`, a valid `sampling_weight` must be provided "
                        "to sample the distillation dataset."
                    )
                    assert exists(sampler), (
                        "When `pdb_distillation=True`, a `WeightedPDBSampler` must be provided "
                        "for the training set to ensure that the distillation data is correctly "
                        "redundancy-reduced during sampling."
                    )

                    distillation_sample_only_pdb_ids = {
                        r[0] for r in sampler.mappings.select("pdb_id").rows()
                    }
                    distillation_sample_only_pdb_ids = (
                        distillation_sample_only_pdb_ids.intersection(sample_only_pdb_ids)
                        if exists(sample_only_pdb_ids)
                        else distillation_sample_only_pdb_ids
                    )

                    distillation_sampling_weight = distillation_kwargs.pop("sampling_weight")
                    distillation_dataset = distillation_dataset_klass(
                        filter_out_pdb_ids=filter_out_pdb_ids,
                        sample_only_pdb_ids=distillation_sample_only_pdb_ids,
                        **{
                            k: v
                            for k, v in dataset_kwargs.items()
                            if k
                            not in (
                                "mmcif_metadata_filepath",
                                "pdbbind_binding_affinity_values_path",
                            )
                        },
                        **distillation_kwargs,
                    )

                    if dataset_config.pdb_distillation_only:
                        # NOTE: if `pdb_distillation_only=True`, the distillation dataset is used as the sole training dataset;
                        # this may be useful if one is adding new distillation data to the existing PDB distillation dataset
                        # and wants to cache MSA or input features for the new data before training
                        total_len_dataset = len(distillation_dataset)

                        combined_dataset_train_weights = [1.0] * total_len_dataset

                        dataset = distillation_dataset

                    elif len(distillation_dataset) == 0:
                        # NOTE: if the distillation dataset is empty after filtering, the training dataset is used as is
                        total_len_dataset = len(dataset)

                        combined_dataset_train_weights = [1.0] * total_len_dataset

                    else:
                        # otherwise, the distillation dataset is weightedly combined with the training dataset
                        len_dataset = len(dataset)
                        len_distillation_dataset = len(distillation_dataset)
                        total_len_dataset = len_dataset + len_distillation_dataset

                        combined_dataset_train_weights = [
                            1.0 - distillation_sampling_weight
                        ] * len_dataset + [distillation_sampling_weight] * len_distillation_dataset

                        dataset = ConcatDataset([dataset, distillation_dataset])

                    distillation_train_sampler = WeightedRandomSampler(
                        combined_dataset_train_weights,
                        num_samples=total_len_dataset,
                        replacement=False,
                    )

                    trainer_kwargs.update(train_sampler=distillation_train_sampler)

                trainer_kwargs.update(**{trainer_kwarg_key: dataset})

        assert (
            "dataset" in trainer_kwargs
        ), "Dataset is absent - dataset_type must be specified along with train folders (pdb for now), or the Dataset instance must be passed in."

        # loggers

        if (
            trainer_kwargs["logger_kwargs"]
            and "name" in trainer_kwargs["logger_kwargs"]
            and trainer_kwargs["logger_kwargs"]["name"] == "auto"
        ):
            trainer_kwargs["logger_kwargs"]["name"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # models

        megafold.diffusion_num_augmentations = trainer_kwargs.pop("diffusion_num_augmentations")
        megafold.diffusion_add_smooth_lddt_loss = trainer_kwargs.pop(
            "diffusion_add_smooth_lddt_loss"
        )
        megafold.diffusion_add_bond_loss = trainer_kwargs.pop("diffusion_add_bond_loss")
        megafold.train_structure_and_distogram = trainer_kwargs.pop(
            "train_structure_and_distogram"
        )
        megafold.train_pae = trainer_kwargs.pop("train_pae")

        # handle rest

        trainer_kwargs.update(
            dict(
                model=megafold,
                fabric=fabric,
                optimizer=optimizer,
                scheduler=scheduler,
                map_dataset_input_fn=map_dataset_input_fn,
            )
        )

        trainer = Trainer(**trainer_kwargs)
        return trainer

    @classmethod
    def create_instance_from_yaml_file(
        cls, path: str | Path, dotpath: str | List[str] = [], **kwargs
    ) -> Trainer:
        """Create an instance of Trainer from a yaml file at the given path."""
        trainer_config = cls.from_yaml_file(path, dotpath)
        return trainer_config.create_instance(**kwargs)


# conductor config
# which contains multiple trainer configs for the main and various finetuning stages


class ConductorConfig(BaseModelWithExtra):
    """Pydantic model for the Conductor configuration."""

    model: MegaFoldConfig | None = None
    checkpoint_folder: str
    checkpoint_prefix: str
    training_order: List[str]
    training: Dict[str, TrainerConfig]

    @model_validator(mode="after")
    def check_valid_conductor_order(self) -> "ConductorConfig":
        """Ensure that the training_order contains all the keys under the training field."""
        training_order = set(self.training_order)
        trainer_names = set(self.training.keys())

        if training_order != trainer_names:
            raise ValueError(
                "`training_order` needs to contain all the keys (trainer name) under the `training` field."
            )

        return self

    @classmethod
    @typecheck
    def from_yaml_file(cls, path: str | Path, dotpath: str | List[str] = []) -> ConductorConfig:
        """Create an instance of ConductorConfig from a yaml file at the given path."""
        config_dict = yaml_config_path_to_dict(path)
        config_dict = safe_deep_get(config_dict, dotpath)
        assert exists(config_dict), f'config not found at path {".".join(dotpath)}'

        return cls(**config_dict)

    def create_instance(self, trainer_name: str, **kwargs) -> Trainer:
        """Create an instance of Trainer from the configuration."""
        assert (
            trainer_name in self.training
        ), f"{trainer_name} not found among available trainers {tuple(self.training.keys())}."

        trainer_config = self.training[trainer_name]

        # nest the checkpoint_folder of the trainer within the main checkpoint_folder

        nested_checkpoint_folder = str(
            Path(self.checkpoint_folder) / Path(trainer_config.checkpoint_folder)
        )

        trainer_config.checkpoint_folder = nested_checkpoint_folder

        # prepend the main training checkpoint_prefix

        nested_checkpoint_prefix = self.checkpoint_prefix + trainer_config.checkpoint_prefix

        trainer_config.checkpoint_prefix = nested_checkpoint_prefix

        # create the Trainer, accounting for root level config

        trainer = trainer_config.create_instance(model=self.model, **kwargs)

        return trainer

    @classmethod
    def create_instance_from_yaml_file(
        cls, path: str | Path, dotpath: str | List[str] = [], **kwargs
    ) -> Trainer:
        """Create an instance of Trainer from a yaml file at the given path."""
        training_config = cls.from_yaml_file(path, dotpath)
        return training_config.create_instance(**kwargs)


# convenience functions

create_megafold_from_yaml = MegaFoldConfig.create_instance_from_yaml_file
create_trainer_from_yaml = TrainerConfig.create_instance_from_yaml_file
create_trainer_from_conductor_yaml = ConductorConfig.create_instance_from_yaml_file
