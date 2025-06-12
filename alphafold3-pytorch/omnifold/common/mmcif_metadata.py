"""MmCIF metadata."""

from importlib.metadata import version

import numpy as np
from beartype.typing import Mapping, Sequence

from omnifold.tensor_typing import typecheck
from omnifold.utils.utils import exists

omnifold_version = "1.0" # version("omnifold")

_DISCLAIMER = """THE INFORMATION IS NOT INTENDED FOR, HAS NOT BEEN VALIDATED FOR, AND IS NOT
APPROVED FOR CLINICAL USE. IT SHOULD NOT BE USED FOR CLINICAL PURPOSE OR RELIED
ON FOR MEDICAL OR OTHER PROFESSIONAL ADVICE. IT IS THEORETICAL MODELLING ONLY
AND CAUTION SHOULD BE EXERCISED IN ITS USE. IT IS PROVIDED "AS-IS" WITHOUT ANY
WARRANTY OF ANY KIND, WHETHER EXPRESSED OR IMPLIED. NO WARRANTY IS GIVEN THAT
USE OF THE INFORMATION SHALL NOT INFRINGE THE RIGHTS OF ANY THIRD PARTY."""

# Authors of the methods paper we reference in the mmCIF.
_MMCIF_PAPER_AUTHORS = (
    "University of Missouri",
    "Meta",
    "Seoul National University",
    "University of Adelaide",
    "University of Massachusetts",
    "University of Illinois",
    "Leipzig University Medical School",
    "University of Groningen",
    "Toyota Technological Institute",
    "University of North Carolina",
    "Pawsey Supercomputing Research Centre",
    "Flinders University",
)

# Authors of the mmCIF - we set them to be equal to the authors of the paper.
_MMCIF_AUTHORS = _MMCIF_PAPER_AUTHORS


@typecheck
def add_metadata_to_mmcif(
    old_cif: Mapping[str, Sequence[str]],
    insert_omnifold_mmcif_metadata: bool = True,
    model_number: int | None = None,
) -> Mapping[str, Sequence[str]]:
    """Adds OmniFold metadata in the given mmCIF."""
    cif = {}

    if insert_omnifold_mmcif_metadata:
        # ModelCIF conformation dictionary.
        cif["_audit_conform.dict_name"] = ["mmcif_ma.dic"]
        cif["_audit_conform.dict_version"] = ["1.4.5"]
        cif["_audit_conform.dict_location"] = [
            "https://raw.githubusercontent.com/ihmwg/ModelCIF/master/dist/" "mmcif_ma.dic"
        ]

        # License and disclaimer.
        cif["_pdbx_data_usage.id"] = ["1", "2"]
        cif["_pdbx_data_usage.type"] = ["license", "disclaimer"]
        cif["_pdbx_data_usage.details"] = [
            "MIT License",
            _DISCLAIMER,
        ]
        cif["_pdbx_data_usage.url"] = [
            "?",
            "?",
        ]

        # Structure author details.
        cif["_audit_author.name"] = []
        cif["_audit_author.pdbx_ordinal"] = []
        for author_index, author_name in enumerate(_MMCIF_AUTHORS, start=1):
            cif["_audit_author.name"].append(author_name)
            cif["_audit_author.pdbx_ordinal"].append(str(author_index))

        # Paper author details.
        cif["_citation_author.citation_id"] = []
        cif["_citation_author.name"] = []
        cif["_citation_author.ordinal"] = []
        for author_index, author_name in enumerate(_MMCIF_PAPER_AUTHORS, start=1):
            cif["_citation_author.citation_id"].append("primary")
            cif["_citation_author.name"].append(author_name)
            cif["_citation_author.ordinal"].append(str(author_index))

        # Paper citation details.
        cif["_citation.id"] = ["primary"]
        cif["_citation.title"] = [
            "OmniFold: Open-Source Biomolecular Structure Prediction for All"
        ]
        cif["_citation.journal_full"] = ["bioRxiv"]
        cif["_citation.journal_volume"] = ["?"]
        cif["_citation.page_first"] = ["?"]
        cif["_citation.page_last"] = ["?"]
        cif["_citation.year"] = ["2024"]
        cif["_citation.journal_id_ASTM"] = ["N/A"]
        cif["_citation.country"] = ["USA"]
        cif["_citation.journal_id_ISSN"] = ["N/A"]
        cif["_citation.journal_id_CSD"] = ["N/A"]
        cif["_citation.book_publisher"] = ["?"]
        cif["_citation.pdbx_database_id_PubMed"] = ["?"]
        cif["_citation.pdbx_database_id_DOI"] = ["?"]

        # Type of data in the dataset including data used in the model generation.
        cif["_ma_data.id"] = ["1"]
        cif["_ma_data.name"] = ["Model"]
        cif["_ma_data.content_type"] = ["model coordinates"]

    # Description of number of instances for each entity.
    cif["_ma_target_entity_instance.asym_id"] = old_cif["_struct_asym.id"]
    cif["_ma_target_entity_instance.entity_id"] = old_cif["_struct_asym.entity_id"]
    cif["_ma_target_entity_instance.details"] = ["."] * len(
        cif["_ma_target_entity_instance.entity_id"]
    )

    # Details about the target entities.
    cif["_ma_target_entity.entity_id"] = cif["_ma_target_entity_instance.entity_id"]
    cif["_ma_target_entity.data_id"] = ["1"] * len(cif["_ma_target_entity.entity_id"])
    cif["_ma_target_entity.origin"] = ["."] * len(cif["_ma_target_entity.entity_id"])

    if insert_omnifold_mmcif_metadata:
        # Details of the models being deposited.
        cif["_ma_model_list.ordinal_id"] = ["1"]
        cif["_ma_model_list.model_id"] = ["1"]
        cif["_ma_model_list.model_group_id"] = ["1"]
        cif["_ma_model_list.model_name"] = ["Top ranked model"]

        cif["_ma_model_list.model_group_name"] = [f"OmniFold v{omnifold_version}"]
        cif["_ma_model_list.data_id"] = ["1"]
        cif["_ma_model_list.model_type"] = ["Ab initio model"]

        # Software used.
        cif["_software.pdbx_ordinal"] = ["1"]
        cif["_software.name"] = ["OmniFold"]
        cif["_software.version"] = [f"v{omnifold_version}"]
        cif["_software.type"] = ["package"]
        cif["_software.description"] = ["Structure prediction"]
        cif["_software.classification"] = ["other"]
        cif["_software.date"] = ["?"]

        # Collection of software into groups.
        cif["_ma_software_group.ordinal_id"] = ["1"]
        cif["_ma_software_group.group_id"] = ["1"]
        cif["_ma_software_group.software_id"] = ["1"]

        # Method description to conform with ModelCIF.
        cif["_ma_protocol_step.ordinal_id"] = ["1", "2", "3"]
        cif["_ma_protocol_step.protocol_id"] = ["1", "1", "1"]
        cif["_ma_protocol_step.step_id"] = ["1", "2", "3"]
        cif["_ma_protocol_step.method_type"] = [
            "coevolution MSA",
            "template search",
            "modeling",
        ]

        # Details of the metrics use to assess model confidence.
        cif["_ma_qa_metric.id"] = ["1", "2"]
        cif["_ma_qa_metric.name"] = ["pLDDT", "pLDDT"]
        # Accepted values are distance, energy, normalised score, other, zscore.
        cif["_ma_qa_metric.type"] = ["pLDDT", "pLDDT"]
        cif["_ma_qa_metric.mode"] = ["global", "local"]
        cif["_ma_qa_metric.software_group_id"] = ["1", "1"]

        # Global model confidence metric value.
        cif["_ma_qa_metric_global.ordinal_id"] = ["1"]
        cif["_ma_qa_metric_global.model_id"] = ["1"]
        cif["_ma_qa_metric_global.metric_id"] = ["1"]
        global_plddt = np.mean([float(v) for v in old_cif["_atom_site.B_iso_or_equiv"]])
        cif["_ma_qa_metric_global.metric_value"] = [f"{global_plddt:.2f}"]

        # Local model confidence metric values.
        cif["_ma_qa_metric_local.ordinal_id"] = old_cif["_atom_site.id"]
        cif["_ma_qa_metric_local.model_id"] = [
            str(model_number) if exists(model_number) else "1"
        ] * len(old_cif["_atom_site.id"])
        cif["_ma_qa_metric_local.label_asym_id"] = old_cif["_atom_site.label_asym_id"]
        cif["_ma_qa_metric_local.label_seq_id"] = old_cif["_atom_site.label_seq_id"]
        cif["_ma_qa_metric_local.label_comp_id"] = old_cif["_atom_site.label_comp_id"]
        cif["_ma_qa_metric_local.metric_id"] = ["2"] * len(old_cif["_atom_site.id"])
        cif["_ma_qa_metric_local.metric_value"] = old_cif["_atom_site.B_iso_or_equiv"]

    cif["_atom_type.symbol"] = sorted(set(old_cif["_atom_site.type_symbol"]))

    return cif
