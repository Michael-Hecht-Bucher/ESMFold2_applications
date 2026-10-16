# /// script
# requires-python = "<=3.13"
# dependencies = [
#     "abnumber",
#     "esm",
# ]
# ///
"""
Local/Vast.ai adaptation of Biohub's binder design code for ESMFold2 and ESMC.

As described in [Language Modeling Materializes a World Model of Protein Biology](https://www.biorxiv.org/content/10.64898/2026.06.03.729735).

Upstream: Biohub/esm, cookbook/tutorials/binder_design.py
Pinned revision: "esm==3.4.1.post1 (PyPI)"
Copyright 2026 Chan Zuckerberg Biohub, Inc. (MIT License).
"""

import argparse
import csv
import io
import json
import logging
import math
import os
import random
import re
import shutil
import string
import time
from dataclasses import dataclass
from functools import cache, partial
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import biotite.structure
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile, set_structure as set_pdbx_structure

from esm.models.esmc import EsmcForMaskedLM, EsmcTokenizer
from esm.models.esmc.layers import EsmcUnifiedTransformerBlock as TransformerBlock
from esm.models.esmfold2.experimental import (
    EsmFold2ExperimentalModel,
    MSAEncoder as ESMFold2MSAEncoder,
)
from esm.models.esmfold2.layers import (
    BACKEND_CUEQ,
    BACKEND_FUSED,
    CUE_AVAILABLE,
    TRITON_KERNELS_AVAILABLE,
    PairUpdateBlock,
)
from esm.models.esmfold2.layers import _seed_context as seed_context

from esm.models.esmfold2 import (
    ELEMENT_NUMBER_TO_SYMBOL,
    LigandInput,
    Modification,
    ProteinInput,
    StructurePredictionInput,
    load_ccd,
    prepare_esmfold2_input,
)
from esm.models.esmfold2.constants import (
    MOL_TYPE_NONPOLYMER,
    PROTEIN_1TO3,
    PROTEIN_3TO1,
    RES_TYPE_TO_CCD,
)
from esm.utils.structure.mmcif_parsing import PLDDT_B_FACTOR_SCALE

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TrajectoryStep = dict[str, torch.Tensor | float]
Trajectory = dict[int, TrajectoryStep]
RUNNER_VERSION = "2026-09-01.ptm-ligand-token-aware"


# ---- Constants ----


# General
TOKENS = ["<pad>", "-"] + [RES_TYPE_TO_CCD[i] for i in range(2, 33)]
ELEMENTS = ["X"] * (max(ELEMENT_NUMBER_TO_SYMBOL) + 1)
ELEMENTS[0] = "<pad>"
for _atomic_num, _symbol in ELEMENT_NUMBER_TO_SYMBOL.items():
    ELEMENTS[_atomic_num] = _symbol[:1] + _symbol[1:].lower()
TOKEN_IDS = {token: idx for idx, token in enumerate(TOKENS)}
AA_DIMS = 20
# Cysteine index in the 20-dim AA space (TOKEN_IDS are offset by 2 for <pad> and -)
CYS_IDX = TOKEN_IDS[PROTEIN_1TO3["C"]] - 2
MUTABLE_TOKEN = "#"
# Contains AA chars at fixed positions and MUTABLE_TOKEN at mutable positions.
BinderPromptStr = str

# Design
LOSS_WEIGHTS = {"intra_contact": 0.5, "inter_contact": 0.5, "glob": 0.2, "epitope": 0.5}
STEPS = 150
LOG_INTERVAL = 5
LEARNING_RATE = 0.1
TEMPERATURE_MIN = 1e-2
#ESMC_MASK_FRACTION = 0.15 # no longer used - can be removed
# Number of masked sequences processed simultaneously by ESMC.
LM_LOSS_BATCH_SIZE = 4
# Seven approximately 1/7-sized groups cover every mutable position once.
LM_MASK_PASSES = 7
CHECKPOINT_LM = True
BYTES_PER_GIB = 1024**3
COMPILE = False
# NOTE - This significantly reduces VRAM usage.
# On config (target_name="cd45", binder_name="trastuzumab_framework_vhvl", batch_size=1)
# this reduces VRAM from 51GB -> 27GB.  And enables increasing batch size up to 6.
# We are testing this setting in silico, and may change the default to True, in the future.
REUSE_ESMC = True

# Phosphorylated residue CCD codes used by the PDB Chemical Component Dictionary.
PHOSPHO_CCD_BY_RESIDUE = {"S": "SEP", "T": "TPO", "Y": "PTR"}

# Gradient weights for different types of binders.
ANTIBODY_PLM_GRAD_WEIGHT = 0.05
MINIBINDER_PLM_GRAD_WEIGHT = 0.50

# ---- Prompts ----


@dataclass(frozen=True)
class PromptFactory:
    """A simple factory for making binder prompt strings."""

    name: str
    template: str  # string with format fields
    length_ranges: dict[str, tuple[int, int]]  # map from field name tp length range
    is_antibody: bool  # Used to set LM loss weight for antibodies.

    def sample(self, seed: int) -> BinderPromptStr:
        random.seed(seed)
        return self.template.format(
            **{
                key: MUTABLE_TOKEN * random.randint(low, high)
                for key, (low, high) in self.length_ranges.items()
            }
        )


# fmt: off
BINDER_PROMPT_FACTORIES = {
    "minibinder": PromptFactory(name="minibinder", template="{seq}", length_ranges={"seq": (60, 200)}, is_antibody=False),
    "trastuzumab_framework_vhvl": PromptFactory(
        name="trastuzumab_framework_vhvl",
        template="EVQLVESGGGLVQPGGSLRLSCAAS{hcdr1}YIHWVRQAPGKGLEWVARI{hcdr2}TRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSR{hcdr3}WGQGTLVTVSSGGGSGGGSGGGSGGGSDIQMTQSPSSLSASVGDRVTITC{lcdr1}WYQQKPGKAPKLLIY{lcdr2}GVPSRFSGSRSGTDFTLTISSLQPEDFATYYC{lcdr3}FGQGTKVEIK",
        length_ranges = {"hcdr1": (7, 9), "hcdr2": (5, 6), "hcdr3": (9, 15), "lcdr1": (11, 16), "lcdr2": (7, 7), "lcdr3": (9, 9)},
        is_antibody=True,
    ),
    "atezolizumab_framework_vhvl": PromptFactory(
        name="atezolizumab_framework_vhvl",
        template="EVQLVESGGGLVQPGGSLRLSCAAS{hcdr1}WIHWVRQAPGKGLEWVAWI{hcdr2}TYYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCAR{hcdr3}WGQGTLVTVSSGGGSGGGSGGGSGGGSDIQMTQSPSSLSASVGDRVTITC{lcdr1}WYQQKPGKAPKLLIY{lcdr2}GVPSRFSGSGSGTDFTLTISSLQPEDFATYYC{lcdr3}FGQGTKVEIK",
        length_ranges = {"hcdr1": (7, 9), "hcdr2": (5, 6), "hcdr3": (9, 15), "lcdr1": (11, 16), "lcdr2": (7, 7), "lcdr3": (9, 9)},
        is_antibody=True,
    ),
    "ocankitug_framework_vhvl": PromptFactory(
        name="ocankitug_framework_vhvl",
        template="QVQLVQSGAEVKKPGSSVKVSCKAS{hcdr1}WMHWVRQAPGQGLEWMGII{hcdr2}TSLNQKFQGRVTITADTSTSTAYMELSSLRSEDTAVYYCAR{hcdr3}WGQGTLVTVSSGGGSGGGSGGGSGGGSDIQMTQSPSSLSASVGDRVTITC{lcdr1}WYQQKPGKAPKLLIY{lcdr2}GVPSRFSGSGSGTDFTLTISSLQPEDFATYYC{lcdr3}FGQGTKVEIK",
        length_ranges = {"hcdr1": (7, 9), "hcdr2": (5, 6), "hcdr3": (8, 14), "lcdr1": (11, 16), "lcdr2": (7, 7), "lcdr3": (9, 9)},
        is_antibody=True,
    )
}


TARGET_SEQUENCES = {
    # https://www.uniprot.org/uniprotkb/P08575  389-574
    "cd45": "GSPGEPQIIFCRSEAAHQGVITWNPPQRSFHNFTLCYIKETEKDCLNLDKNLIKYDLQNLKPYTKYVLSLHAYIIAKVQRNGSAAMCHFTTKSAPPSQVWNMTVSMTSDNSMHVKCRPPRDRNGPHERYHLEVEAGNTLVRNESHKNCDFRVKDLQYSTDYTFKAYFHNGDYPGEPFILHHSTSY",
    # https://www.uniprot.org/uniprotkb/P16410  37-155
    "ctla4": "MHVAQPAVVLASSRGIASFVCEYASPGKATEVRVTVLRQADSQVTEVCAATYMMGNELTFLDDSICTGTSSGNQVNLTIQGLRAMDTGLYICKVELMYPPPYYLGIGNGTQIYVIDPE",
    # https://www.uniprot.org/uniprotkb/P00533  333-524
    "egfr": "RKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCV",
    # https://www.uniprot.org/uniprotkb/Q9NZQ7  17-132
    "pd-l1": "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNA",
    # https://www.uniprot.org/uniprotkb/P09619  125-312
    "pdgfr": "GFLPNDAEELFIFLTEITEITIPCRVTDPQLVVTLHEKKGDVALPVPYDHQRGFSGIFEDRSYICKTTIGDREVDSDAYYVYRLQVSSINVSVNAVQTVVRQGENITLMCIVIGNEVVNFEWTYPRKESGRLVEPVTDFLLDMPYHIRSILHIPSAELEDSGTYTCNVTESVNDHQDEKAINITVVE",
}
# fmt: on


# ---- Helper functions ----


def build_initial_soft_sequence_logits(sequence: str, batch_size: int) -> torch.Tensor:
    """
    Initialize logits with:
    - High confidence (10.0) for fixed positions
    - Random (~0) for mutable positions
    - -1e6 for cysteines
    """
    if all(aa == MUTABLE_TOKEN for aa in sequence):
        logits = 0.01 * torch.randn([batch_size, len(sequence), AA_DIMS])
        logits[:, :, CYS_IDX] = -1e6  # remove cysteines
    else:
        logits = torch.zeros([batch_size, len(sequence), AA_DIMS])
        for i, aa in enumerate(sequence):
            if aa == MUTABLE_TOKEN:  # mutable position - random
                logits[:, i, :] = 0.01 * torch.randn(batch_size, AA_DIMS)
                logits[:, i, CYS_IDX] = -1e6
            else:  # fixed position
                assert aa in PROTEIN_1TO3, aa
                token_id = TOKEN_IDS[PROTEIN_1TO3[aa]]
                logits[:, i, token_id - 2] = 10.0

    return logits.requires_grad_(True)


def build_gradient_mask(sequence: str, batch_size: int) -> torch.Tensor:
    """
    Build gradient mask [B, L, V]:
    - 0 for fixed (all amino acids)
    - 0 for cysteine at all positions
    - 1 for non-cysteine amino acids at mutable positions
    """
    mask = torch.ones([batch_size, len(sequence), AA_DIMS])
    fixed_positions = [i for i, aa in enumerate(sequence) if aa != MUTABLE_TOKEN]
    mask[:, fixed_positions, :] = 0.0
    mask[:, :, CYS_IDX] = 0.0
    return mask


def sequence_to_one_hot(sequence: str, device="cuda") -> torch.Tensor:
    """Convert target string to one-hot tensor [1, L_target, num_tokens]."""

    const_dict = {token: i for i, token in enumerate(TOKENS)}
    target_index = [const_dict[PROTEIN_1TO3[letter]] for letter in sequence]
    one_hot = F.one_hot(torch.tensor(target_index), num_classes=len(TOKENS))
    return one_hot.to(device).unsqueeze(0).float()


def parse_target_modification(spec: str, target_sequence: str) -> Modification:
    """Parse a user-facing PTM specification into a zero-indexed Modification.

    Accepted forms are ``S123`` (phosphorylation inferred from the residue),
    ``123:SEP``, and ``S123:SEP``. Sequence positions in the CLI are 1-indexed.
    """

    match = re.fullmatch(
        r"(?:(?P<residue>[A-Za-z]))?(?P<position>[1-9][0-9]*)(?::(?P<ccd>[A-Za-z0-9]+))?",
        spec.strip(),
    )
    if match is None:
        raise ValueError(
            f"Invalid modification {spec!r}. Use S123, 123:SEP, or S123:SEP."
        )

    position_1based = int(match.group("position"))
    if not 1 <= position_1based <= len(target_sequence):
        raise ValueError(
            f"Modification {spec!r} is outside target length {len(target_sequence)}."
        )

    sequence_residue = target_sequence[position_1based - 1]
    stated_residue = (match.group("residue") or sequence_residue).upper()
    if stated_residue != sequence_residue:
        raise ValueError(
            f"Modification {spec!r} does not match the target: position "
            f"{position_1based} is {sequence_residue}, not {stated_residue}."
        )

    ccd = match.group("ccd")
    if ccd is None:
        try:
            ccd = PHOSPHO_CCD_BY_RESIDUE[sequence_residue]
        except KeyError as exc:
            raise ValueError(
                f"Cannot infer phosphorylation for residue {sequence_residue!r}; "
                "specify a CCD code explicitly, for example 123:MSE."
            ) from exc

    return Modification(position=position_1based - 1, ccd=ccd.upper())


def modification_hotspot_ids(
    target_sequence: str,
    target_modifications: list[Modification] | None,
    radius: int,
) -> list[str]:
    """Return ESMFold2 binder-design hotspot IDs around all target PTMs."""

    if radius < 0:
        raise ValueError("hotspot radius must be >= 0")
    hotspot_ids: list[str] = []
    for modification in target_modifications or []:
        start = max(0, modification.position - radius)
        stop = min(len(target_sequence), modification.position + radius + 1)
        hotspot_ids.extend(
            f"{target_sequence[index]}{index + 1}" for index in range(start, stop)
        )
    return list(dict.fromkeys(hotspot_ids))


def get_mid_points() -> torch.Tensor:
    """128 distance bin midpoints (2p-52 Angstrom range)."""

    boundaries = torch.linspace(2, 52.0, 127)
    lower = torch.tensor([1.0])
    upper = torch.tensor([52.0 + 5.0])
    exp_boundaries = torch.cat((lower, boundaries, upper))
    return (exp_boundaries[:-1] + exp_boundaries[1:]) / 2


def binned_entropy(
    dgram: torch.Tensor, bin_distance: torch.Tensor, cutoff: float
) -> torch.Tensor:
    """Entropy of distance distribution within cutoff (design losses only)."""

    bin_mask = ~(bin_distance < cutoff)
    masked_dgram = dgram - (1e7 * bin_mask)
    px = torch.softmax(masked_dgram, dim=-1)
    log_px = torch.log_softmax(dgram, dim=-1)
    return -(px * log_px).sum(-1)


def masked_min_k(x: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """Mean of the smallest k values in x under mask along the last dimension."""

    mask = mask.bool()
    y = torch.sort(torch.where(mask, x, float("nan")))[0]
    k_mask = (torch.arange(y.shape[-1]).to(y.device) < k) & (~torch.isnan(y))
    return torch.where(k_mask, y, 0).sum(-1) / (k_mask.sum(-1) + 1e-8)


def masked_average(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean along last axis."""

    mask = mask.bool()
    return torch.where(mask, x, 0).sum(-1) / (torch.where(mask, 1, 0).sum(-1) + 1e-8)


# ---- Loss functions ----


def compute_contact_loss(
    distogram_logits: torch.Tensor,
    bin_distance: torch.Tensor,
    num_contacts: int,
    min_sep: int,
    cutoff: float,
    chain_mask: torch.Tensor,
    binder_mask: torch.Tensor,
) -> torch.Tensor:
    """Algorithm 12 Contact Losses.

    Entropy-based contact loss with sequence separation constraint."""

    con_loss = binned_entropy(distogram_logits, bin_distance, cutoff)
    position = torch.arange(distogram_logits.shape[1])
    p_dist = position[:, None] - position[None, :]
    if min_sep > 0:
        separation_mask = (torch.abs(p_dist) >= min_sep).to(distogram_logits.device)
        binder_mask = torch.logical_and(separation_mask, binder_mask)
    per_residue = masked_min_k(con_loss, mask=binder_mask, k=num_contacts).to(
        distogram_logits.device
    )
    return masked_average(per_residue, mask=chain_mask).to(distogram_logits.device)


def compute_intra_contact_loss(
    distogram_logits: torch.Tensor,
    binder_token_mask: torch.Tensor,
    bin_distance: torch.Tensor,
) -> torch.Tensor:
    """Binder internal contacts (k=2, min_sep=9, cutoff=14A)."""

    is_binder = binder_token_mask.to(
        device=distogram_logits.device, dtype=torch.bool
    )
    return compute_contact_loss(
        distogram_logits,
        bin_distance,
        num_contacts=2,
        min_sep=9,
        cutoff=14.0,
        chain_mask=is_binder,
        binder_mask=is_binder,
    )


def compute_inter_contact_loss(
    distogram_logits: torch.Tensor,
    target_token_mask: torch.Tensor,
    binder_token_mask: torch.Tensor,
    bin_distance: torch.Tensor,
) -> torch.Tensor:
    """Protein-target/binder interface, explicitly excluding ligand tokens."""

    is_target = target_token_mask.to(
        device=distogram_logits.device, dtype=torch.bool
    )
    is_binder = binder_token_mask.to(
        device=distogram_logits.device, dtype=torch.bool
    )
    return compute_contact_loss(
        distogram_logits,
        bin_distance,
        num_contacts=1,
        min_sep=0,
        cutoff=22.0,
        chain_mask=is_target,
        binder_mask=is_binder,
    )


def compute_globularity_loss(
    distogram_logits: torch.Tensor,
    binder_token_mask: torch.Tensor,
    bin_distance: torch.Tensor,
) -> torch.Tensor:
    """Algorithm 13 Globularity Loss.

    Radius of gyration vs theoretical packed protein."""

    binder_indices = torch.nonzero(
        binder_token_mask.to(device=distogram_logits.device, dtype=torch.bool),
        as_tuple=False,
    ).flatten()
    binder_disto = distogram_logits.index_select(1, binder_indices).index_select(
        2, binder_indices
    )
    n = binder_disto.shape[1]
    disto_probs = torch.softmax(binder_disto, dim=-1)
    bin_distance = bin_distance.clamp(max=27)
    e_sq_dist = torch.sum(disto_probs * torch.square(bin_distance), dim=-1)
    sum_sq_dist = torch.sum(torch.tril(e_sq_dist, diagonal=-1), dim=(1, 2))
    rg_term = torch.sqrt(sum_sq_dist / (n * n))
    rg_th = 2.38 * (n**0.365)
    return F.elu(rg_term - rg_th)


def get_epitope_mask(
    target_sequence: str,
    binder_length: int,
    target_hotspot_ids: list[str],
    batch_size: int,
    device: torch.device,
    target_token_residue_index: torch.Tensor | None = None,
    target_token_indices: torch.Tensor | None = None,
    full_token_count: int | None = None,
) -> torch.Tensor:
    """Full-complex token mask for sequence-local target hotspot residue IDs.

    ESMFold2 represents an ordinary amino acid with one model token, but may
    represent a CCD modification with several tokens.  ``target_sequence`` and
    the command-line hotspot IDs remain residue based.  When
    ``target_token_residue_index`` is supplied, every token belonging to the
    selected biological residue is marked.  This is what lets an epitope loss
    address the complete TPO/SEP/PTR/M3L chemical group instead of only one
    arbitrarily selected token.
    """
    target_residue_count = len(target_sequence)
    if target_token_residue_index is None:
        target_token_residue_index = torch.arange(
            target_residue_count, dtype=torch.long, device=device
        )
    else:
        target_token_residue_index = target_token_residue_index.to(
            device=device, dtype=torch.long
        )
        if target_token_residue_index.ndim != 1:
            raise ValueError(
                "target_token_residue_index must be one-dimensional, got "
                f"{tuple(target_token_residue_index.shape)}."
            )
        invalid = (target_token_residue_index < 0) | (
            target_token_residue_index >= target_residue_count
        )
        if bool(invalid.any()):
            bad_values = torch.unique(target_token_residue_index[invalid]).tolist()
            raise ValueError(
                "Target token-to-residue mapping contains indices outside "
                f"0..{target_residue_count - 1}: {bad_values}."
            )

    target_token_count = int(target_token_residue_index.numel())
    if target_token_indices is None:
        target_token_indices = torch.arange(
            target_token_count, dtype=torch.long, device=device
        )
        full_token_count = target_token_count + binder_length
    else:
        target_token_indices = target_token_indices.to(device=device, dtype=torch.long)
        if target_token_indices.ndim != 1:
            raise ValueError("target_token_indices must be one-dimensional.")
        if target_token_indices.numel() != target_token_count:
            raise ValueError(
                "target_token_indices and target_token_residue_index must have "
                "the same number of entries."
            )
        if full_token_count is None:
            raise ValueError(
                "full_token_count is required with explicit target_token_indices."
            )
        if bool((target_token_indices < 0).any()) or bool(
            (target_token_indices >= full_token_count).any()
        ):
            raise ValueError("A target token index is outside the full complex.")

    assert full_token_count is not None
    epitope_mask = torch.zeros(full_token_count, dtype=torch.bool, device=device)
    for hotspot_id in target_hotspot_ids:
        assert len(hotspot_id) >= 2, (
            f"Hotspot residue ID {hotspot_id!r} must have form <residue><idx>, "
            "for example L150."
        )
        residue = hotspot_id[0]
        hotspot_idx_text = hotspot_id[1:]
        assert residue in PROTEIN_1TO3, (
            f"Hotspot residue ID {hotspot_id!r} starts with unknown residue "
            f"{residue!r}."
        )
        assert hotspot_idx_text.isdecimal(), (
            f"Hotspot residue ID {hotspot_id!r} must have form <residue><idx>, "
            "for example L150."
        )
        hotspot_idx = int(hotspot_idx_text)
        assert 1 <= hotspot_idx <= target_residue_count, (
            f"Hotspot residue {hotspot_id!r} is out of range for target sequence "
            f"length {target_residue_count}. Hotspots are 1-indexed within the provided "
            "target sequence."
        )
        target_residue = target_sequence[hotspot_idx - 1]
        assert target_residue == residue, (
            f"Hotspot residue ID {hotspot_id!r} does not match target sequence: "
            f"target residue at 1-indexed position {hotspot_idx} is "
            f"{target_residue!r}."
        )
        token_mask = target_token_residue_index == (hotspot_idx - 1)
        if not bool(token_mask.any()):
            raise ValueError(
                f"No ESMFold2 model token maps to hotspot {hotspot_id!r}."
            )
        epitope_mask[target_token_indices[token_mask]] = True
    return epitope_mask[None].repeat(batch_size, 1)


def compute_epitope_loss(
    distogram_logits: torch.Tensor,
    epitope_mask: torch.Tensor,
    binder_token_mask: torch.Tensor,
    bin_distance: torch.Tensor,
    epitope_contact_distance: float = 12.0,
) -> torch.Tensor:
    """Encourage binder contacts to the requested target hotspot residues.

    Converts full-complex distogram logits into contact probabilities below
    ``epitope_contact_distance``, then penalizes each hotspot by the best few
    binder contacts. Explicit token masks keep intervening ligand entities out
    of this loss.

    Parameters
    ----------
    distogram_logits
        Full-complex distogram logits with shape ``(B, L, L, num_bins)``.
    epitope_mask
        Boolean full-complex mask with shape ``(B, L)`` and true values at
        target hotspot positions.
    binder_token_mask
        Boolean full-complex mask selecting only designed binder tokens.
    bin_distance
        Distance represented by each distogram bin.
    epitope_contact_distance
        Distance threshold used to define a hotspot-binder contact.

    Returns
    -------
    torch.Tensor
        Per-example epitope losses with shape ``(B,)``.
    """

    B, L, L2, _ = distogram_logits.shape
    assert L == L2, (L, L2)
    binder_mask = binder_token_mask.to(
        device=distogram_logits.device, dtype=torch.bool
    )
    if binder_mask.ndim == 1:
        binder_mask = binder_mask[None].expand(B, -1)

    positive_mask = bin_distance < epitope_contact_distance
    assert binder_mask.shape == (B, L), f"Binder mask: {binder_mask.shape} != {B, L}"
    assert epitope_mask.shape == (B, L), f"Epitope mask: {epitope_mask.shape} != {B, L}"
    # NOTE - this could be made bidirectional.
    epitope_binder_mask = epitope_mask[:, :, None] & binder_mask[:, None, :]

    probabilities = torch.softmax(distogram_logits, dim=-1)
    positive_probs = torch.sum(probabilities * positive_mask, dim=-1)
    contact_measure = -torch.log(positive_probs + 1e-8)

    epitope_loss = masked_min_k(contact_measure, mask=epitope_binder_mask, k=5)
    return masked_average(epitope_loss, mask=epitope_mask)


def compute_structure_losses(
    distogram_logits: torch.Tensor,
    binder_length: int,
    target_sequence: str | None = None,
    target_hotspot_ids: list[str] | None = None,
    epitope_contact_distance: float = 12.0,
    target_token_residue_index: torch.Tensor | None = None,
    target_token_indices: torch.Tensor | None = None,
    target_token_mask: torch.Tensor | None = None,
    binder_token_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute losses for the target protein/binder interface only."""

    bin_distance = get_mid_points().to(distogram_logits.device)
    full_token_count = distogram_logits.shape[1]
    if binder_token_mask is None:
        binder_token_mask = torch.zeros(
            full_token_count, dtype=torch.bool, device=distogram_logits.device
        )
        binder_token_mask[-binder_length:] = True
    else:
        binder_token_mask = binder_token_mask.to(
            device=distogram_logits.device, dtype=torch.bool
        )
    if int(binder_token_mask.sum()) != binder_length:
        raise ValueError(
            f"Binder mask selects {int(binder_token_mask.sum())} tokens, expected "
            f"{binder_length}."
        )
    if target_token_mask is None:
        target_token_mask = ~binder_token_mask
    else:
        target_token_mask = target_token_mask.to(
            device=distogram_logits.device, dtype=torch.bool
        )
    if target_token_mask.shape != (full_token_count,) or binder_token_mask.shape != (
        full_token_count,
    ):
        raise ValueError("Target and binder masks must span the full complex.")
    if bool((target_token_mask & binder_token_mask).any()):
        raise ValueError("Target and binder token masks overlap.")

    losses: dict[str, torch.Tensor] = {}
    losses["intra_contact_loss"] = compute_intra_contact_loss(
        distogram_logits, binder_token_mask, bin_distance
    )
    losses["inter_contact_loss"] = compute_inter_contact_loss(
        distogram_logits, target_token_mask, binder_token_mask, bin_distance
    )
    losses["glob_loss"] = compute_globularity_loss(
        distogram_logits, binder_token_mask, bin_distance
    )
    B = distogram_logits.size(0)
    total = torch.tensor([0.0] * B, device=distogram_logits.device, requires_grad=True)
    total = total + LOSS_WEIGHTS["intra_contact"] * losses["intra_contact_loss"]
    total = total + LOSS_WEIGHTS["inter_contact"] * losses["inter_contact_loss"]
    total = total + LOSS_WEIGHTS["glob"] * losses["glob_loss"]
    if target_hotspot_ids is not None:
        target_token_count = int(target_token_mask.sum())
        assert (
            target_sequence is not None
        ), "target_sequence is required when target_hotspot_ids is provided."
        assert (
            "|" not in target_sequence
        ), "Epitope loss only supports one target chain."
        if target_token_residue_index is None:
            assert len(target_sequence) == target_token_count, (
                f"Target sequence length {len(target_sequence)} does not match "
                f"distogram target-token length {target_token_count}; a PTM-aware "
                "target_token_residue_index mapping is required."
            )
        else:
            assert target_token_residue_index.numel() == target_token_count, (
                "Target token-to-residue mapping length "
                f"{target_token_residue_index.numel()} does not match distogram "
                f"target-token length {target_token_count}."
            )
        epitope_mask = get_epitope_mask(
            target_sequence=target_sequence,
            binder_length=binder_length,
            target_hotspot_ids=target_hotspot_ids,
            batch_size=B,
            device=distogram_logits.device,
            target_token_residue_index=target_token_residue_index,
            target_token_indices=target_token_indices,
            full_token_count=full_token_count,
        )
        losses["epitope_loss"] = compute_epitope_loss(
            distogram_logits=distogram_logits,
            epitope_mask=epitope_mask,
            binder_token_mask=binder_token_mask,
            bin_distance=bin_distance,
            epitope_contact_distance=epitope_contact_distance,
        )
        total = total + LOSS_WEIGHTS["epitope"] * losses["epitope_loss"]
    losses["total_loss"] = total
    return losses

def compute_sequence_quality_losses(
    binder_design: torch.Tensor,
    window_size: int = 12,
    min_window_neff: float = 4.0,
    max_residue_fraction: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Differentiable penalties against local collapse and short repeats."""

    # Hard sequence in the forward pass, soft gradient in the backward pass.
    hard = F.one_hot(
        binder_design.argmax(dim=-1),
        num_classes=binder_design.size(-1),
    ).to(binder_design.dtype)
    p = binder_design + (hard - binder_design).detach()

    # Penalize the worst 10% of sliding windows below target N_eff.
    window_freq = F.avg_pool1d(
        p.transpose(1, 2),
        kernel_size=window_size,
        stride=1,
    ).transpose(1, 2)

    window_entropy = -(
        window_freq * window_freq.clamp_min(1e-8).log()
    ).sum(dim=-1)

    window_violation = F.relu(
        math.log(min_window_neff) - window_entropy
    ).square()

    k = max(1, math.ceil(0.10 * window_violation.size(1)))
    local_complexity_loss = (
        window_violation.topk(k, dim=1).values.mean(dim=1)
    )

    # Penalize any single residue occupying more than 25% of the chain.
    global_freq = p.mean(dim=1)
    composition_loss = (
        F.relu(global_freq - max_residue_fraction).square().sum(dim=-1)
    )

    # Catch homopolymers and periodic patterns such as GNGNGN or ITITIT.
    repeat_loss = torch.zeros(
        p.size(0), device=p.device, dtype=p.dtype
    )
    for lag, weight in ((1, 1.0), (2, 0.5), (3, 0.25)):
        same_fraction = (
            p[:, lag:] * p[:, :-lag]
        ).sum(dim=-1).mean(dim=-1)

        repeat_loss += weight * F.relu(
            same_fraction - 0.10
        ).square()

    total = (
        local_complexity_loss
        + 10.0 * composition_loss
        + repeat_loss
    )

    return {
        "local_complexity_loss": local_complexity_loss,
        "composition_loss": composition_loss,
        "repeat_loss": repeat_loss,
        "sequence_quality_loss": total,
    }

# ---- Distogram iptm proxy ----


def _binding_confidence_entropy(
    dgram: torch.Tensor, bin_distance: torch.Tensor, cutoff: float
) -> torch.Tensor:
    """Pair entropy within cutoff."""

    probs = torch.softmax(dgram, dim=-1)
    cutoff_mask = bin_distance < cutoff
    p_cut = probs[..., cutoff_mask]
    p_cut = p_cut / (p_cut.sum(-1, keepdim=True) + 1e-8)
    return -(p_cut * torch.log(p_cut + 1e-10)).sum(-1)


def _entropy_to_confidence(mean_entropy: float) -> float:
    """Map mean pair entropy to [0, 1]; lower entropy → higher score."""
    return float(max(0.0, min(1.0, 1.0 - mean_entropy / math.log(51))))


def _is_antibody_sequence(binder_sequence: str) -> bool:
    """Return True if ANARCI recognizes binder_sequence as antibody variable domain(s)."""
    from abnumber.common import _anarci_align

    sequence = binder_sequence.replace(MUTABLE_TOKEN, "A")
    result = _anarci_align(
        sequences=[sequence], scheme="chothia", allowed_species=None
    )[0]
    if not result:
        return False
    valid_chain_types = {"H", "K", "L"}
    return all(chain_type in valid_chain_types for _, chain_type, *_ in result)


def _cdr_indices(binder_sequence: str) -> list[int]:
    """0-based binder indices for all Chothia CDRs."""
    from abnumber import Chain
    from abnumber.common import _anarci_align

    result = _anarci_align(
        sequences=[binder_sequence], scheme="chothia", allowed_species=None
    )[0]
    chains = [
        Chain("".join(result[i][0].values()), scheme="chothia")
        for i in range(len(result))
    ]
    if len(chains) == 2 and not chains[0].is_heavy_chain():
        chains.reverse()
    indices: list[int] = []
    for chain in chains:
        for cdr in (chain.cdr1_seq, chain.cdr2_seq, chain.cdr3_seq):
            start = binder_sequence.find(cdr)
            assert start >= 0
            indices.extend(range(start, start + len(cdr)))
    return indices


def compute_distogram_iptm_proxy(
    distogram_logits: torch.Tensor,
    target_length: int,
    binder_sequence: str,
    is_antibody: bool,
) -> dict[str, float]:
    """Algorithm 15 Distogram ipTM Proxy.

    Distogram iptm proxy for a target|binder complex (binder at suffix).

    Returns distogram_iptm_proxy for all designs and
    cdr_distogram_iptm_proxy when the binder can be annotated as an
    antibody; otherwise the CDR score is NaN.
    """
    if distogram_logits.ndim == 4:
        distogram_logits = distogram_logits[0]

    binder_length = len(binder_sequence)
    assert distogram_logits.shape[0] == target_length + binder_length

    bin_distance = get_mid_points().to(distogram_logits.device)
    binder_start = target_length

    def _mean_lowest_k(entropies: torch.Tensor, k: int) -> float:
        sorted_entropies, _ = torch.sort(entropies.reshape(-1))
        k = min(k, sorted_entropies.numel())
        return float(sorted_entropies[:k].mean())

    binder_to_target_entropy = _binding_confidence_entropy(
        distogram_logits[binder_start:, :target_length, :], bin_distance, cutoff=22.0
    )
    distogram_iptm_proxy = _entropy_to_confidence(
        _mean_lowest_k(binder_to_target_entropy, k=binder_length)
    )

    if not is_antibody:
        cdr_distogram_iptm_proxy = float("nan")
    else:
        cdr_indices = _cdr_indices(binder_sequence)
        cdr_rows = [binder_start + i for i in cdr_indices]
        cdr_to_target_entropy = _binding_confidence_entropy(
            distogram_logits[cdr_rows, :target_length, :], bin_distance, cutoff=22.0
        )
        cdr_distogram_iptm_proxy = _entropy_to_confidence(
            _mean_lowest_k(cdr_to_target_entropy, k=len(cdr_indices))
        )

    return {
        "distogram_iptm_proxy": distogram_iptm_proxy,
        "cdr_distogram_iptm_proxy": cdr_distogram_iptm_proxy,
    }


# ---- Folding ----


def _resize_tensor(tensor: torch.Tensor, *, dim: int, size: int) -> torch.Tensor:
    current = tensor.shape[dim]
    if current >= size:
        return tensor.narrow(dim, 0, size)

    pad_shape = list(tensor.shape)
    pad_shape[dim] = size - current
    pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, pad), dim=dim)


_ATOM_FEATURE_DIMS = {
    "ref_pos": 0,
    "ref_element": 0,
    "ref_charge": 0,
    "ref_atom_name_chars": 0,
    "ref_space_uid": 0,
    "atom_attention_mask": 0,
    "atom_to_token": 0,
    "is_resolved": 0,
    "gt_coords": 1,
}


@dataclass(frozen=True)
class ComplexTokenLayout:
    """Explicit target-protein, ligand, and binder token partitions."""

    target_token_count: int
    target_token_indices: torch.Tensor
    target_token_residue_index: torch.Tensor
    target_representative_token_index: torch.Tensor
    ligand_token_indices: torch.Tensor
    binder_token_indices: torch.Tensor

    @property
    def full_token_count(self) -> int:
        return int(
            self.target_token_indices.numel()
            + self.ligand_token_indices.numel()
            + self.binder_token_indices.numel()
        )

    def target_token_mask(self, device: torch.device | None = None) -> torch.Tensor:
        mask = torch.zeros(self.full_token_count, dtype=torch.bool, device=device)
        mask[self.target_token_indices.to(device=mask.device)] = True
        return mask

    def binder_token_mask(self, device: torch.device | None = None) -> torch.Tensor:
        mask = torch.zeros(self.full_token_count, dtype=torch.bool, device=device)
        mask[self.binder_token_indices.to(device=mask.device)] = True
        return mask

    def ligand_token_mask(self, device: torch.device | None = None) -> torch.Tensor:
        mask = torch.zeros(self.full_token_count, dtype=torch.bool, device=device)
        mask[self.ligand_token_indices.to(device=mask.device)] = True
        return mask


def _decode_atom_name(atom_name_chars: torch.Tensor) -> str:
    return "".join(chr(int(value) + 32) for value in atom_name_chars if int(value) != 0)


def _infer_complex_token_layout(
    features: dict[str, torch.Tensor],
    *,
    target_residue_count: int,
    binder_residue_count: int,
    ligand_entity_count: int = 0,
) -> ComplexTokenLayout:
    """Infer the biological-residue mapping from ESMFold2's own features.

    The all-atom input builder is the source of truth.  In particular, a
    modified target residue and a ligand may each occupy several atom-level
    model tokens. The target must be the first entity and the binder the last;
    any intervening entities are treated only as fixed ligands.
    """
    required = {
        "res_type",
        "asym_id",
        "residue_index",
        "atom_to_token",
        "ref_atom_name_chars",
        "atom_attention_mask",
    }
    missing = sorted(required - features.keys())
    if missing:
        raise KeyError(
            "ESMFold2 input is missing features required for PTM-aware token "
            f"mapping: {', '.join(missing)}."
        )

    res_type = features["res_type"]
    asym_id = features["asym_id"]
    residue_index = features["residue_index"].to(dtype=torch.long)
    if res_type.ndim != 1 or asym_id.ndim != 1 or residue_index.ndim != 1:
        raise ValueError(
            "Expected unbatched one-dimensional res_type, asym_id, and "
            "residue_index features."
        )
    token_count = int(res_type.numel())
    if asym_id.numel() != token_count or residue_index.numel() != token_count:
        raise ValueError("ESMFold2 token-level feature lengths are inconsistent.")
    if binder_residue_count < 1 or binder_residue_count >= token_count:
        raise ValueError(
            f"Invalid binder length {binder_residue_count} for {token_count} tokens."
        )

    target_asym_id = asym_id[0]
    binder_asym_id = asym_id[-1]
    if bool(target_asym_id == binder_asym_id):
        raise ValueError("Target and binder unexpectedly share an asym_id.")
    target_mask = asym_id == target_asym_id
    binder_mask = asym_id == binder_asym_id
    target_positions = torch.nonzero(target_mask, as_tuple=False).flatten()
    binder_positions = torch.nonzero(binder_mask, as_tuple=False).flatten()
    ligand_mask = ~(target_mask | binder_mask)
    ligand_positions = torch.nonzero(ligand_mask, as_tuple=False).flatten()

    unique_ligand_asym_ids = torch.unique(asym_id[ligand_mask])
    if unique_ligand_asym_ids.numel() != ligand_entity_count:
        raise ValueError(
            "ESMFold2 token layout contains "
            f"{unique_ligand_asym_ids.numel()} intervening entities, expected "
            f"{ligand_entity_count} ligand entities."
        )
    expected_target_positions = torch.arange(
        target_positions.numel(), device=target_positions.device
    )
    if not torch.equal(target_positions, expected_target_positions):
        raise ValueError("The target protein is not the first contiguous entity.")

    binder_start = token_count - binder_residue_count
    expected_binder_positions = torch.arange(
        binder_start, token_count, device=binder_positions.device
    )
    if not torch.equal(binder_positions, expected_binder_positions):
        raise ValueError(
            "The binder chain is not a one-token-per-residue suffix. This runner "
            "supports CCD modifications on the target, but not on the designed "
            "binder."
        )
    if ligand_entity_count == 0 and ligand_positions.numel() != 0:
        raise ValueError("Unexpected non-target tokens occur before the binder.")

    target_token_residue_index = residue_index[target_positions].clone()
    invalid = (target_token_residue_index < 0) | (
        target_token_residue_index >= target_residue_count
    )
    if bool(invalid.any()):
        bad_values = torch.unique(target_token_residue_index[invalid]).tolist()
        raise ValueError(
            "ESMFold2 target residue_index values do not map onto the supplied "
            f"{target_residue_count}-residue target: {bad_values}."
        )

    # Select one chemically comparable representative token per target residue
    # for final distogram proxy scoring. Prefer the token containing C-alpha;
    # fall back to the first token if a CCD lacks an identifiable CA atom.
    atom_to_token = features["atom_to_token"].to(dtype=torch.long)
    atom_name_chars = features["ref_atom_name_chars"]
    atom_mask = features["atom_attention_mask"].to(dtype=torch.bool)
    representative_indices: list[int] = []
    for target_residue_idx in range(target_residue_count):
        candidates = torch.nonzero(
            target_token_residue_index == target_residue_idx, as_tuple=False
        ).flatten()
        if candidates.numel() == 0:
            raise ValueError(
                f"No ESMFold2 model token maps to target residue {target_residue_idx + 1}."
            )
        candidate_token_indices = target_positions[candidates]
        representative = int(candidate_token_indices[0])
        for candidate in candidate_token_indices.tolist():
            candidate_atoms = torch.nonzero(
                atom_mask & (atom_to_token == candidate), as_tuple=False
            ).flatten()
            atom_names = {
                _decode_atom_name(atom_name_chars[atom_idx])
                for atom_idx in candidate_atoms.tolist()
            }
            if "CA" in atom_names:
                representative = int(candidate)
                break
        representative_indices.append(representative)

    return ComplexTokenLayout(
        target_token_count=int(target_positions.numel()),
        target_token_indices=target_positions,
        target_token_residue_index=target_token_residue_index,
        target_representative_token_index=torch.tensor(
            representative_indices,
            dtype=torch.long,
            device=residue_index.device,
        ),
        ligand_token_indices=ligand_positions,
        binder_token_indices=binder_positions,
    )


def _extract_target_binder_distogram_for_scoring(
    distogram_logits: torch.Tensor,
    *,
    target_representative_token_index: torch.Tensor,
    binder_token_indices: torch.Tensor,
) -> torch.Tensor:
    """Extract EF-Tu representatives plus binder, excluding every ligand token."""
    if distogram_logits.ndim != 4:
        raise ValueError(
            "Expected batched distogram logits [B, L, L, bins], got "
            f"{tuple(distogram_logits.shape)}."
        )
    full_length = distogram_logits.shape[1]
    if distogram_logits.shape[2] != full_length:
        raise ValueError("Distogram residue/token axes must be square.")
    target_indices = target_representative_token_index.to(
        device=distogram_logits.device, dtype=torch.long
    )
    binder_indices = binder_token_indices.to(
        device=distogram_logits.device, dtype=torch.long
    )
    if bool((target_indices < 0).any()) or bool(
        (target_indices >= full_length).any()
    ):
        raise ValueError("Representative target-token index is outside the target.")
    if bool((binder_indices < 0).any()) or bool(
        (binder_indices >= full_length).any()
    ):
        raise ValueError("A binder token index is outside the full complex.")
    if bool(torch.isin(target_indices, binder_indices).any()):
        raise ValueError("Target and binder scoring token sets overlap.")
    keep = torch.cat((target_indices, binder_indices))
    return distogram_logits.index_select(1, keep).index_select(2, keep)


@cache
def _ensure_ccd_loaded() -> None:
    load_ccd()


def prepare_esmfold2_tensors(
    input: StructurePredictionInput,
    max_tokens: int | None = None,
    max_atoms: int | None = None,
    max_seqs: int = 16384,
    pad_to_max_seqs: bool = False,
    seed: int | None = None,
    use_vectorized_msa_assembly: bool = True,
) -> dict[str, torch.Tensor]:
    del max_tokens, max_seqs, pad_to_max_seqs, use_vectorized_msa_assembly
    _ensure_ccd_loaded()
    features, _ = prepare_esmfold2_input(input, seed=seed)
    if max_atoms is not None:
        for key, dim in _ATOM_FEATURE_DIMS.items():
            if key in features:
                features[key] = _resize_tensor(features[key], dim=dim, size=max_atoms)
    return features


def fold_and_get_distogram(
    model: EsmFold2ExperimentalModel,
    target_seq: str,
    target_one_hot: torch.Tensor,
    design: torch.Tensor,
    target_modifications: list[Modification] | None = None,
    target_ligand_ccds: list[str] | None = None,
    num_loops: int = 0,
    num_sampling_steps: int = 1,
    calculate_confidence: bool = False,
    seed: int | None = None,
) -> dict:
    """Prepare PTM/ligand-aware inputs and return explicit token metadata."""
    del target_one_hot  # Kept in the public signature for backward compatibility.
    padding = (2, 11)
    padded_design = F.pad(design, padding, mode="constant", value=0)

    # Argmax to get the designed sequence string.
    token_lists = torch.argmax(padded_design, dim=-1)
    designed_seq = [
        [PROTEIN_3TO1[TOKENS[int(tkn.item())]] for tkn in token_list]
        for token_list in token_lists
    ]
    seq_list = [target_seq + "|" + "".join(seq) for seq in designed_seq]

    inputs_list: list[dict[str, torch.Tensor]] = []
    token_layouts: list[ComplexTokenLayout] = []
    ligand_ccds = list(target_ligand_ccds or [])
    entity_count = 2 + len(ligand_ccds)
    if entity_count > len(_CHAIN_ID_ALPHABET):
        raise ValueError(
            f"This runner supports at most {len(_CHAIN_ID_ALPHABET) - 2} ligands."
        )
    for seq in seq_list:
        target_chain, binder_chain = seq.split("|", maxsplit=1)
        sequence_entities: list[ProteinInput | LigandInput] = [
            ProteinInput(
                id=_CHAIN_ID_ALPHABET[0],
                sequence=target_chain,
                modifications=target_modifications,
                msa=None,
            )
        ]
        sequence_entities.extend(
            LigandInput(id=_CHAIN_ID_ALPHABET[i + 1], ccd=[ccd])
            for i, ccd in enumerate(ligand_ccds)
        )
        sequence_entities.append(
            ProteinInput(
                id=_CHAIN_ID_ALPHABET[len(ligand_ccds) + 1],
                sequence=binder_chain,
                msa=None,
            )
        )
        inputs_raw = StructurePredictionInput(
            sequences=sequence_entities
        )
        prepared = prepare_esmfold2_tensors(inputs_raw)
        inputs_list.append(prepared)
        token_layouts.append(
            _infer_complex_token_layout(
                prepared,
                target_residue_count=len(target_chain),
                binder_residue_count=len(binder_chain),
                ligand_entity_count=len(ligand_ccds),
            )
        )

    first_layout = token_layouts[0]
    for layout in token_layouts[1:]:
        if (
            layout.target_token_count != first_layout.target_token_count
            or not torch.equal(
                layout.target_token_indices,
                first_layout.target_token_indices,
            )
            or not torch.equal(
                layout.target_token_residue_index,
                first_layout.target_token_residue_index,
            )
            or not torch.equal(
                layout.target_representative_token_index,
                first_layout.target_representative_token_index,
            )
            or not torch.equal(
                layout.ligand_token_indices,
                first_layout.ligand_token_indices,
            )
            or not torch.equal(
                layout.binder_token_indices,
                first_layout.binder_token_indices,
            )
        ):
            raise ValueError(
                "Target tokenization changed across binder batch members; cannot "
                "construct a single differentiable batch."
            )

    # Different designed amino acids can produce different atom counts. Pad all
    # atom-level features to the largest member of the batch without truncating
    # modified residues (the tutorial's length*14 heuristic can be too small).
    max_atoms = max(inp["ref_pos"].shape[0] for inp in inputs_list)
    for inp in inputs_list:
        for key, dim in _ATOM_FEATURE_DIMS.items():
            if key in inp:
                inp[key] = _resize_tensor(inp[key], dim=dim, size=max_atoms)

    inputs = {
        key: torch.stack([inp[key] for inp in inputs_list], dim=0).cuda()
        for key in inputs_list[0]
    }
    fixed_token_count = int(first_layout.binder_token_indices[0])
    fixed_res_type = inputs["res_type"][:, :fixed_token_count]
    if bool((fixed_res_type < 0).any()) or bool(
        (fixed_res_type >= len(TOKENS)).any()
    ):
        raise ValueError(
            "ESMFold2 produced a fixed target/ligand res_type outside the design "
            f"vocabulary 0..{len(TOKENS) - 1}."
        )
    fixed_token_one_hot = F.one_hot(
        fixed_res_type.to(dtype=torch.long), num_classes=len(TOKENS)
    ).to(dtype=padded_design.dtype)
    inputs["res_type_soft"] = torch.cat(
        (fixed_token_one_hot, padded_design), dim=1
    )
    expected_token_count = inputs["res_type"].shape[1]
    if inputs["res_type_soft"].shape[1] != expected_token_count:
        raise ValueError(
            "PTM-aware soft residue tensor has length "
            f"{inputs['res_type_soft'].shape[1]}, but ESMFold2 expects "
            f"{expected_token_count} model tokens."
        )

    with seed_context(seed), torch.autocast(
        device_type=design.device.type,
        dtype=torch.bfloat16,
        enabled=design.is_cuda,
    ):
        output = model(
            **inputs,
            num_diffusion_samples=1,
            num_sampling_steps=num_sampling_steps,
            num_loops=num_loops,
            seed=seed,
        )

    result: dict = {
        "distogram_logits": output["distogram_logits"],
        "inputs": inputs,
        "inputs_list": inputs_list,
        "output": output,
        "seq_list": seq_list,
        "target_token_count": first_layout.target_token_count,
        "target_token_indices": first_layout.target_token_indices,
        "target_token_residue_index": first_layout.target_token_residue_index,
        "target_representative_token_index": (
            first_layout.target_representative_token_index
        ),
        "ligand_token_indices": first_layout.ligand_token_indices,
        "binder_token_indices": first_layout.binder_token_indices,
        "target_token_mask": first_layout.target_token_mask(),
        "ligand_token_mask": first_layout.ligand_token_mask(),
        "binder_token_mask": first_layout.binder_token_mask(),
        "target_ligand_ccds": ligand_ccds,
    }
    if calculate_confidence:
        result.update(
            {
                "ptm": output.get("ptm"),
                "iptm": output.get("iptm"),
                "plddt": output.get("plddt"),
            }
        )
    return result


_CHAIN_ID_ALPHABET = string.ascii_uppercase + string.ascii_lowercase + string.digits


def _asym_id_to_chain_label(asym_id: int) -> str:
    if asym_id < 0:
        raise ValueError(f"asym_id must be >= 0, got {asym_id}")
    label = ""
    n = len(_CHAIN_ID_ALPHABET)
    while True:
        label = _CHAIN_ID_ALPHABET[asym_id % n] + label
        asym_id = asym_id // n - 1
        if asym_id < 0:
            return label


def to_atom_array(
    coords: np.ndarray,
    atom_to_token: np.ndarray,
    res_type: np.ndarray,
    residue_index: np.ndarray,
    asym_id: np.ndarray,
    mol_type: np.ndarray,
    ref_atom_name_chars: np.ndarray,
    ref_element: np.ndarray,
    atom_attention_mask: np.ndarray,
    plddt_per_atom: np.ndarray | None = None,
) -> biotite.structure.AtomArray:
    atoms = []
    for atom_i, (
        atom_coord,
        token_idx,
        atom_name_chars,
        element_idx,
        is_not_pad,
    ) in enumerate(
        zip(
            coords, atom_to_token, ref_atom_name_chars, ref_element, atom_attention_mask
        )
    ):
        if not is_not_pad:
            continue
        atoms.append(
            biotite.structure.Atom(
                coord=atom_coord,
                chain_id=_asym_id_to_chain_label(int(asym_id[token_idx])),
                res_id=residue_index[token_idx] + 1,
                res_name=TOKENS[res_type[token_idx]],
                atom_name="".join(chr(c + 32) for c in atom_name_chars if c != 0),
                element=ELEMENTS[element_idx],
                ins_code=" ",
                hetero=mol_type[token_idx] == MOL_TYPE_NONPOLYMER,
                b_factor=float(plddt_per_atom[atom_i])
                if plddt_per_atom is not None
                else 0.0,
            )
        )
    return biotite.structure.array(atoms)


@dataclass
class PredictedAllAtomComplex:
    """Minimal all-atom result wrapper that preserves noncanonical CCD atoms."""

    atom_array: biotite.structure.AtomArray

    def to_pdb_string(self) -> str:
        pdb_file = PDBFile()
        pdb_file.set_structure(self.atom_array)
        buffer = io.StringIO()
        pdb_file.write(buffer)
        return buffer.getvalue()

    def to_mmcif_string(self) -> str:
        cif_file = CIFFile()
        set_pdbx_structure(cif_file, self.atom_array, data_block="esmfold2_design")
        buffer = io.StringIO()
        cif_file.write(buffer)
        return buffer.getvalue()


def build_complex(
    inputs: dict[str, torch.Tensor],
    output: dict[str, Any],
    target_modifications: list[Modification] | None = None,
    target_ligand_ccds: list[str] | None = None,
) -> PredictedAllAtomComplex:
    """Build an all-atom complex preserving PTM and ligand CCD identities."""
    plddt_per_atom = output.get("plddt_per_atom")
    if plddt_per_atom is not None:
        plddt_per_atom = (plddt_per_atom[0].detach().to(device="cpu", dtype=torch.float32).numpy() * PLDDT_B_FACTOR_SCALE)
    atom_arr = to_atom_array(
        coords=output["sample_atom_coords"][0].detach().to(device="cpu", dtype=torch.float32).numpy(),
        atom_to_token=inputs["atom_to_token"][0].cpu().numpy(),
        res_type=inputs["res_type"][0].cpu().numpy(),
        residue_index=inputs["residue_index"][0].cpu().numpy(),
        asym_id=inputs["asym_id"][0].cpu().numpy(),
        mol_type=inputs["mol_type"][0].cpu().numpy(),
        ref_atom_name_chars=inputs["ref_atom_name_chars"][0].cpu().numpy(),
        ref_element=inputs["ref_element"][0].cpu().numpy(),
        atom_attention_mask=inputs["atom_attention_mask"][0].cpu().numpy(),
        plddt_per_atom=plddt_per_atom,
    )
    # ``res_type`` retains the canonical parent amino acid. Restore the target
    # CCD residue name in the written structure so SEP/TPO/PTR atoms are not
    # mislabeled as SER/THR/TYR.
    for modification in target_modifications or []:
        mask = (atom_arr.chain_id == "A") & (atom_arr.res_id == modification.position + 1)
        atom_arr.res_name[mask] = modification.ccd
    for ligand_index, ligand_ccd in enumerate(target_ligand_ccds or [], start=1):
        ligand_chain = _asym_id_to_chain_label(ligand_index)
        mask = atom_arr.chain_id == ligand_chain
        if not bool(mask.any()):
            raise ValueError(
                f"No output atoms were found for ligand {ligand_ccd!r} "
                f"(expected chain {ligand_chain})."
            )
        atom_arr.res_name[mask] = ligand_ccd
        atom_arr.hetero[mask] = True
    return PredictedAllAtomComplex(atom_array=atom_arr)


# ---- LM loss ----


@cache
def _folding_trunk_to_lm_aa_vocab_matrix(device: torch.device) -> torch.Tensor:
    """Build a matrix of shape [ft_aas=20, lm_aas=20]."""
    three_to_one_map = {v: k for k, v in PROTEIN_1TO3.items()}
    ft_aas = [three_to_one_map[tok_3letter] for tok_3letter in TOKENS[2:22]]

    lm_vocab = sorted(EsmcTokenizer().vocab.items(), key=lambda x: x[1])
    lm_aas = [lm_vocab[i][0] for i in range(4, 24)]

    ft_to_lm_aa_matrix = torch.zeros(20, 20)
    for ft_idx, ft_aa in enumerate(ft_aas):
        lm_idx = lm_aas.index(ft_aa)
        ft_to_lm_aa_matrix[ft_idx, lm_idx] = 1

    return ft_to_lm_aa_matrix.to(device=device)


def _one_hot_from_probs(probs: torch.Tensor) -> torch.Tensor:
    return F.one_hot(torch.argmax(probs, dim=-1), num_classes=probs.size(-1)).to(
        probs.dtype
    )


def _straight_through(discrete: torch.Tensor, continuous: torch.Tensor) -> torch.Tensor:
    return continuous + (discrete - continuous).detach()


def compute_esmc_pseudoperplexity_nll(
    esmc_model: EsmcForMaskedLM,
    binder_design: torch.Tensor,
    score_mask: torch.Tensor,
    batch_size: int = 4,
    n_passes: int = 4,
) -> torch.Tensor:
    """Algorithm 14 ESMC Pseudo-perplexity Sequence Regularization.

    Approximate pseudoperplexity NLL via multiple sampled masks."""
    device = binder_design.device
    lm_vocab_size = esmc_model.config.vocab_size
    model_dtype = esmc_model.esmc.embed.weight.dtype

    target_esm = binder_design @ _folding_trunk_to_lm_aa_vocab_matrix(device)
    input_esm = _straight_through(_one_hot_from_probs(target_esm), target_esm)
    input_ids = torch.zeros(
        (binder_design.size(0), binder_design.size(1) + 2, lm_vocab_size),
        dtype=model_dtype,
        device=device,
    )
    tokenizer = EsmcTokenizer()
    input_ids[:, 0, tokenizer.cls_token_id] = 1  # pyright: ignore
    input_ids[:, -1, tokenizer.eos_token_id] = 1  # pyright: ignore
    input_ids[:, 1:-1, 4:24] = input_esm.to(model_dtype)

    if score_mask.ndim == 1:
        score_mask = score_mask.unsqueeze(0).expand(binder_design.size(0), -1)
    elif score_mask.shape != binder_design.shape[:2]:
        raise ValueError(
            f"Expected score_mask with shape {(binder_design.size(0), binder_design.size(1))}, "
            f"got {tuple(score_mask.shape)}"
        )
    score_mask = score_mask.to(device=device, dtype=torch.bool)

    mask_token = torch.zeros(lm_vocab_size, dtype=model_dtype, device=device)
    mask_token[esmc_model.config.mask_token_id] = 1
    esmc = esmc_model.esmc

    all_masked_sequences = []
    all_pass_masks = []
    for batch_idx in range(binder_design.size(0)):
        position_indices = score_mask[batch_idx].nonzero(as_tuple=False).flatten()
        num_positions = int(position_indices.numel())
        if num_positions == 0:
            raise ValueError(
                "ESMC pseudoperplexity score mask selected zero positions."
            )

        effective_n_passes = min(n_passes, num_positions)

        permuted_positions = position_indices[
            torch.randperm(num_positions, device=device)
        ]
        position_groups = torch.tensor_split(
            permuted_positions,
            effective_n_passes,
        )

        pass_masks = torch.zeros(
            (effective_n_passes, binder_design.size(1)),
            dtype=torch.bool,
            device=device,
        )

        for pass_idx, positions in enumerate(position_groups):
            pass_masks[pass_idx, positions] = True

        masked_sequences = input_ids[batch_idx : batch_idx + 1].repeat(effective_n_passes, 1, 1)
        mask_rows, mask_cols = pass_masks.nonzero(as_tuple=True)
        masked_sequences[mask_rows, mask_cols + 1] = mask_token

        all_masked_sequences.append(masked_sequences)
        all_pass_masks.append(pass_masks)

    logit_chunks = []
    masked_sequence_rows = torch.cat(all_masked_sequences, dim=0)
    for start in range(0, masked_sequence_rows.size(0), batch_size):
        chunk = masked_sequence_rows[start : start + batch_size]
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            hidden, *_ = esmc.transformer(
                chunk @ esmc.embed.weight.to(chunk.dtype),
                sequence_id=None,
                layers_to_collect=[],
                output_attentions=False,
            )
            logit_chunks.append(esmc_model.lm_head(hidden))
    logits = torch.cat(logit_chunks, dim=0)

    losses = []
    row_start = 0

    for batch_idx, pass_masks in enumerate(all_pass_masks):
        effective_n_passes = pass_masks.size(0)
        row_stop = row_start + effective_n_passes

        target_weights = target_esm[batch_idx]
        log_probs = (
            logits[row_start:row_stop]
            .log_softmax(dim=-1)[:, 1:-1, 4:24]
        )

        nlls = -(
            log_probs
            * target_weights.to(log_probs.dtype).unsqueeze(0)
        ).sum(dim=-1)

        losses.append(nlls[pass_masks].mean())
        row_start = row_stop

    return torch.stack(losses, dim=0)


# ---- Design ----


def normalized_gradient_tensor(
    grad: torch.Tensor, gradient_mask: torch.Tensor
) -> torch.Tensor:
    masked_grad = grad * gradient_mask
    index_has_nonzero_grad = torch.square(masked_grad).sum(-1) > 0  # (B, L)
    eff_L = index_has_nonzero_grad.sum(-1)  # (B,)
    grad_norm = torch.linalg.norm(masked_grad, axis=(-1, -2))  # (B,)
    normalized_grad = (masked_grad / (grad_norm[:, None, None] + 1e-7)) * torch.sqrt(
        eff_L[:, None, None]
    )
    return normalized_grad * gradient_mask

def per_batch_gradient_norm(
    grad: torch.Tensor,
    gradient_mask: torch.Tensor,
) -> torch.Tensor:
    masked = (grad * gradient_mask).flatten(start_dim=1)
    return torch.linalg.vector_norm(masked, dim=1)


def per_batch_gradient_cosine(
    grad_a: torch.Tensor,
    grad_b: torch.Tensor,
    gradient_mask: torch.Tensor,
) -> torch.Tensor:
    a = (grad_a * gradient_mask).flatten(start_dim=1)
    b = (grad_b * gradient_mask).flatten(start_dim=1)

    return F.cosine_similarity(a, b, dim=1, eps=1e-8)

def design_binder(
    inversion_models: dict[str, EsmFold2ExperimentalModel],
    hf_critic_models: dict[str, EsmFold2ExperimentalModel],
    esmc_model: EsmcForMaskedLM,
    scaling_critic_names: list[str] | None,
    target_name: str,
    target_sequence: str | None,
    binder_name: str,
    binder_sequence: str | None,
    is_antibody: bool | None,
    seed: int,
    batch_size: int = 1,
    target_modifications: list[Modification] | None = None,
    target_ligand_ccds: list[str] | None = None,
    target_hotspot_ids: list[str] | None = None,
    epitope_contact_distance: float = 12.0,
    score_unmodified_control: bool = True,
) -> tuple[list[str], Trajectory, list[dict]]:
    """
    Algorithm 11 Gradient-Guided Binder Sequence Optimization.

    Run the full optimization loop.
    Returns dict with designed_sequence, complex, and trajectory.

    Every critic is folded once on the best designed sequence via HF ESMFold2.
    Hero critics expose iPTM; scaling critics contribute distogram scores only.
    ``distogram_binding_confidence`` / ``cdr_distogram_binding_confidence`` come
    from the distogram in all cases.

    ``target_hotspot_ids`` enables epitope loss and is interpreted as 1-indexed
    residue IDs within the provided target sequence, for example ``["L150"]``.
    ``target_modifications`` uses zero-indexed positions, matching ESMFold2's
    :class:`Modification`. If requested, each final design is also scored
    against an otherwise identical unmodified target as a selectivity control.
    ``target_ligand_ccds`` are fixed entities placed between target and binder
    in every optimization and critic call, including the unmodified control.
    """
    # Setup
    device = "cuda"
    if target_name in TARGET_SEQUENCES:
        if target_sequence is not None:
            raise ValueError(
                f"{target_name!r} is a preset target; omit target_sequence."
            )
        target_sequence = TARGET_SEQUENCES[target_name]
    elif target_sequence is None:
        raise ValueError(
            f"{target_name!r} is not a preset target; provide target_sequence."
        )
    target_one_hot = sequence_to_one_hot(target_sequence, device=device)

    if binder_name in BINDER_PROMPT_FACTORIES:
        if binder_sequence is not None:
            raise ValueError(
                f"{binder_name!r} is a preset binder; omit binder_sequence."
            )
        binder_prompt_factor = BINDER_PROMPT_FACTORIES[binder_name]
        if is_antibody is not None:
            assert (
                binder_prompt_factor.is_antibody == is_antibody
            ), "Conflict in is_antibody settings."
        is_antibody = binder_prompt_factor.is_antibody
        binder_sequence = binder_prompt_factor.sample(seed=seed)
    elif binder_sequence is None:
        raise ValueError(
            f"{binder_name!r} is not a preset binder; provide binder_sequence."
        )
    elif is_antibody is None:
        is_antibody = _is_antibody_sequence(binder_sequence)

    binder_length = len(binder_sequence)

    # By default, we only support single binder and target chains.
    # To support this case, remove the asserts below and check that losses
    # and selection metrics are appropriate for your multi-chain case.
    assert "|" not in target_sequence
    assert "|" not in binder_sequence

    with seed_context(seed), torch.device(device):
        logits = build_initial_soft_sequence_logits(
            binder_sequence, batch_size=batch_size
        )
        gradient_mask = build_gradient_mask(binder_sequence, batch_size=batch_size)

    # step -> {loss_name: [B] tensor on CPU, metric_name: float}
    trajectory: Trajectory = {}
    global_step = 0

    def run_step(
        logits: torch.Tensor,
        optimizer: optim.Optimizer,
        temperature: float,
        calculate_confidence: bool,
    ) -> tuple[torch.Tensor, list[str], list[float] | None]:
        nonlocal global_step
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        optimizer.zero_grad()

        random.seed(seed + global_step)
        replicate_choice = random.randint(0, len(inversion_models) - 1)
        inversion_model = list(inversion_models.values())[replicate_choice]
        design = F.softmax(logits / temperature, dim=-1)

        fold_result = fold_and_get_distogram(
            inversion_model,
            target_sequence,
            target_one_hot,
            design,
            target_modifications=target_modifications,
            target_ligand_ccds=target_ligand_ccds,
            num_loops=1,
            num_sampling_steps=50 if calculate_confidence else 1,
            calculate_confidence=calculate_confidence,
            seed=seed + global_step,
        )
        sequences: list[str] = fold_result["seq_list"]
        losses = compute_structure_losses(
            fold_result["distogram_logits"],
            binder_length,
            target_sequence=target_sequence,
            target_hotspot_ids=target_hotspot_ids,
            epitope_contact_distance=epitope_contact_distance,
            target_token_residue_index=fold_result["target_token_residue_index"],
            target_token_indices=fold_result["target_token_indices"],
            target_token_mask=fold_result["target_token_mask"],
            binder_token_mask=fold_result["binder_token_mask"],
        )
        structure_loss = losses["total_loss"]
        structure_grad = torch.autograd.grad(structure_loss.mean(), logits)[0]

        # Recompute the logits -> design transform for a fresh graph.
        design = F.softmax(logits / temperature, dim=-1)
        score_mask = gradient_mask.sum(dim=-1) > 0
        with seed_context(seed + global_step):
            plm_loss = compute_esmc_pseudoperplexity_nll(
                esmc_model=esmc_model,
                binder_design=design,
                score_mask=score_mask,
                batch_size=LM_LOSS_BATCH_SIZE,
                n_passes=LM_MASK_PASSES,
            )
        # PLM gradient 
        plm_grad = torch.autograd.grad(plm_loss.mean(), logits)[0]

        # Rebuild a fresh graph for the explicit quality constraint.
        design = F.softmax(logits / temperature, dim=-1)
        quality_losses = compute_sequence_quality_losses(design)
        quality_loss = quality_losses["sequence_quality_loss"]
        quality_grad = torch.autograd.grad(quality_loss.mean(), logits)[0]

        structure_direction = normalized_gradient_tensor(structure_grad, gradient_mask)
        plm_direction = normalized_gradient_tensor(plm_grad, gradient_mask)
        quality_direction = normalized_gradient_tensor(quality_grad, gradient_mask)

        plm_grad_weight = (ANTIBODY_PLM_GRAD_WEIGHT if is_antibody else MINIBINDER_PLM_GRAD_WEIGHT)
        quality_grad_weight = 0.0 if is_antibody else 1.0

        combined_direction = (structure_direction + plm_grad_weight * plm_direction + quality_grad_weight * quality_direction)

        # Calculate diagnostics before optimizer.step().
        gradient_diagnostics = {
            # Raw gradient magnitudes before normalization
            "structure_grad_norm": per_batch_gradient_norm(
                structure_grad, gradient_mask
            ),
            "plm_grad_norm": per_batch_gradient_norm(
                plm_grad, gradient_mask
            ),
            "quality_grad_norm": per_batch_gradient_norm(
                quality_grad, gradient_mask
            ),

            # Directional agreement/conflict
            "cos_structure_plm": per_batch_gradient_cosine(
                structure_grad, plm_grad, gradient_mask
            ),
            "cos_structure_quality": per_batch_gradient_cosine(
                structure_grad, quality_grad, gradient_mask
            ),
            "cos_plm_quality": per_batch_gradient_cosine(
                plm_grad, quality_grad, gradient_mask
            ),

            # Magnitudes after normalization and weighting
            "structure_component_norm": per_batch_gradient_norm(
                structure_direction, gradient_mask
            ),
            "weighted_plm_component_norm": per_batch_gradient_norm(
                plm_grad_weight * plm_direction, gradient_mask
            ),
            "weighted_quality_component_norm": per_batch_gradient_norm(
                quality_grad_weight * quality_direction, gradient_mask
            ),
            "combined_grad_norm": per_batch_gradient_norm(
                combined_direction, gradient_mask
            ),
        }

        logits.grad = combined_direction

        # Trajectory block 
        for group in optimizer.param_groups:
            group["lr"] = LEARNING_RATE * temperature

        effective_lr = optimizer.param_groups[0]["lr"]
        optimizer.step()

        step = global_step

        # Preserve the individual structure-loss components, but rename their
        # existing total_loss to avoid confusing it with the optimizer objective.
        step_losses: TrajectoryStep = {
            k: v.detach().cpu()
            for k, v in losses.items()
            if k != "total_loss"
        }

        step_losses["structure_loss"] = structure_loss.detach().cpu()
        step_losses["plm_loss"] = plm_loss.detach().cpu()

        step_losses.update({
            k: v.detach().cpu()
            for k, v in quality_losses.items()
        })

        step_losses.update({
            k: v.detach().cpu()
            for k, v in gradient_diagnostics.items()
        })

        # There is no true scalar "total loss", because the gradients are
        # normalized separately before being combined.
        step_losses["raw_unweighted_loss_sum"] = (
            structure_loss + plm_loss + quality_loss
        ).detach().cpu()

        step_losses["plm_grad_weight"] = float(plm_grad_weight)
        step_losses["quality_grad_weight"] = float(quality_grad_weight)
        step_losses["temperature"] = float(temperature)
        step_losses["effective_learning_rate"] = float(effective_lr)
        step_losses["time"] = time.time() - start

        step_losses["peak_allocated_gib"] = (
            torch.cuda.max_memory_allocated() / BYTES_PER_GIB
        )
        step_losses["peak_reserved_gib"] = (
            torch.cuda.max_memory_reserved() / BYTES_PER_GIB
        )

        trajectory[step] = step_losses

        loss_str = "  ".join(
            f"{k}={v.mean().item() if torch.is_tensor(v) else v:.4f}"
            for k, v in step_losses.items()
        )

        if step % LOG_INTERVAL == 0:
            logger.info(
                f"  step {step:3d}  |  {loss_str}  T={temperature:.4f}"
            )

        global_step += 1
        selection_metric = fold_result.get("iptm", None)
        if target_ligand_ccds and calculate_confidence:
            # Global iPTM covers all entities in a ligand-containing complex.
            # Select trajectories using only the EF-Tu/binder distogram proxy.
            ligand_aware_scores: list[float] = []
            for batch_idx, complex_sequence in enumerate(sequences):
                binder_seq = complex_sequence.split("|")[-1]
                scoring_distogram = _extract_target_binder_distogram_for_scoring(
                    fold_result["distogram_logits"][
                        batch_idx : batch_idx + 1
                    ].detach(),
                    target_representative_token_index=fold_result[
                        "target_representative_token_index"
                    ],
                    binder_token_indices=fold_result["binder_token_indices"],
                )
                proxy_scores = compute_distogram_iptm_proxy(
                    scoring_distogram,
                    len(target_sequence),
                    binder_seq,
                    bool(is_antibody),
                )
                proxy_key = (
                    "cdr_distogram_iptm_proxy"
                    if is_antibody
                    else "distogram_iptm_proxy"
                )
                proxy = proxy_scores[proxy_key]
                if not math.isfinite(proxy):
                    proxy = proxy_scores["distogram_iptm_proxy"]
                ligand_aware_scores.append(proxy)
            selection_metric = ligand_aware_scores
        return logits, sequences, selection_metric

    # Optimize
    optimizer = optim.SGD([logits], lr=LEARNING_RATE)
    best_selection_metric: list[float] = [-1.0] * batch_size
    best_sequences: list[str] = [""] * batch_size
    for step in range(STEPS):
        # Cosine schedule
        t = (step + 1) / STEPS
        remaining = 0.5 * (1 + math.cos(math.pi * t))
        temperature = TEMPERATURE_MIN + (1 - TEMPERATURE_MIN) * remaining
        logits, sequences, selection_metric = run_step(
            logits,
            optimizer,
            temperature=temperature,
            calculate_confidence=temperature < 0.05,
        )
        if selection_metric is not None:
            for b in range(batch_size):
                if (
                    selection_metric[b] is not None
                    and selection_metric[b] > best_selection_metric[b]
                ):
                    best_selection_metric[b] = float(selection_metric[b])
                    best_sequences[b] = sequences[b]
    assert all(seq != "" for seq in best_sequences)

    # Score
    critic_results: list[dict] = []
    target_length = len(target_sequence.replace("|", ""))
    final_raw_loss_sum = trajectory[global_step - 1]["raw_unweighted_loss_sum"]
    assert isinstance(final_raw_loss_sum, torch.Tensor)

    def score_critic(
        critic_name: str,
        critic_model: EsmFold2ExperimentalModel,
        batch_idx: int,
        is_scaling_critic: bool,
    ) -> None:
        best_seq = best_sequences[batch_idx]
        binder_seq = best_seq.split("|")[-1]
        binder_design = sequence_to_one_hot(binder_seq)[..., 2:22]
        final_fold = fold_and_get_distogram(
            critic_model,
            target_sequence,
            target_one_hot,
            binder_design,
            target_modifications=target_modifications,
            target_ligand_ccds=target_ligand_ccds,
            num_loops=3,
            num_sampling_steps=1 if is_scaling_critic else 200,
            calculate_confidence=not is_scaling_critic,
            seed=seed,
        )
        pred_complex = (
            None
            if is_scaling_critic
            else build_complex(
                final_fold["inputs"],
                final_fold["output"],
                target_modifications=target_modifications,
                target_ligand_ccds=target_ligand_ccds,
            )
        )
        scoring_distogram = _extract_target_binder_distogram_for_scoring(
            final_fold["distogram_logits"],
            target_representative_token_index=final_fold[
                "target_representative_token_index"
            ],
            binder_token_indices=final_fold["binder_token_indices"],
        )
        iptm_proxy_scores = compute_distogram_iptm_proxy(
            scoring_distogram, target_length, binder_seq, is_antibody
        )
        iptm_tensor = final_fold.get("iptm")
        iptm = iptm_tensor.item() if iptm_tensor is not None else None
        result = {
            "is_antibody": is_antibody,
            "critic_name": critic_name,
            "is_scaling_critic": is_scaling_critic,
            "batch_idx": batch_idx,
            "designed_sequence": best_seq,
            "complex": pred_complex,
            "final_loss": final_raw_loss_sum[batch_idx].item(),
            "iptm": iptm,
            "iptm_scope": (
                "whole_complex" if target_ligand_ccds else "target_binder"
            ),
            "target_ligands": ",".join(target_ligand_ccds or []),
            "logits": logits[batch_idx].detach().cpu(),
            **iptm_proxy_scores,
        }

        if target_modifications and score_unmodified_control:
            control_fold = fold_and_get_distogram(
                critic_model,
                target_sequence,
                target_one_hot,
                binder_design,
                target_modifications=None,
                target_ligand_ccds=target_ligand_ccds,
                num_loops=3,
                num_sampling_steps=1 if is_scaling_critic else 200,
                calculate_confidence=not is_scaling_critic,
                seed=seed,
            )
            control_scoring_distogram = _extract_target_binder_distogram_for_scoring(
                control_fold["distogram_logits"],
                target_representative_token_index=control_fold[
                    "target_representative_token_index"
                ],
                binder_token_indices=control_fold["binder_token_indices"],
            )
            control_proxy_scores = compute_distogram_iptm_proxy(
                control_scoring_distogram,
                target_length,
                binder_seq,
                is_antibody,
            )
            control_iptm_tensor = control_fold.get("iptm")
            control_iptm = (
                control_iptm_tensor.item() if control_iptm_tensor is not None else None
            )
            result.update(
                {
                    "unmodified_iptm": control_iptm,
                    "delta_iptm": (
                        iptm - control_iptm
                        if iptm is not None and control_iptm is not None
                        else None
                    ),
                    "unmodified_distogram_iptm_proxy": control_proxy_scores[
                        "distogram_iptm_proxy"
                    ],
                    "delta_distogram_iptm_proxy": (
                        iptm_proxy_scores["distogram_iptm_proxy"]
                        - control_proxy_scores["distogram_iptm_proxy"]
                    ),
                    "unmodified_cdr_distogram_iptm_proxy": control_proxy_scores[
                        "cdr_distogram_iptm_proxy"
                    ],
                    "delta_cdr_distogram_iptm_proxy": (
                        iptm_proxy_scores["cdr_distogram_iptm_proxy"]
                        - control_proxy_scores["cdr_distogram_iptm_proxy"]
                    ),
                }
            )
            del control_fold

        critic_results.append(result)
        del final_fold

    for critic_name, critic_model in hf_critic_models.items():
        for batch_idx in range(batch_size):
            score_critic(critic_name, critic_model, batch_idx, is_scaling_critic=False)

    for critic_name in scaling_critic_names or []:
        critic_model = _load_hf_model(
            critic_name, lm_dropout=0.25, cache_esmc=False, device="cuda"
        )
        for batch_idx in range(batch_size):
            score_critic(critic_name, critic_model, batch_idx, is_scaling_critic=True)
        del critic_model
        torch.cuda.empty_cache()

    if not critic_results:
        for batch_idx in range(batch_size):
            critic_results.append(
                {
                    "is_antibody": is_antibody,
                    "batch_idx": batch_idx,
                    "designed_sequence": best_sequences[batch_idx],
                    "final_loss": final_raw_loss_sum[batch_idx].item(),
                    "logits": logits[batch_idx].detach().cpu(),
                }
            )

    return best_sequences, trajectory, critic_results


# ---- Model Loading ----

_ESMC_CACHE: dict[str, torch.nn.Module] = {}


def _load_hf_model(
    critic_name: str, lm_dropout: float, cache_esmc: bool, device: str
) -> Any:
    """Loads ESMFold2 from huggingface. Will cache ESMC by checkpoint ID among
    all non-scaling checkpoints, to save on VRAM and load time."""
    repo_id = f"biohub/{critic_name}"
    model = EsmFold2ExperimentalModel.from_pretrained(repo_id, load_esmc=not cache_esmc)
    if cache_esmc:
        esmc_id = model.config.esmc_id
        if esmc_id not in _ESMC_CACHE:
            model.load_esmc(esmc_id)
            assert model.esmc is not None
            _ESMC_CACHE[esmc_id] = model.esmc
        model.esmc = _ESMC_CACHE[esmc_id]
    model.configure_lm_dropout(lm_dropout, force_lm_dropout_during_inference=True)
    kernel_backend = None
    if TRITON_KERNELS_AVAILABLE:
        kernel_backend = BACKEND_FUSED
    elif CUE_AVAILABLE:
        kernel_backend = BACKEND_CUEQ
    model.set_kernel_backend(kernel_backend)
    return model.to(device=device).eval().requires_grad_(False)


def _apply_torch_compile(model: torch.nn.Module) -> None:
    """A helper for torch compiling the model."""
    torch._dynamo.config.cache_size_limit = 512
    torch._dynamo.config.accumulated_cache_size_limit = 512

    compile_targets = (ESMFold2MSAEncoder, PairUpdateBlock)

    def _maybe_compile_module(module: torch.nn.Module) -> None:
        if not isinstance(module, compile_targets):
            return
        module.forward = torch.compile(module.forward)  # ty:ignore[invalid-assignment]

    model.apply(_maybe_compile_module)


class ESMFold2Design:
    lm_name = "biohub/ESMC-6B"
    inversion_model_names: list[str] = [
        "ESMFold2-Experimental-Fast",
        "ESMFold2-Experimental-Fast-Cutoff2025",
    ]
    hero_critic_hf_paths: list[str] = [
        "ESMFold2-Experimental-Fast",
        "ESMFold2-Experimental-Fast-Cutoff2025",
        "ESMFold2-Experimental",
        "ESMFold2-Experimental-Cutoff2025",
    ]
    scaling_critic_hf_paths: list[str] = []

    def __init__(
        self,
        *,
        inversion_model_names: list[str] | None = None,
        hero_critic_hf_paths: list[str] | None = None,
        compile_models: bool = COMPILE,
    ) -> None:
        if inversion_model_names is not None:
            self.inversion_model_names = inversion_model_names
        if hero_critic_hf_paths is not None:
            self.hero_critic_hf_paths = hero_critic_hf_paths
        self.compile_models = compile_models

    def load(self, use_scaling_critics: bool):
        self.scaling_critic_hf_paths = []
        if use_scaling_critics:
            self.scaling_critic_hf_paths = [
                f"ESMFold2-Experimental-Fast-base{size}-step{step}k"
                for size in ("300M", "600M", "6B")
                for step in ("250", "500", "750", "1000", "1500")
            ]

        self.inversion_models = {
            model_name: _load_hf_model(
                model_name, lm_dropout=0.5, cache_esmc=True, device="cuda"
            )
            for model_name in self.inversion_model_names
        }
        if self.compile_models:
            for model in self.inversion_models.values():
                _apply_torch_compile(model)

        self.hf_critic_models: dict[str, Any] = {}
        for name in self.hero_critic_hf_paths:
            self.hf_critic_models[name] = _load_hf_model(
                name, lm_dropout=0.25, cache_esmc=True, device="cuda"
            )

        self.esmc_model = EsmcForMaskedLM.from_pretrained(
            self.lm_name, dtype=torch.float32
        )
        if REUSE_ESMC:
            reusable_esmc_model = self.inversion_models["ESMFold2-Experimental-Fast"]
            assert reusable_esmc_model.config.esmc_id == self.lm_name, (
                f"Cannot reuse ESMC trunk from {reusable_esmc_model.config.esmc_id!r} "
                f"with LM head from {self.lm_name!r}."
            )
            assert reusable_esmc_model.esmc is not None
            del self.esmc_model.esmc
            torch.cuda.empty_cache()
            self.esmc_model.esmc = reusable_esmc_model.esmc
        self.esmc_model = self.esmc_model.cuda().eval().requires_grad_(False)

        if CHECKPOINT_LM:
            apply_activation_checkpointing(
                self.esmc_model,
                checkpoint_wrapper_fn=partial(
                    checkpoint_wrapper,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                ),
                check_fn=lambda module: isinstance(
                    module, TransformerBlock
                ),
            )

    def design(
        self,
        target_name: str,
        binder_name: str,
        target_sequence: str | None = None,
        binder_sequence: str | None = None,
        is_antibody: bool | None = None,
        seed: int = 0,
        batch_size: int = 1,
        target_modifications: list[Modification] | None = None,
        target_ligand_ccds: list[str] | None = None,
        target_hotspot_ids: list[str] | None = None,
        epitope_contact_distance: float = 12.0,
        score_unmodified_control: bool = True,
    ) -> tuple[list[str], Trajectory, list[dict]]:
        return design_binder(
            self.inversion_models,
            self.hf_critic_models,
            self.esmc_model,
            self.scaling_critic_hf_paths,
            target_name=target_name,
            target_sequence=target_sequence,
            binder_name=binder_name,
            binder_sequence=binder_sequence,
            is_antibody=is_antibody,
            seed=seed,
            batch_size=batch_size,
            target_modifications=target_modifications,
            target_ligand_ccds=target_ligand_ccds,
            target_hotspot_ids=target_hotspot_ids,
            epitope_contact_distance=epitope_contact_distance,
            score_unmodified_control=score_unmodified_control,
        )


# ---- Vast.ai/local command-line runner ----


TUTORIAL_COMMIT = "esm==3.4.1.post1 (PyPI)"
SCFV_PRESETS = {
    "trastuzumab": "trastuzumab_framework_vhvl",
    "atezolizumab": "atezolizumab_framework_vhvl",
    "ocankitug": "ocankitug_framework_vhvl",
}
HERO_CRITICS = [
    "ESMFold2-Experimental-Fast",
    "ESMFold2-Experimental-Fast-Cutoff2025",
    "ESMFold2-Experimental",
    "ESMFold2-Experimental-Cutoff2025",
]
MINIMAL_CRITICS = ["ESMFold2-Experimental-Cutoff2025"]


def _read_single_fasta(path: Path) -> tuple[str, str]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    sequence_parts: list[str] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(sequence_parts)))
            header = line[1:].strip()
            sequence_parts = []
        else:
            if header is None:
                raise ValueError(f"{path} contains sequence text before a FASTA header.")
            sequence_parts.append(line)
    if header is not None:
        records.append((header, "".join(sequence_parts)))
    if len(records) != 1:
        raise ValueError(f"{path} must contain exactly one FASTA record, found {len(records)}.")
    return records[0]


def _validate_protein_sequence(sequence: str, label: str) -> str:
    sequence = re.sub(r"\s+", "", sequence).upper()
    invalid = sorted(set(sequence) - set(PROTEIN_1TO3))
    if invalid:
        raise ValueError(
            f"{label} contains unsupported residue symbols: {', '.join(invalid)}."
        )
    if not sequence:
        raise ValueError(f"{label} is empty.")
    return sequence


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_.") or "item"


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Modification):
        return {"position": value.position, "ccd": value.ccd}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_trajectory(
    path: Path, trajectory: Trajectory, batch_size: int, seed: int
) -> None:
    rows: list[dict[str, Any]] = []
    for step, values in trajectory.items():
        for batch_idx in range(batch_size):
            row: dict[str, Any] = {
                "seed": seed,
                "step": step,
                "batch_idx": batch_idx,
            }
            for key, value in values.items():
                if isinstance(value, torch.Tensor):
                    flat = value.detach().cpu().reshape(-1)
                    if flat.numel() == batch_size:
                        row[key] = flat[batch_idx].item()
                    elif flat.numel() == 1:
                        row[key] = flat.item()
                    else:
                        row[key] = json.dumps(flat.tolist())
                else:
                    row[key] = value
            rows.append(row)
    _write_csv(path, rows)


def _save_seed_result(
    seed_dir: Path,
    seed: int,
    best_sequences: list[str],
    trajectory: Trajectory,
    critic_results: list[dict[str, Any]],
) -> None:
    seed_dir.mkdir(parents=True, exist_ok=True)
    scalar_rows: list[dict[str, Any]] = []
    structures_dir = seed_dir / "structures"
    structures_dir.mkdir(exist_ok=True)

    for result in critic_results:
        row = {
            "seed": seed,
            **{
                key: _jsonable(value)
                for key, value in result.items()
                if key not in {"complex", "logits"}
            },
        }
        scalar_rows.append(row)
        protein_complex = result.get("complex")
        if protein_complex is not None:
            pdb_name = (
                f"batch_{int(result['batch_idx']):03d}_"
                f"{_slug(str(result['critic_name']))}.pdb"
            )
            (structures_dir / pdb_name).write_text(protein_complex.to_pdb_string())
            cif_name = Path(pdb_name).with_suffix(".cif").name
            (structures_dir / cif_name).write_text(
                protein_complex.to_mmcif_string()
            )

    _write_csv(seed_dir / "critic_results.csv", scalar_rows)
    _write_trajectory(
        seed_dir / "trajectory.csv",
        trajectory,
        batch_size=len(best_sequences),
        seed=seed,
    )
    with (seed_dir / "designs.fasta").open("w") as handle:
        for batch_idx, complex_sequence in enumerate(best_sequences):
            binder_sequence = complex_sequence.split("|")[-1]
            handle.write(f">seed_{seed}_batch_{batch_idx}\n{binder_sequence}\n")

    torch.save(
        {
            "seed": seed,
            "best_sequences": best_sequences,
            "final_logits": {
                int(result["batch_idx"]): result["logits"]
                for result in critic_results
                if "logits" in result
            },
        },
        seed_dir / "design_state.pt",
    )
    (seed_dir / "complete.json").write_text(
        json.dumps({"seed": seed, "status": "complete"}, indent=2) + "\n"
    )


def _float_or_none(value: Any) -> float | None:
    if value in (None, "", "None", "nan", "NaN"):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _mean(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def _rank_completed_designs(
    output_dir: Path,
    top_n: int,
    selectivity_weight: float,
) -> None:
    rows: list[dict[str, str]] = []
    for csv_path in sorted(output_dir.glob("seed_*/critic_results.csv")):
        with csv_path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        return

    by_sequence: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_sequence.setdefault(row["designed_sequence"], []).append(row)

    ranked: list[dict[str, Any]] = []
    for designed_sequence, group in by_sequence.items():
        is_antibody = group[0].get("is_antibody", "False").lower() == "true"
        has_target_ligands = bool(group[0].get("target_ligands", "").strip())
        hero_rows = [
            row
            for row in group
            if row.get("is_scaling_critic", "False").lower() != "true"
        ]
        scaling_rows = [
            row
            for row in group
            if row.get("is_scaling_critic", "False").lower() == "true"
        ]
        proxy_key = (
            "cdr_distogram_iptm_proxy"
            if is_antibody
            else "distogram_iptm_proxy"
        )
        delta_proxy_key = (
            "delta_cdr_distogram_iptm_proxy"
            if is_antibody
            else "delta_distogram_iptm_proxy"
        )
        iptm_score = _mean([_float_or_none(row.get("iptm")) for row in hero_rows])
        proxy_source = scaling_rows if scaling_rows else hero_rows
        proxy_score = _mean(
            [_float_or_none(row.get(proxy_key)) for row in proxy_source]
        )
        # With ligands, global iPTM includes target-ligand and binder-ligand
        # interfaces. Use only the explicitly extracted target-binder proxy for
        # ranking; retain global iPTM in critic_results.csv as a diagnostic.
        primary_candidates = (
            (proxy_score,) if has_target_ligands else (iptm_score, proxy_score)
        )
        primary_parts = [value for value in primary_candidates if value is not None]
        base_score = sum(primary_parts) / len(primary_parts) if primary_parts else 0.0

        delta_iptm = _mean(
            [_float_or_none(row.get("delta_iptm")) for row in hero_rows]
        )
        delta_proxy = _mean(
            [_float_or_none(row.get(delta_proxy_key)) for row in proxy_source]
        )
        selectivity_candidates = (
            (delta_proxy,) if has_target_ligands else (delta_iptm, delta_proxy)
        )
        selectivity_parts = [
            value for value in selectivity_candidates if value is not None
        ]
        selectivity_score = (
            sum(selectivity_parts) / len(selectivity_parts)
            if selectivity_parts
            else None
        )
        selection_score = base_score + selectivity_weight * (selectivity_score or 0.0)
        ranked.append(
            {
                "designed_sequence": designed_sequence,
                "binder_sequence": designed_sequence.split("|")[-1],
                "is_antibody": is_antibody,
                "target_ligands": group[0].get("target_ligands", ""),
                "iptm_score": iptm_score,
                "iptm_proxy_score": proxy_score,
                "ptm_selectivity_score": selectivity_score,
                "selection_score": selection_score,
                "seed": group[0]["seed"],
                "batch_idx": group[0]["batch_idx"],
            }
        )

    ranked.sort(key=lambda row: row["selection_score"], reverse=True)
    selected = ranked[: min(top_n, len(ranked))]
    selected_structures_dir = output_dir / "selected_structures"
    selected_structures_dir.mkdir(exist_ok=True)
    for rank, row in enumerate(selected, start=1):
        row["rank"] = rank
        candidate_rows = by_sequence[row["designed_sequence"]]
        preferred = next(
            (
                candidate
                for candidate in candidate_rows
                if candidate.get("critic_name")
                == "ESMFold2-Experimental-Cutoff2025"
            ),
            None,
        )
        if preferred is None:
            preferred = next(
                (
                    candidate
                    for candidate in candidate_rows
                    if candidate.get("is_scaling_critic", "False").lower() != "true"
                ),
                None,
            )
        if preferred is not None:
            source = (
                output_dir
                / f"seed_{int(preferred['seed']):06d}"
                / "structures"
                / (
                    f"batch_{int(preferred['batch_idx']):03d}_"
                    f"{_slug(preferred['critic_name'])}.pdb"
                )
            )
            if source.exists():
                destination = selected_structures_dir / (
                    f"{rank:04d}_score_{row['selection_score']:.4f}_"
                    f"seed_{int(row['seed']):06d}_batch_{int(row['batch_idx']):03d}.pdb"
                )
                shutil.copy2(source, destination)
                row["selected_structure"] = str(destination.relative_to(output_dir))
                source_cif = source.with_suffix(".cif")
                if source_cif.exists():
                    destination_cif = destination.with_suffix(".cif")
                    shutil.copy2(source_cif, destination_cif)
                    row["selected_structure_cif"] = str(
                        destination_cif.relative_to(output_dir)
                    )
    _write_csv(output_dir / "ranked_designs.csv", selected)
    with (output_dir / "ranked_designs.fasta").open("w") as handle:
        for row in selected:
            handle.write(
                f">rank_{row['rank']}_score_{row['selection_score']:.4f}_"
                f"seed_{row['seed']}_batch_{row['batch_idx']}\n"
                f"{row['binder_sequence']}\n"
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the official ESMFold2/ESMC binder-design protocol directly on "
            "one local CUDA GPU (for example a Vast.ai A100/H100 instance)."
        )
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-fasta", type=Path)
    target.add_argument("--target-sequence")
    target.add_argument("--target-preset", choices=sorted(TARGET_SEQUENCES))
    parser.add_argument(
        "--target-name",
        help="Output label for a custom target; defaults to the FASTA header or 'target'.",
    )
    parser.add_argument(
        "--target-modification",
        action="append",
        default=[],
        metavar="SITE[:CCD]",
        help=(
            "Target PTM, repeatable. S123/T123/Y123 infer SEP/TPO/PTR; explicit "
            "forms such as S123:SEP or 123:SEP are also accepted."
        ),
    )
    parser.add_argument(
        "--target-ligand",
        action="append",
        default=[],
        metavar="CCD",
        help=(
            "Fixed target-bound ligand as a PDB CCD code; repeat once per ligand "
            "entity (for example --target-ligand GDP --target-ligand MG)."
        ),
    )
    parser.add_argument(
        "--target-hotspot",
        action="append",
        default=[],
        metavar="S123",
        help="1-indexed target hotspot used by the epitope loss; repeatable.",
    )
    parser.add_argument(
        "--hotspot-radius",
        type=int,
        default=0,
        help=(
            "If PTMs are supplied but no hotspots are given, target each PTM plus "
            "this many flanking residues on either side (default: 0)."
        ),
    )
    parser.add_argument("--epitope-contact-distance", type=float, default=12.0)

    parser.add_argument(
        "--binder-type", choices=("minibinder", "scfv", "custom"), required=True
    )
    parser.add_argument(
        "--scfv-framework", choices=sorted(SCFV_PRESETS), default="trastuzumab"
    )
    parser.add_argument(
        "--binder-template",
        help="Custom binder prompt containing fixed amino acids and # at mutable sites.",
    )
    parser.add_argument(
        "--custom-is-antibody",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Declare whether a custom template is an antibody/scFv.",
    )
    parser.add_argument("--minibinder-length-min", type=int, default=60)
    parser.add_argument("--minibinder-length-max", type=int, default=200)

    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument(
        "--critic-set",
        choices=("minimal", "hero"),
        default="hero",
        help="'hero' reproduces the four official hero critics; 'minimal' is faster.",
    )
    parser.add_argument(
        "--hero-critic",
        action="append",
        default=[],
        help="Override --critic-set with one or more explicit checkpoint names.",
    )
    parser.add_argument(
        "--use-scaling-critics",
        action="store_true",
        help="Also score with the 15 scaling checkpoints (slow; large download).",
    )
    parser.add_argument(
        "--score-unmodified-control",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For PTM targets, also score each final binder against unmodified target.",
    )
    parser.add_argument(
        "--selection-selectivity-weight",
        type=float,
        default=0.0,
        help=(
            "Optional weight for modified-minus-unmodified score during final "
            "ranking. Default 0 reports, but does not rank by, this unvalidated proxy."
        ),
    )
    parser.add_argument("--top-n", type=int, default=84)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=COMPILE,
        help="Compile selected ESMFold2 modules (slower first job, faster campaign).",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and write the resolved run configuration without loading models.",
    )
    return parser


def _resolve_target(
    args: argparse.Namespace,
) -> tuple[str, str | None, str]:
    if args.target_preset:
        return args.target_preset, None, TARGET_SEQUENCES[args.target_preset]
    if args.target_fasta:
        fasta_header, sequence = _read_single_fasta(args.target_fasta)
        target_name = args.target_name or fasta_header.split()[0]
    else:
        sequence = args.target_sequence
        target_name = args.target_name or "target"
    assert sequence is not None
    sequence = _validate_protein_sequence(sequence, "target sequence")
    if target_name in TARGET_SEQUENCES:
        raise ValueError(
            f"Custom target name {target_name!r} collides with a built-in preset; "
            "choose another --target-name."
        )
    return target_name, sequence, sequence


def _resolve_binder(
    args: argparse.Namespace,
) -> tuple[str, str | None, bool | None]:
    if args.binder_type == "minibinder":
        if not 1 <= args.minibinder_length_min <= args.minibinder_length_max:
            raise ValueError("Invalid minibinder length range.")
        BINDER_PROMPT_FACTORIES["minibinder"] = PromptFactory(
            name="minibinder",
            template="{seq}",
            length_ranges={
                "seq": (
                    args.minibinder_length_min,
                    args.minibinder_length_max,
                )
            },
            is_antibody=False,
        )
        return "minibinder", None, False
    if args.binder_type == "scfv":
        return SCFV_PRESETS[args.scfv_framework], None, True
    if not args.binder_template:
        raise ValueError("--binder-template is required for --binder-type custom.")
    template = args.binder_template.strip().upper()
    invalid = sorted(set(template) - set(PROTEIN_1TO3) - {MUTABLE_TOKEN})
    if invalid:
        raise ValueError(
            f"Custom binder template contains unsupported symbols: {', '.join(invalid)}."
        )
    if MUTABLE_TOKEN not in template:
        raise ValueError("Custom binder template must contain at least one #.")
    return "custom", template, args.custom_is_antibody


def _resolve_target_ligands(values: list[str]) -> list[str]:
    ligand_ccds: list[str] = []
    for value in values:
        ccd = value.strip().upper()
        if not ccd or not re.fullmatch(r"[A-Z0-9_-]+", ccd):
            raise ValueError(f"Invalid ligand CCD code: {value!r}.")
        ligand_ccds.append(ccd)
    if len(ligand_ccds) > len(_CHAIN_ID_ALPHABET) - 2:
        raise ValueError(
            f"At most {len(_CHAIN_ID_ALPHABET) - 2} ligand entities are supported."
        )
    return ligand_ccds


def main(argv: list[str] | None = None) -> int:
    global STEPS, LEARNING_RATE

    args = _build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this design protocol.")
    if args.num_seeds < 1 or args.batch_size < 1:
        raise ValueError("--num-seeds and --batch-size must be >= 1.")
    STEPS = args.steps
    LEARNING_RATE = args.learning_rate
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    target_name, target_sequence_arg, resolved_target_sequence = _resolve_target(args)
    binder_name, binder_sequence, is_antibody = _resolve_binder(args)
    target_ligand_ccds = _resolve_target_ligands(args.target_ligand)
    modifications = [
        parse_target_modification(spec, resolved_target_sequence)
        for spec in args.target_modification
    ]
    modification_positions = [modification.position for modification in modifications]
    if len(modification_positions) != len(set(modification_positions)):
        raise ValueError("Each target position may be modified only once.")
    hotspots = list(args.target_hotspot)
    if modifications and not hotspots:
        hotspots = modification_hotspot_ids(
            resolved_target_sequence, modifications, args.hotspot_radius
        )
    if hotspots:
        get_epitope_mask(
            target_sequence=resolved_target_sequence,
            binder_length=1,
            target_hotspot_ids=hotspots,
            batch_size=1,
            device=torch.device("cpu"),
        )

    hero_critics = (
        args.hero_critic
        if args.hero_critic
        else (MINIMAL_CRITICS if args.critic_set == "minimal" else HERO_CRITICS)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        **vars(args),
        "target_name_resolved": target_name,
        "target_length": len(resolved_target_sequence),
        "binder_name_resolved": binder_name,
        "target_modifications_zero_indexed": modifications,
        "target_ligands_resolved": target_ligand_ccds,
        "target_hotspots_resolved": hotspots,
        "hero_critics_resolved": hero_critics,
        "tutorial_commit": TUTORIAL_COMMIT,
        "runner_version": RUNNER_VERSION,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(),
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(_jsonable(run_config), indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "target.fasta").write_text(
        f">{target_name}\n{resolved_target_sequence}\n"
    )
    (args.output_dir / "target_entities.json").write_text(
        json.dumps(
            {
                "target_protein": {
                    "id": "A",
                    "sequence": resolved_target_sequence,
                    "modifications": _jsonable(modifications),
                },
                "ligands": [
                    {"id": _CHAIN_ID_ALPHABET[i + 1], "ccd": [ccd]}
                    for i, ccd in enumerate(target_ligand_ccds)
                ],
                "binder_id": _CHAIN_ID_ALPHABET[len(target_ligand_ccds) + 1],
            },
            indent=2,
        )
        + "\n"
    )
    if args.dry_run:
        logger.info("Dry run successful. Resolved configuration: %s", args.output_dir)
        return 0

    logger.info("Loading ESMFold2/ESMC design and critic checkpoints...")
    designer = ESMFold2Design(
        hero_critic_hf_paths=hero_critics,
        compile_models=args.compile,
    )
    designer.load(use_scaling_critics=args.use_scaling_critics)

    failures = 0
    for seed in range(args.start_seed, args.start_seed + args.num_seeds):
        seed_dir = args.output_dir / f"seed_{seed:06d}"
        if args.resume and (seed_dir / "complete.json").exists():
            logger.info("Seed %d already complete; skipping.", seed)
            continue
        logger.info(
            "Starting seed %d (%s, batch_size=%d, target_length=%d, ligands=%s).",
            seed,
            binder_name,
            args.batch_size,
            len(resolved_target_sequence),
            target_ligand_ccds or "none",
        )
        try:
            best_sequences, trajectory, results = designer.design(
                target_name=target_name,
                target_sequence=target_sequence_arg,
                binder_name=binder_name,
                binder_sequence=binder_sequence,
                is_antibody=is_antibody,
                seed=seed,
                batch_size=args.batch_size,
                target_modifications=modifications or None,
                target_ligand_ccds=target_ligand_ccds or None,
                target_hotspot_ids=hotspots or None,
                epitope_contact_distance=args.epitope_contact_distance,
                score_unmodified_control=args.score_unmodified_control,
            )
            _save_seed_result(
                seed_dir=seed_dir,
                seed=seed,
                best_sequences=best_sequences,
                trajectory=trajectory,
                critic_results=results,
            )
            logger.info("Completed seed %d with %d designs.", seed, len(best_sequences))
        except Exception as exc:
            failures += 1
            seed_dir.mkdir(parents=True, exist_ok=True)
            (seed_dir / "failed.json").write_text(
                json.dumps(
                    {"seed": seed, "status": "failed", "error": repr(exc)}, indent=2
                )
                + "\n"
            )
            logger.exception("Seed %d failed.", seed)
            if args.fail_fast:
                raise
        finally:
            torch.cuda.empty_cache()

    _rank_completed_designs(
        args.output_dir,
        top_n=args.top_n,
        selectivity_weight=args.selection_selectivity_weight,
    )
    logger.info(
        "Campaign finished: %d requested seeds, %d failed. Results: %s",
        args.num_seeds,
        failures,
        args.output_dir.resolve(),
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
