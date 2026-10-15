#!/usr/bin/env python3

import argparse
import math
import csv
import gc
import re
import string
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for matplotlib, important for headless Vast.ai / cluster runs
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches

import torch

try:
    # Current README-style import
    from esm.models.esmfold2 import (
        ProteinInput,
        LigandInput,
        Modification,
        StructurePredictionInput,
        ESMFold2InputBuilder,
    )
except ImportError:
    # Fallback for installations where the builder is still under processor
    from esm.models.esmfold2 import ProteinInput, LigandInput, Modification, StructurePredictionInput
    from esm.models.esmfold2.processor import ESMFold2InputBuilder

from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model


@dataclass
class FastaRecord:
    index: int
    header: str
    sequence: str
    output_prefix: str

@dataclass(frozen=True)
class LigandSpec:
    """
    One non-polymer component, e.g. GTP, MG, ZN, or a SMILES ligand.

    Use either ccd or smiles, not both.
    If id is None, an automatic chain/entity ID is assigned.
    """
    ccd: str | list[str] | None = None
    smiles: str | None = None
    id: str | None = None


@dataclass(frozen=True)
class ModificationSpec:
    """
    Residue-level modification.

    chain_ref:
        Chain identifier after automatic assignment, e.g. "A", "B",
        or a 1-based chain number as string/int, e.g. "1", 1.

    position0:
        Zero-indexed residue position, as expected by BioHub/ESMFold2.

    ccd:
        CCD code for the modified residue, e.g. SEP, TPO, PTR.
    """
    chain_ref: str | int
    position0: int
    ccd: str


@dataclass(frozen=True)
class ParsedComplexSpec:
    protein_chains: list[str]
    ligands: list[LigandSpec]
    modifications: list[ModificationSpec]

#########################################################################################
# GPU VRAM tracing helper functions. Useful for debugging OOM issues on large complexes.
#########################################################################################

VRAM_CSV_FIELDS = [
    "cuda_baseline_allocated_gb",
    "cuda_baseline_reserved_gb",
    "cuda_peak_allocated_gb",
    "cuda_peak_reserved_gb",
    "cuda_peak_allocated_delta_gb",
    "cuda_peak_reserved_delta_gb",
]


def bytes_to_gib(x: int | float | None) -> float | None:
    if x is None:
        return None
    return float(x) / (1024 ** 3)


def format_optional_gib(x: float | None) -> str:
    if x is None:
        return ""
    return f"{x:.3f}"


def get_cuda_device_or_none(device: str | torch.device) -> torch.device | None:
    if not torch.cuda.is_available():
        return None

    device_str = str(device)

    if not device_str.startswith("cuda"):
        return None

    return torch.device(device_str)


def empty_vram_stats() -> dict[str, str]:
    return {
        "cuda_baseline_allocated_gb": "",
        "cuda_baseline_reserved_gb": "",
        "cuda_peak_allocated_gb": "",
        "cuda_peak_reserved_gb": "",
        "cuda_peak_allocated_delta_gb": "",
        "cuda_peak_reserved_delta_gb": "",
    }


def get_current_cuda_memory_gib(device: torch.device) -> dict[str, float]:
    return {
        "allocated_gb": bytes_to_gib(torch.cuda.memory_allocated(device)),
        "reserved_gb": bytes_to_gib(torch.cuda.memory_reserved(device)),
    }


def get_peak_cuda_memory_gib(device: torch.device) -> dict[str, float]:
    return {
        "peak_allocated_gb": bytes_to_gib(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_gb": bytes_to_gib(torch.cuda.max_memory_reserved(device)),
    }


def summarize_vram_stats(
    baseline: dict[str, float] | None,
    peak: dict[str, float] | None,
) -> dict[str, str]:
    if baseline is None or peak is None:
        return empty_vram_stats()

    baseline_allocated = baseline["allocated_gb"]
    baseline_reserved = baseline["reserved_gb"]
    peak_allocated = peak["peak_allocated_gb"]
    peak_reserved = peak["peak_reserved_gb"]

    return {
        "cuda_baseline_allocated_gb": format_optional_gib(baseline_allocated),
        "cuda_baseline_reserved_gb": format_optional_gib(baseline_reserved),
        "cuda_peak_allocated_gb": format_optional_gib(peak_allocated),
        "cuda_peak_reserved_gb": format_optional_gib(peak_reserved),
        "cuda_peak_allocated_delta_gb": format_optional_gib(
            peak_allocated - baseline_allocated
        ),
        "cuda_peak_reserved_delta_gb": format_optional_gib(
            peak_reserved - baseline_reserved
        ),
    }

########################################################################################
# PAE extraction and saving helper functions. Useful for downstream analysis of predicted complexes.
########################################################################################
PAE_CSV_FIELDS = [
    "pae_available",
    "pae_npz_file",
    "pae_npz_key",
    "pae_plot_file",
]

def report_pae_availability(results: list, record_header: str) -> None:
    n_with_pae = sum(
        getattr(result, "pae", None) is not None
        for result in results
    )

    if n_with_pae == 0:
        print(
            f"  WARNING: no PAE matrices returned for {record_header!r}. "
            "This local model/output path may not emit 'pae'."
        )
    else:
        print(
            f"  PAE available for {n_with_pae}/{len(results)} model(s)."
        )

def tensor_like_to_numpy(x) -> np.ndarray | None:
    if x is None:
        return None

    if hasattr(x, "detach"):
        x = x.detach()

    if hasattr(x, "cpu"):
        x = x.cpu()

    arr = np.asarray(x)

    return arr


def get_pae_array(result) -> np.ndarray | None:
    """
    Extract a single square PAE matrix from an ESMFold2 result object.

    Expected common case:
        result.pae has shape (N, N)

    Defensive handling:
        result.pae may occasionally have shape (1, N, N).
    """
    pae = getattr(result, "pae", None)

    if pae is None:
        return None

    arr = tensor_like_to_numpy(pae)

    if arr is None:
        return None

    arr = np.squeeze(arr)

    if arr.ndim != 2:
        print(f"  WARNING: PAE has unexpected shape {arr.shape}; skipping PAE output.")
        return None

    if arr.shape[0] != arr.shape[1]:
        print(f"  WARNING: PAE is not square: shape={arr.shape}; skipping PAE output.")
        return None

    return arr.astype(np.float32, copy=False)


def make_pae_component_boundaries(
    chain_ids: list[str],
    protein_lengths: list[int],
    pae_size: int,
) -> list[tuple[int, int, str]]:
    """
    Create component ranges for chain-boundary overlays.

    Protein chains are assumed to come first, matching our input construction.
    If PAE contains additional tokens/components, the remainder is labeled
    as ligand/other.
    """
    components = []
    start = 0

    for chain_id, length in zip(chain_ids, protein_lengths):
        end = start + int(length)

        if end > pae_size:
            # Chain-length bookkeeping does not match PAE dimension.
            return []

        components.append((start, end, chain_id))
        start = end

    if start < pae_size:
        components.append((start, pae_size, "ligand/other"))

    return components


def draw_pae_boundaries(
    ax,
    components: list[tuple[int, int, str]],
    pae_size: int,
) -> None:
    """
    Draw chain/component separators and optional midpoint tick labels.
    """
    if not components:
        return

    for start, end, label in components:
        if start == 0:
            continue

        ax.axhline(start - 0.5, color="black", linewidth=0.8)
        ax.axvline(start - 0.5, color="black", linewidth=0.8)

    # Only label components if this will remain readable.
    if len(components) <= 12:
        mids = [(start + end - 1) / 2 for start, end, _ in components]
        labels = [label for _, _, label in components]

        ax.set_xticks(mids)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticks(mids)
        ax.set_yticklabels(labels, fontsize=7)


def save_combined_pae_plot(
    pae_by_model: dict[int, np.ndarray],
    plot_file: Path,
    chain_ids: list[str],
    protein_lengths: list[int],
    title_prefix: str,
    pae_max: float = 30.0,
) -> None:
    """
    Save one PNG per FASTA record.

    If there are 5 diffusion samples/models, this creates one PNG containing
    5 PAE subplots.
    """
    if not pae_by_model:
        return

    model_items = sorted(pae_by_model.items())
    n_models = len(model_items)

    ncols = min(3, n_models)
    nrows = math.ceil(n_models / ncols)

    fig_width = 4.2 * ncols
    fig_height = 4.5 * nrows

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    last_im = None

    for ax in axes.flat:
        ax.axis("off")

    for ax, (model_index, pae) in zip(axes.flat, model_items):
        ax.axis("on")

        pae_size = pae.shape[0]
        components = make_pae_component_boundaries(
            chain_ids=chain_ids,
            protein_lengths=protein_lengths,
            pae_size=pae_size,
        )

        last_im = ax.imshow(
            pae,
            cmap="bwr",
            vmin=0,
            vmax=pae_max,
            origin="upper",
            aspect="equal",
        )

        draw_pae_boundaries(ax, components, pae_size)

        ax.set_title(f"model {model_index}", fontsize=10)
        ax.set_xlabel("Scored residue/token", fontsize=8)
        ax.set_ylabel("Aligned residue/token", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)

    fig.suptitle(title_prefix, fontsize=12, fontweight="bold")

    if last_im is not None:
        cbar = fig.colorbar(
            last_im,
            ax=axes.ravel().tolist(),
            fraction=0.025,
            pad=0.02,
        )
        cbar.set_label("Predicted aligned error (Å)", fontsize=9)
        cbar.set_ticks([0, 5, 10, 15, 20, 25, 30])
        cbar.ax.invert_yaxis()

    fig.savefig(plot_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_pae_outputs_for_results(
    results: list,
    output_dir: Path,
    output_prefix: str,
    chain_ids: list[str],
    protein_lengths: list[int],
) -> tuple[str, str, dict[int, str]]:
    """
    Save PAE outputs for one FASTA record.

    Returns:
        pae_npz_file_str:
            Path to compressed NPZ file, or "" if no PAE was available.

        pae_plot_file_str:
            Path to combined PNG file, or "" if no PAE was available.

        pae_npz_keys:
            Mapping model_index -> key inside NPZ, e.g. {1: "pae_model_1"}.
    """
    pae_by_model: dict[int, np.ndarray] = {}

    for model_index, result in enumerate(results, start=1):
        pae = get_pae_array(result)

        if pae is not None:
            pae_by_model[model_index] = pae

    if not pae_by_model:
        return "", "", {}

    pae_npz_file = output_dir / f"{output_prefix}_pae.npz"
    pae_plot_file = output_dir / f"{output_prefix}_pae.png"

    npz_payload = {}
    pae_npz_keys: dict[int, str] = {}

    for model_index, pae in pae_by_model.items():
        key = f"pae_model_{model_index}"
        npz_payload[key] = pae
        pae_npz_keys[model_index] = key

    npz_payload["available_model_indices"] = np.array(
        sorted(pae_by_model.keys()),
        dtype=np.int16,
    )

    npz_payload["protein_chain_ids"] = np.array(chain_ids, dtype=object)
    npz_payload["protein_chain_lengths"] = np.array(protein_lengths, dtype=np.int32)

    np.savez_compressed(pae_npz_file, **npz_payload)

    save_combined_pae_plot(
        pae_by_model=pae_by_model,
        plot_file=pae_plot_file,
        chain_ids=chain_ids,
        protein_lengths=protein_lengths,
        title_prefix=output_prefix,
    )

    return str(pae_npz_file), str(pae_plot_file), pae_npz_keys

########################################################################################
# FASTA parsing and complex sequence handling.
########################################################################################

def sanitize_for_filename(text: str, max_len: int = 120) -> str:
    """
    Convert a FASTA header into a safe filename stem.
    Keeps letters, numbers, dots, dashes and underscores.
    """
    text = text.strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-")

    if not text:
        text = "unnamed_complex"

    return text[:max_len]


def parse_fasta(fasta_path: Path) -> list[tuple[str, str]]:
    records = []
    header = None
    seq_lines = []

    with open(fasta_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_lines)))
                header = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line)

    if header is not None:
        records.append((header, "".join(seq_lines)))

    if not records:
        raise ValueError(f"No FASTA records found in {fasta_path}")

    return records


def load_fasta_records(fasta_path: Path) -> list[FastaRecord]:
    raw_records = parse_fasta(fasta_path)
    records = []

    for i, (header, sequence) in enumerate(raw_records, start=1):
        safe_header = sanitize_for_filename(header)
        output_prefix = f"{i:05d}_{safe_header}"

        records.append(
            FastaRecord(
                index=i,
                header=header,
                sequence=sequence,
                output_prefix=output_prefix,
            )
        )

    return records


def parse_complex_sequence(sequence: str) -> ParsedComplexSpec:
    """
    Parse one FASTA record into protein chains, ligands, and modifications.
    Backward-compatible legacy syntax:

        Monomer:
            MSEQUENCECHAINA

        Multimer:
            MSEQUENCECHAINA:MGSEQUENCECHAINB

        Also accepted:
            MSEQUENCECHAINA|MGSEQUENCECHAINB

    New typed syntax:
    
        PROT:MSEQUENCECHAINA|CCD:GTP|CCD:MG

        PROT:MSEQUENCECHAINA|PROT:MGSEQUENCECHAINB|CCD:GTP

        PROT:MSEQUENCECHAINA|MOD1:A:15:SEP|CCD:GTP

    PTM notation:

        MOD1:A:15:SEP
            chain A, residue 15 using human-readable 1-based numbering

        MOD0:A:14:SEP
            chain A, residue 14 using BioHub/internal 0-based numbering
    """
    raw = "".join(sequence.split())

    if not raw:
        raise ValueError("FASTA record contains an empty sequence.")

    # Detect new typed syntax.
    tokens_pipe = raw.split("|")
    typed_prefixes = (
        "PROT:",
        "PROTEIN:",
        "CCD:",
        "LIGAND_CCD:",
        "ION_CCD:",
        "SMILES:",
        "LIGAND_SMILES:",
        "MOD0:",
        "MOD1:",
    )

    is_typed = any(
        token.upper().startswith(typed_prefixes)
        for token in tokens_pipe
    )

    if not is_typed:
        # Legacy mode: all fields are protein chains.
        seq = raw.replace("|", ":").upper()

        if ":" in seq:
            chains = seq.split(":")

            empty_chain_positions = [
                i + 1 for i, chain in enumerate(chains) if not chain
            ]

            if empty_chain_positions:
                raise ValueError(
                    "FASTA record contains empty chain(s) after splitting by ':' or '|'. "
                    f"Empty chain position(s): {empty_chain_positions}"
                )
        else:
            chains = [seq]

        return ParsedComplexSpec(
            protein_chains=chains,
            ligands=[],
            modifications=[],
        )

    # Typed mode: use only "|" between entries.
    protein_chains: list[str] = []
    ligands: list[LigandSpec] = []
    modifications: list[ModificationSpec] = []

    for token in tokens_pipe:
        if not token:
            raise ValueError(
                "Empty token in typed FASTA record. "
                "Check for duplicated or trailing '|'."
            )

        if ":" not in token:
            raise ValueError(
                f"Typed FASTA token {token!r} lacks ':'. "
                "Expected e.g. PROT:SEQUENCE, CCD:GTP, MOD1:A:15:SEP."
            )

        key, value = token.split(":", 1)
        key = key.strip().upper()
        value = value.strip()

        if not value:
            raise ValueError(f"Typed FASTA token {token!r} has an empty value.")

        if key in {"PROT", "PROTEIN"}:
            seq = value.upper()

            if ":" in seq:
                raise ValueError(
                    "In typed FASTA mode, separate entries with '|', not ':'. "
                    "Use PROT:CHAINA|PROT:CHAINB, not PROT:CHAINA:CHAINB."
                )

            protein_chains.append(seq)

        elif key in {"CCD", "LIGAND_CCD", "ION_CCD"}:
            # Default: each CCD code is treated as an independent ligand/entity.
            # Example CCD:GTP,MG -> LigandInput(... ccd=['GTP']) and LigandInput(... ccd=['MG'])
            ccds = [x.strip().upper() for x in value.split(",") if x.strip()]

            if not ccds:
                raise ValueError(f"No CCD code found in token {token!r}.")

            for ccd in ccds:
                ligands.append(LigandSpec(ccd=ccd))

        elif key in {"SMILES", "LIGAND_SMILES"}:
            # Preserve case/syntax of SMILES.
            ligands.append(LigandSpec(smiles=value))

        elif key in {"MOD0", "MOD1"}:
            # Expected value:
            #   A:14:SEP  for MOD0
            #   A:15:SEP  for MOD1
            parts = value.split(":")

            if len(parts) != 3:
                raise ValueError(
                    f"Invalid modification token {token!r}. "
                    "Expected MOD1:<chain>:<position>:<ccd>, "
                    "for example MOD1:A:15:SEP."
                )

            chain_ref, pos_text, ccd = parts
            chain_ref = chain_ref.strip().upper()
            ccd = ccd.strip().upper()

            try:
                pos = int(pos_text)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid modification position {pos_text!r} in token {token!r}."
                ) from exc

            if key == "MOD1":
                if pos < 1:
                    raise ValueError(
                        f"MOD1 uses 1-based residue numbering; got {pos}."
                    )
                position0 = pos - 1
            else:
                if pos < 0:
                    raise ValueError(
                        f"MOD0 uses 0-based residue numbering; got {pos}."
                    )
                position0 = pos

            modifications.append(
                ModificationSpec(
                    chain_ref=chain_ref,
                    position0=position0,
                    ccd=ccd,
                )
            )

        else:
            raise ValueError(
                f"Unknown typed FASTA token {key!r}. "
                "Allowed token types: PROT, CCD, SMILES, MOD0, MOD1."
            )

    if not protein_chains:
        raise ValueError("Typed FASTA record contains no PROT:<sequence> entry.")

    return ParsedComplexSpec(
        protein_chains=protein_chains,
        ligands=ligands,
        modifications=modifications,
    )


def split_complex_sequence(sequence: str) -> list[str]:
    """
    Backward-compatible wrapper.

    Existing code that only expects protein chains can still call this.
    New v4 code should preferably call parse_complex_sequence().
    """
    return parse_complex_sequence(sequence).protein_chains


def validate_protein_sequence(seq: str, chain_id: str) -> None:
    """
    Conservative validation for protein sequences.
    Allows common ambiguous/special amino-acid symbols as well.
    """
    allowed = set("ACDEFGHIKLMNPQRSTVWYBXZUOJ")
    invalid = sorted(set(seq) - allowed)

    if invalid:
        raise ValueError(
            f"Chain {chain_id} contains invalid protein sequence characters: {invalid}"
        )


def make_chain_ids(n: int) -> list[str]:
    """
    Generates A, B, C, ..., Z.

    For most protein:protein complex screening jobs this is sufficient.
    """
    alphabet = list(string.ascii_uppercase)

    if n > len(alphabet):
        raise ValueError(
            f"Too many chains ({n}). Max supported in this script is {len(alphabet)}."
        )

    return alphabet[:n]


def resolve_chain_ref(chain_ref: str | int, chain_ids: list[str]) -> str:
    """
    Resolve a modification chain reference to an actual protein chain ID.

    Accepted:
        "A", "B", ...
        1, 2, ...
        "1", "2", ...
    """
    if isinstance(chain_ref, int):
        idx = chain_ref - 1
        if idx < 0 or idx >= len(chain_ids):
            raise ValueError(
                f"Invalid chain number {chain_ref}. "
                f"Available protein chains: {chain_ids}"
            )
        return chain_ids[idx]

    ref = str(chain_ref).strip().upper()

    if ref in chain_ids:
        return ref

    if ref.isdigit():
        idx = int(ref) - 1
        if idx < 0 or idx >= len(chain_ids):
            raise ValueError(
                f"Invalid chain number {ref}. "
                f"Available protein chains: {chain_ids}"
            )
        return chain_ids[idx]

    raise ValueError(
        f"Invalid chain reference {chain_ref!r}. "
        f"Available protein chains: {chain_ids}"
    )


def normalize_ccd_list(ccd: str | list[str]) -> list[str]:
    if isinstance(ccd, str):
        ccds = [ccd]
    else:
        ccds = list(ccd)

    ccds = [x.strip().upper() for x in ccds if x and x.strip()]

    if not ccds:
        raise ValueError("Ligand CCD list is empty.")

    return ccds

def summarize_ligands(ligands: list[LigandSpec]) -> str:
    """
    Compact CSV-friendly ligand summary.

    Examples:
        CCD:GTP;CCD:MG
        SMILES:CCO
    """
    parts = []

    for ligand in ligands:
        if ligand.ccd is not None:
            if isinstance(ligand.ccd, str):
                ccd_text = ligand.ccd
            else:
                ccd_text = ",".join(ligand.ccd)
            parts.append(f"CCD:{ccd_text}")
        elif ligand.smiles is not None:
            parts.append(f"SMILES:{ligand.smiles}")
        else:
            parts.append("UNKNOWN")

    return ";".join(parts)


def summarize_modifications(modifications: list[ModificationSpec]) -> str:
    """
    Compact CSV-friendly PTM summary.

    Uses zero-indexed positions internally because this is what BioHub/ESMFold2 expects.
    """
    return ";".join(
        f"{mod.chain_ref}:{mod.position0}:{mod.ccd}"
        for mod in modifications
    )

def build_structure_prediction_input(
    chains: list[str],
    ligands: list[LigandSpec] | None = None,
    modifications: list[ModificationSpec] | None = None,
) -> tuple[StructurePredictionInput, list[str]]:
    """
    Build BioHub/ESMFold2 StructurePredictionInput from:

        - protein chains
        - optional free ligands / ions
        - optional residue-level PTMs

    Returns:
        spi:
            StructurePredictionInput for ESMFold2.

        chain_ids:
            Protein chain IDs only. Ligand IDs are not included here,
            because downstream CSV/logging usually cares about protein chains.
    """
    if not chains:
        raise ValueError("At least one protein chain is required.")

    ligands = ligands or []
    modifications = modifications or []

    # Assign IDs to proteins first, then ligands.
    # Example:
    #   2 protein chains + 2 ligands -> A, B, C, D
    all_ids = make_chain_ids(len(chains) + len(ligands))
    chain_ids = all_ids[:len(chains)]
    auto_ligand_ids = all_ids[len(chains):]

    # Validate protein chains.
    clean_chains: list[str] = []
    for chain_id, seq in zip(chain_ids, chains):
        seq = seq.strip().upper()
        validate_protein_sequence(seq, chain_id)
        clean_chains.append(seq)

    # Collect modifications by resolved protein chain ID.
    mods_by_chain_id: dict[str, list[Modification]] = {
        chain_id: [] for chain_id in chain_ids
    }

    for mod in modifications:
        chain_id = resolve_chain_ref(mod.chain_ref, chain_ids)
        chain_idx = chain_ids.index(chain_id)
        chain_len = len(clean_chains[chain_idx])

        if mod.position0 < 0 or mod.position0 >= chain_len:
            raise ValueError(
                f"Modification position {mod.position0} is out of range for "
                f"chain {chain_id} with length {chain_len}. "
                "BioHub/ESMFold2 modification positions are zero-indexed."
            )

        mods_by_chain_id[chain_id].append(
            Modification(
                position=mod.position0,
                ccd=mod.ccd.strip().upper(),
            )
        )

    sequence_inputs = []

    # Protein chains with optional PTMs.
    for chain_id, seq in zip(chain_ids, clean_chains):
        chain_mods = mods_by_chain_id[chain_id] or None

        sequence_inputs.append(
            ProteinInput(
                id=chain_id,
                sequence=seq,
                modifications=chain_mods,
            )
        )

    # Free ligands / ions.
    used_ids = set(chain_ids)

    for auto_id, ligand in zip(auto_ligand_ids, ligands):
        ligand_id = ligand.id.strip().upper() if ligand.id else auto_id

        if ligand_id in used_ids:
            raise ValueError(
                f"Duplicate chain/entity ID {ligand_id!r}. "
                "Protein and ligand IDs must be unique."
            )

        used_ids.add(ligand_id)

        has_ccd = ligand.ccd is not None
        has_smiles = ligand.smiles is not None

        if has_ccd == has_smiles:
            raise ValueError(
                f"Ligand {ligand_id!r} must define exactly one of ccd or smiles."
            )

        if has_ccd:
            sequence_inputs.append(
                LigandInput(
                    id=ligand_id,
                    ccd=normalize_ccd_list(ligand.ccd),
                )
            )
        else:
            smiles = ligand.smiles.strip()
            if not smiles:
                raise ValueError(f"Ligand {ligand_id!r} has an empty SMILES string.")

            sequence_inputs.append(
                LigandInput(
                    id=ligand_id,
                    smiles=smiles,
                )
            )

    spi = StructurePredictionInput(sequences=sequence_inputs)

    return spi, chain_ids


def ensure_list(result_or_results):
    if isinstance(result_or_results, list):
        return result_or_results
    return [result_or_results]


def scalar_to_float(value):
    if value is None:
        return None

    if hasattr(value, "detach"):
        return float(value.detach().cpu().item())

    return float(value)


def mean_plddt_to_float(plddt):
    if plddt is None:
        return None

    if hasattr(plddt, "detach"):
        return float(plddt.detach().cpu().mean().item())

    return float(plddt.mean())


def format_optional_float(value: float | None, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if value is not None else ""


def format_optional_float_for_print(value: float | None, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if value is not None else "NA"


def synchronize_cuda_if_needed(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def expected_cif_files(
    output_dir: Path,
    output_prefix: str,
    num_diffusion_samples: int,
) -> list[Path]:
    return [
        output_dir / f"{output_prefix}_model_{model_index}.cif"
        for model_index in range(1, num_diffusion_samples + 1)
    ]


def all_expected_outputs_exist(
    output_dir: Path,
    output_prefix: str,
    num_diffusion_samples: int,
) -> bool:
    return all(
        path.exists() and path.stat().st_size > 0
        for path in expected_cif_files(output_dir, output_prefix, num_diffusion_samples)
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
        "Fold many protein monomers or multimers from one multi-entry FASTA file "
        "with local ESMFold2. One FASTA record corresponds to one prediction target. "
        "Monomers can be provided as plain sequences; multimer chains can be "
        "separated by ':' or '|'."
        )
    )

    parser.add_argument(
        "--input_fasta",
        required=True,
        help="Multi-entry FASTA. Use a plain sequence for monomers, or separate multimer chains by ':' or '|'.",
    )
    parser.add_argument(
        "--output_dir",
        default="esmfold2_outputs",
        help="Directory for CIF files and CSV log.",
    )
    parser.add_argument(
        "--num_loops",
        type=int,
        default=20,
        help="Number of loops/recycles passed to ESMFold2.",
    )
    parser.add_argument(
        "--num_sampling_steps",
        type=int,
        default=100,
        help="Number of diffusion sampling steps.",
    )
    parser.add_argument(
        "--num_diffusion_samples",
        type=int,
        default=5,
        help="Number of predicted models per complex.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed. FASTA record index is added to this value.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use, e.g. cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip complexes for which all expected CIF files already exist.",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Abort immediately if one complex fails. Default: continue with next complex.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=1,
        help="1-based FASTA record index to start from.",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="1-based FASTA record index to end at, inclusive.",
    )

    args = parser.parse_args()

    input_fasta = Path(args.input_fasta)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_fasta_records(input_fasta)

    if args.end_index is not None:
        records = [
            record
            for record in records
            if args.start_index <= record.index <= args.end_index
        ]
    else:
        records = [
            record
            for record in records
            if record.index >= args.start_index
        ]

    if not records:
        raise ValueError("No FASTA records selected after applying start/end index filters.")

    print(f"Selected {len(records)} FASTA records for prediction.")
    print(f"Loading ESMFold2 on {args.device}...")

    model = ESMFold2Model.from_pretrained("biohub/ESMFold2").to(args.device).eval()
    builder = ESMFold2InputBuilder()

    cuda_device = get_cuda_device_or_none(args.device)

    log_csv = output_dir / "prediction_log.csv"

    fieldnames = [
        "record_index",
        "header",
        "output_prefix",
        "status",
        "model_index",
        "num_protein_chains",
        "num_ligands",
        "num_modifications",
        "chain_ids",
        "sequence_lengths",
        "ligands",
        "modifications",
        "seed",
        "num_loops",
        "num_sampling_steps",
        "num_diffusion_samples",

        # Runtime / VRAM
        "start_time_unix",
        "end_time_unix",
        "elapsed_seconds",
        "cuda_baseline_allocated_gb",
        "cuda_baseline_reserved_gb",
        "cuda_peak_allocated_gb",
        "cuda_peak_reserved_gb",
        "cuda_peak_allocated_delta_gb",
        "cuda_peak_reserved_delta_gb",

        # Confidence
        "mean_plddt",
        "ptm",
        "iptm",

        # Outputs
        "cif_file",
        "pae_available",
        "pae_npz_file",
        "pae_npz_key",
        "pae_plot_file",

        "error",
    ]

    write_header = not log_csv.exists() or log_csv.stat().st_size == 0

    with open(log_csv, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        for record in records:
            start_time = None
            end_time = None
            spi = None
            results = None

            chain_ids_str = ""
            sequence_lengths = ""
            ligands_str = ""
            modifications_str = ""
            num_protein_chains = ""
            num_ligands = ""
            num_modifications = ""

            vram_baseline = None
            vram_stats = empty_vram_stats()

            pae_npz_file = ""
            pae_plot_file = ""
            pae_npz_keys = {}

            try:
                if args.resume and all_expected_outputs_exist(
                    output_dir,
                    record.output_prefix,
                    args.num_diffusion_samples,
                ):
                    print(
                        f"\nSkipping record {record.index}: {record.header} "
                        f"because expected CIF files already exist."
                    )

                    writer.writerow(
                        {
                            "record_index": record.index,
                            "header": record.header,
                            "output_prefix": record.output_prefix,
                            "status": "skipped_existing_outputs",
                            "model_index": "",
                            "num_protein_chains": "",
                            "num_ligands": "",
                            "num_modifications": "",
                            "chain_ids": "",
                            "sequence_lengths": "",
                            "ligands": "",
                            "modifications": "",
                            "seed": args.seed + record.index,
                            "num_loops": args.num_loops,
                            "num_sampling_steps": args.num_sampling_steps,
                            "num_diffusion_samples": args.num_diffusion_samples,
                            "start_time_unix": "",
                            "end_time_unix": "",
                            "elapsed_seconds": "",

                            **empty_vram_stats(),

                            "mean_plddt": "",
                            "ptm": "",
                            "iptm": "",

                            "cif_file": "",
                            "pae_available": "",
                            "pae_npz_file": "",
                            "pae_npz_key": "",
                            "pae_plot_file": "",

                            "error": "",
                        }
                    )
                    csvfile.flush()
                    continue

                complex_spec = parse_complex_sequence(record.sequence)

                spi, chain_ids = build_structure_prediction_input(
                    chains=complex_spec.protein_chains,
                    ligands=complex_spec.ligands,
                    modifications=complex_spec.modifications,
                )

                chain_ids_str = ",".join(chain_ids)
                sequence_lengths = ",".join(str(len(seq)) for seq in complex_spec.protein_chains)
                ligands_str = summarize_ligands(complex_spec.ligands)
                modifications_str = summarize_modifications(complex_spec.modifications)

                num_protein_chains = len(complex_spec.protein_chains)
                num_ligands = len(complex_spec.ligands)
                num_modifications = len(complex_spec.modifications)

                per_complex_seed = args.seed + record.index

                print(
                    f"\nFolding record {record.index}: {record.header}\n"
                    f"  output_prefix={record.output_prefix}\n"
                    f"  protein_chains={chain_ids_str}\n"
                    f"  protein_lengths={sequence_lengths}\n"
                    f"  ligands={ligands_str or 'none'}\n"
                    f"  modifications={modifications_str or 'none'}\n"
                    f"  seed={per_complex_seed}"
                )

                # Clean up stale allocator cache from the previous target before measuring.
                if cuda_device is not None:
                    torch.cuda.empty_cache()
                    synchronize_cuda_if_needed(args.device)
                    torch.cuda.reset_peak_memory_stats(cuda_device)
                    vram_baseline = get_current_cuda_memory_gib(cuda_device)
                else:
                    vram_baseline = None 

                synchronize_cuda_if_needed(args.device)
                start_time = time.time()

                with torch.inference_mode():
                    results = builder.fold(
                        model,
                        spi,
                        num_loops=args.num_loops,
                        num_sampling_steps=args.num_sampling_steps,
                        num_diffusion_samples=args.num_diffusion_samples,
                        seed=per_complex_seed,
                        complex_id=record.output_prefix,
                    )

                synchronize_cuda_if_needed(args.device)
                end_time = time.time()

                elapsed_seconds = end_time - start_time

                if cuda_device is not None:
                    vram_peak = get_peak_cuda_memory_gib(cuda_device)
                    vram_stats = summarize_vram_stats(vram_baseline, vram_peak)
                else:
                    vram_stats = empty_vram_stats()

                results = ensure_list(results)

                report_pae_availability(results, record.header)

                protein_lengths_list = [
                    len(seq) for seq in complex_spec.protein_chains
                ]

                try:
                    pae_npz_file, pae_plot_file, pae_npz_keys = save_pae_outputs_for_results(
                        results=results,
                        output_dir=output_dir,
                        output_prefix=record.output_prefix,
                        chain_ids=chain_ids,
                        protein_lengths=protein_lengths_list,
                    )
                except Exception as pae_exc:
                    print(
                        f"  WARNING: PAE output failed for {record.header!r}: "
                        f"{type(pae_exc).__name__}: {pae_exc}"
                    )
                    pae_npz_file = ""
                    pae_plot_file = ""
                    pae_npz_keys = {}

                for model_index, result in enumerate(results, start=1):
                    mean_plddt = mean_plddt_to_float(getattr(result, "plddt", None))
                    ptm = scalar_to_float(getattr(result, "ptm", None))
                    iptm = scalar_to_float(getattr(result, "iptm", None))

                    cif_file = output_dir / f"{record.output_prefix}_model_{model_index}.cif"

                    with open(cif_file, "w") as f:
                        f.write(result.complex.to_mmcif())
                    
                    pae_npz_key = pae_npz_keys.get(model_index, "")

                    writer.writerow(
                        {
                            "record_index": record.index,
                            "header": record.header,
                            "output_prefix": record.output_prefix,
                            "status": "success",
                            "model_index": model_index,
                            "num_protein_chains": num_protein_chains,
                            "num_ligands": num_ligands,
                            "num_modifications": num_modifications,
                            "chain_ids": chain_ids_str,
                            "sequence_lengths": sequence_lengths,
                            "ligands": ligands_str,
                            "modifications": modifications_str,
                            "seed": per_complex_seed,
                            "num_loops": args.num_loops,
                            "num_sampling_steps": args.num_sampling_steps,
                            "num_diffusion_samples": args.num_diffusion_samples,
                            "start_time_unix": f"{start_time:.6f}",
                            "end_time_unix": f"{end_time:.6f}",
                            "elapsed_seconds": f"{elapsed_seconds:.6f}",

                            **vram_stats,

                            "mean_plddt": format_optional_float(mean_plddt),
                            "ptm": format_optional_float(ptm),
                            "iptm": format_optional_float(iptm),

                            "cif_file": str(cif_file),
                            "pae_available": "yes" if pae_npz_key else "no",
                            "pae_npz_file": pae_npz_file,
                            "pae_npz_key": pae_npz_key,
                            "pae_plot_file": pae_plot_file,

                            "error": "",
                        }
                    )

                    print(
                        f"  model {model_index}: "
                        f"pLDDT={format_optional_float_for_print(mean_plddt)} "
                        f"pTM={format_optional_float_for_print(ptm)} "
                        f"ipTM={format_optional_float_for_print(iptm)} "
                        f"batch_time={elapsed_seconds:.2f}s "
                        f"peak_allocated={vram_stats['cuda_peak_allocated_gb'] or 'NA'}GB "
                        f"peak_reserved={vram_stats['cuda_peak_reserved_gb'] or 'NA'}GB "
                        f"PAE={'yes' if pae_npz_key else 'no'}"
                    )

                csvfile.flush()

            except Exception as exc:
                end_time = time.time()
                elapsed_seconds = (
                    end_time - start_time
                    if start_time is not None
                    else None
                )

                error_message = f"{type(exc).__name__}: {exc}"
                print(f"\nERROR in record {record.index}: {record.header}")
                print(error_message)

                try:
                    if cuda_device is not None and vram_baseline is not None:
                        synchronize_cuda_if_needed(args.device)
                        vram_peak = get_peak_cuda_memory_gib(cuda_device)
                        vram_stats = summarize_vram_stats(vram_baseline, vram_peak)
                except Exception:
                    vram_stats = empty_vram_stats()

                writer.writerow(
                    {
                        "record_index": record.index,
                        "header": record.header,
                        "output_prefix": record.output_prefix,
                        "status": "failed",
                        "model_index": "",
                        "num_protein_chains": num_protein_chains,
                        "num_ligands": num_ligands,
                        "num_modifications": num_modifications,
                        "chain_ids": chain_ids_str,
                        "sequence_lengths": sequence_lengths,
                        "ligands": ligands_str,
                        "modifications": modifications_str,
                        "seed": args.seed + record.index,
                        "num_loops": args.num_loops,
                        "num_sampling_steps": args.num_sampling_steps,
                        "num_diffusion_samples": args.num_diffusion_samples,
                        "start_time_unix": f"{start_time:.6f}" if start_time else "",
                        "end_time_unix": f"{end_time:.6f}",
                        "elapsed_seconds": (
                            f"{elapsed_seconds:.6f}"
                            if elapsed_seconds is not None
                            else ""
                        ),

                        **vram_stats,

                        "mean_plddt": "",
                        "ptm": "",
                        "iptm": "",

                        "cif_file": "",
                        "pae_available": "",
                        "pae_npz_file": "",
                        "pae_npz_key": "",
                        "pae_plot_file": "",

                        "error": error_message,
                    }
                )
                csvfile.flush()

                traceback.print_exc()

                if args.fail_fast:
                    raise

            finally:
                try:
                    del spi
                except UnboundLocalError:
                    pass

                try:
                    del results
                except UnboundLocalError:
                    pass

                gc.collect()

                if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                    torch.cuda.empty_cache()

    print(f"\nDone. Log written to: {log_csv}")


if __name__ == "__main__":
    main()