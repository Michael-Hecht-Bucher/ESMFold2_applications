# ESMFold2_applications
This repo contains runner scripts for ESMFold2-based high-throughput protein:protein and protein:ligand structure prediction as well as ESMFold2-based protein minibinder design (including ligands and PTMs)

## 1. ESMFold2-based protein:protein interaction screening
This protocol is centered around the runner script `ESMFold2_fold_complexes_from_fasta.py` and shows the conda environment setup, input preparation and run execution below. 
Note that the required GPU VRAM for ESMFold2 scales approximately with L<sup>2</sup> (L ... token count = sum of all protein and ligand residues). Additionally, ESMFold2 shows approximately cubic runtime scaling, i.e. runtime ~ L<sup>3</sup>. <br>
For an A100 80GB GPU, the estimated maximum residue number is 1070 if `--num_diffusion_samples = 5` is used. If batching with 5 parallel models is omitted and all models are predicted sequentially using `--num_diffusion_samples = 1`, the maximum residue number is estimated as 1800 - 1850 on an A100 80GB. 

### 1.1 Prepare conda environment 
```python
conda create -n esmfold2 python=3.12 pip -c conda-forge -y
conda activate esmfold2
python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install "esm@git+https://github.com/Biohub/esm.git@main"
pip install numpy matplotlib pandas
```
### 1.2 Prepare input FASTA file 
For protein:protein interaction screening, create a combined input FASTA file containing all bait:prey protein complex sequences that should be screened. If ligands should be included (e.g. if the bait protein binds to ATP:Mg<sup>2+</sup>), additionally append `"|CCD:ATP|CCD:MG"` to each FASTA entry. <br>
#### 1.2.1 Append bait protein sequence to a list of putative target sequences
To add your bait protein sequence to a collection of putative interactors in FASTA format, execute `Append-bait-seq_and_curate.py`. This script additionally allows for maximum total complex size filtering to avoid subsequent OOM issues. This gives: 
```
>complex1
prey1:bait
>complex2
prey2:bait
>complex3
prey3:bait
...
```

#### 1.2.2 Optionally add ligands
To additionally add ligands to your input FASTA, execute `add_ligands_to_input.py` and specify the required ligands in the script.
```
ligand1 = "ATP"
ligand2 = "MG"
```
This script will additionally reformat your input FASTA to allow correct parsing of protein and ligand entities by the runner script.
```
>complex1
PROT:{prey1_sequence}|PROT:{bait_sequence}|CCD:{ligand1}|CCD:{ligand2}
>complex2
PROT:{prey2_sequence}|PROT:{bait_sequence}|CCD:{ligand1}|CCD:{ligand2}
>complex3
PROT:{prey3_sequence}|PROT:{bait_sequence}|CCD:{ligand1}|CCD:{ligand2}
```

### 1.3 Run protein complex interaction screening 
```python
python ESMFold2_fold_complexes_from_fasta.py --input_fasta ./your/combined/input.fasta --output_dir ./your/output/dir --device cuda --num_loops 20 --num_sampling_steps 100 --num_diffusion_samples 5 --seed 0 --resume
```
