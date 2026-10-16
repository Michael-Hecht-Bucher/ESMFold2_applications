from pathlib import Path 
import pandas as pd

"""
For a protein:protein complex with ATP:Mg2+ bound to one protein, the input should looke like this: 
>complex_1
PROT:MKT...AAA|PROT:MSD...GGK|CCD:ATP|CCD:MG

PROT: prefix to indicate a protein sequence, CCD: prefix to indicate a ligand (small organic, ion, etc.).
"""

input = Path("your/input_file.fasta")
output = Path("your/output_file.fasta")

ligand1 = "ATP"
ligand2 = "MG" # optionally set to None if only one ligand is present

with open(input, "r") as infile, open(output, "w") as outfile:
    for line in infile: 
        line = line.strip()
        if line.startswith(">"):
            outfile.write(line + "\n")
        else:
            # Split the line on the colon and append "PROT:" to both parts, then join them back together with "|"
            parts = line.split(":")
            parts = ["PROT:" + part for part in parts]
            line_final = "|".join(parts) + f"|CCD:{ligand1}"
            if ligand2 is not None:
                line_final += f"|CCD:{ligand2}"
            outfile.write(line_final + "\n")