"""
This script adds the bait protein sequence to each FASTA-entry in the form "Protein:Bait-seq" and optionally removes
all entries exceeding a maximum protein complex size if the "max_len" parameter is specified.
"""

def process_fasta(input_path, output_path, bait_sequence, max_length=None):
    # Read and format the sequences
    formatted_sequences = []
    
    with open(input_path, "r") as file:
        protein_name = None
        protein_sequence = []
        
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if protein_name:  # Save the previous protein sequence if exists
                    combined_sequence = "".join(protein_sequence) + ":" + bait_sequence
                    formatted_sequences.append((protein_name, combined_sequence))
                    
                protein_name = line  # Start a new protein sequence
                protein_sequence = []
            else:
                protein_sequence.append(line)
    
        # Handle the last protein sequence
        if protein_name:
            combined_sequence = "".join(protein_sequence) + ":" + bait_sequence
            formatted_sequences.append((protein_name, combined_sequence))

    # Optionally curate the sequences based on length
    if max_length:
        formatted_sequences = [
            (name, seq) for name, seq in formatted_sequences
            if len(seq.replace(":", "")) <= max_length]

    # Write the formatted (and possibly curated) sequences to the output file
    with open(output_path, "w") as outfile:
        for name, seq in formatted_sequences:
            outfile.write(name + "\n" + seq + "\n")

    return formatted_sequences

# Usage example
bait_sequence = "" # add your bait protein sequence here
input_fasta_path = "/your/input/fasta/file/path.fasta"
output_fasta_path = "/your/output/fasta/file/path.fasta"
max_len = 1070  # Optional: set to None if you don't want to curate based on length

processed_sequences = process_fasta(input_fasta_path, output_fasta_path, bait_sequence, max_len)

# Compare the number of sequences before and after processing
with open(input_fasta_path, "r") as infile:
    original_count = sum(1 for line in infile if line.startswith(">"))

with open(output_fasta_path, "r") as outfile:
    processed_count = sum(1 for line in outfile if line.startswith(">"))

print(f"Number of sequences before processing: {original_count}")
print(f"Number of sequences after processing: {processed_count}")
print(f"Length filtering removed {original_count - processed_count} sequences exceeding the maximum length of {max_len}.")