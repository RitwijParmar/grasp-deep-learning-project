# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2





# %%
import os
import gzip
from rdkit import Chem
from tqdm import tqdm 

def extract_smiles_from_sdf_gz(input_folder, output_smiles_file):

    # Getting a sorted list of all .sdf.gz file
    sdf_gz_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.sdf.gz')])

    if not sdf_gz_files:
        print(f"Error: No .sdf.gz files found in the specified input folder: {input_folder}")
        return

    print(f"Found {len(sdf_gz_files)} .sdf.gz files to process.")
    print(f"All extracted SMILES will be aggregated into: {output_smiles_file}")

    total_processed_smiles = 0
    total_skipped_molecules = 0

    with open(output_smiles_file, 'w', encoding='utf-8') as out_f:
        for i, sdf_gz_filename in enumerate(sdf_gz_files):
            full_path_gz = os.path.join(input_folder, sdf_gz_filename)
            print(f"\n--- Processing file {i+1} of {len(sdf_gz_files)}: {sdf_gz_filename} ---")

            current_file_smiles_count = 0
            current_file_skipped_count = 0

            try:
                
                with gzip.open(full_path_gz, 'rb') as f_gz: 
                    suppl = Chem.ForwardSDMolSupplier(f_gz)

                    # Iterate through molecules in the current SDF file
                    for mol in tqdm(suppl, desc=f"Extracting from {sdf_gz_filename}"):
                        if mol is not None: # returns None for molecules rdkit cannot parse
                            try:
                                # Get canonical SMILES string for consistency
                                smiles = Chem.MolToSmiles(mol, canonical=True)
                                out_f.write(smiles + '\n')
                                current_file_smiles_count += 1
                                total_processed_smiles += 1
                            except Exception as e:
                                current_file_skipped_count += 1
                                total_skipped_molecules += 1
                        else:
                            current_file_skipped_count += 1
                            total_skipped_molecules += 1

                print(f"  Finished {sdf_gz_filename}. Extracted {current_file_smiles_count} SMILES.")
                print(f"  Skipped {current_file_skipped_count} molecules in this file.")

            except Exception as e:
                # Catch errors that prevent opening or processing an entire file
                print(f"  CRITICAL ERROR: Failed to process {sdf_gz_filename}. Reason: {e}")
                print("  Skipping this file and moving to the next. This file was NOT removed.")

    print("\n" + "="*50)
    print("                 Processing Complete!")
    print("="*50)
    print(f"Total SMILES successfully extracted: {total_processed_smiles}")
    print(f"Total molecules skipped (due to parsing or SMILES conversion errors): {total_skipped_molecules}")
    print(f"All SMILES saved to: {output_smiles_file}")
    print("\nNext steps: You can now use this 'pubchem_smiles_for_pretraining.txt' file for your model's pre-training phase.")

# path
input_directory = 'PubChem_compound' 
output_smiles_file_path = 'pubchem_smiles_for_pretraining.txt'

extract_smiles_from_sdf_gz(input_directory, output_smiles_file_path)
