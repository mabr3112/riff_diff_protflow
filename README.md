# riff_diff_protflow
ProtFlow Implementation of the RiffDiff pipeline to design enzymes from theozymes.

# INSTALLATION

The ProtFlow python library is required to run Riff-Diff. You can download it here: https://github.com/mabr3112/ProtFlow together with installation instructions. Required protein design tools for Riff-Diff that need to be installed on your machine:
  - RFdiffusion (https://github.com/RosettaCommons/RFdiffusion, tested for v1.1.0 RFdiffusion Python Module)
  - ESMFold (https://github.com/facebookresearch/esm, tested for ESM-2 Public Release v1.0.3) (If you have troubles installing ESMFold, maybe this helps: https://github.com/mabr3112/One-command-install-ESMfold)   
  - Rosetta (https://github.com/RosettaCommons/rosetta, tested for Rosetta 3.13)
  - LigandMPNN (https://github.com/dauparas/LigandMPNN)
  - ColabFold (https://github.com/YoshitakaMo/localcolabfold, tested for v1.5.5)
  - AttnPacker (optional, https://github.com/MattMcPartlon/AttnPacker)

Make sure all of these are properly configured with ProtFlow (added to the .config file!)
The Riff-Diff pipeline was developed for SLURM-based job allocators.

All Riff-Diff scripts should be started within an active ProtFlow conda environment.

Clone this repository:

```
git clone https://github.com/mabr3112/riff_diff_protflow.git
```

Replace the potentials.py in your RFdiffusion installation folder with the potentials.py in the database directory to gain access to the custom pocket potential. Alternatively, you can just copy the relevant class (custom_recenter_ROG) into the RFdiffusion potentials.py script (do not forget to add the potentials to the dictionary of implemented potentials in the end of the script!).

```
cd riff_diff_protflow
cp database/potentials.py /path/to/RFdiffusion/rfdiffusion/potentials/potentials.py
```

If you want to experiment with the fragment picking pipeline, you need to download the fragment database from https://zenodo.org/records/15482348 and save it at /riff_diff_protflow/database/fraglib_noscore.pkl.


# MANUAL

The riff-diff pipeline is divided into 2 major scripts:
  1) generation of fragments for a provided theozyme to create a motif library (create_fragment_library.py)
  2) generation of structures and refinement (structure_generation.py)
This manual will walk you through each of these steps to create proficient de novo enzymes or small molecule binders.

# MOTIF LIBRARY CREATION

For each of the active site residues in the theozyme, fragments will be created by inverting the rotamers and attaching a backbone fragment. The fragments are mainly selected based on rotamer preference.
A complete output for the generation of the motif library using the example input can be downloaded from https://doi.org/10.5281/zenodo.15588352. 

An example command to run create_fragment_library.py can be found in the examples folder. Make sure to run this command having the python environment of riff_diff activated that has protflow installed. This script uses either cmd-line arguments or a json file as input. An example json file can be found at riff_diff_protflow/examples/inputs/in.json. You can run the script using motif_library_generation.sh.

```
cd examples
sbatch motif_library_generation.sh
```

By default, the backbone fragment consists of a 7-residue idealized helical fragment, but this can be replaced by custom fragments if desired using the --fragment_pdb flag. The mandatory inputs are:
  - --riff_diff_dir: required to find paths to database etc
  - --theozyme_pdb: path to the theozyme input
  - --theozyme_resnums: list of all active site residues with chain information that should be present in the catalytic motif
  - --working_dir: directory where all output will be saved
  - --ligands: list of all ligand residues with chain information that should be present in the catalytic motif
You can specify these commands on the cmd-line, but using an input json offers more flexibility as each option can be set for a specific residue (see the example file in.json). Important options are:
  - --add_equivalent_func_groups: if one of the catalytic residues is ASP/ASN/ILE, also create fragments with GLU/GLN/VAL
  - --rotamer_position: select position of the active site residue on the backbbone fragment (by default, all positions except for N- and C-terminus of the fragment are selected)
  - --custom_channel_path: replaces the default channel with the channel from this pdb
  - --channel_chain: in which chain the channel in the pdb at channel_path is found
  - --preserve_channel_coordinates: preserves the coordinates of the custom channel instead of automatically placing it

If fragment generation fails, this is often due to clashes being detected. You can decrease the Van-der-Waals-multiplier to make clash detection less strict by setting the option --rot_lig_clash_vdw_multiplier to a lower value.
Instead of providing a predefined backbone fragment, Riff-Diff can also search the PDB for the most appropriate fragments based on rotamer and backbone occurrences using the flag --pick_frags_from_db. You need to download the fragment library from https://doi.org/10.5281/zenodo.15482348 first and place it in the Riff-Diff database folder. This will result in fragments with higher rotamer probabilities, however these fragments typically yield worse output in the subsequent diffusion steps compared to the default idealized helix fragment.

After running the script, two folders have been created in the output directory: fragments and motif_library_assembly. The folder fragments contains one multimodel PDB for each active site residue and additional information on selected fragments and rotamers. The folder motif_library_assembly contains the top motifs that passed clash detection and a folder called ligands. In this folder, the Rosetta .params files can be found for each selected ligand. These files can be manually modified, if needed. In the working directory, a a file called selected_paths.json contains information on all selected motifs. This file is the input for the next step, structure generation.


# STRUCTURE GENERATION

Structure generation proceeds in several stages, all bundled within the structure_generation.py script. Specifying a working directory (--working_dir) is mandatory for all. A complete output using the example input can be downloaded from https://doi.org/10.5281/zenodo.15588352. The example script can be started like this:

```
cd examples
sbatch structure_generation.sh
```

## Stage 1: Screening

During the screening stage, individual fragments within the .pdbs of the motif library will be connected by RFdiffusion. Sequences are generated using a combination of LigandMPNN and Rosetta Relax, which are then predicted via ESMFold. You need to specify an input .json file (--screen_input_json), this is usually the output of the previous step (selected_paths.json).
Different RFdiffusion settings will be sampled according to the flags --screen_decentralize_weights (influences the strength of the custom ROG potential, higher = stronger) and --screen_decentralize_distances (places the center of the denoising trajectory along an axis from the center of the motif in the direction of the channel). Increasing the decentralize weights leads to more globular, compact proteins, but increasing it too far results in implausible backbones. If your screening output structures have very exposed active sites, it is recommended to increase the decentralize distances, if the substrate is too buried, it is generally recommended to decrease them. You can also set the center of the denoising trajectory manually, by providing coordinates to the --recenter flag.
Other important flags are:
  - --screen_input_poses: determines how many motifs of the motif library are selected as input for rfdiffusion.
  - --screen_input_selection: determines how structures are selected, it is generally recommended to use either "top" or "weighted".
  - --screen_num_rfdiffusions: number of backbones that will be generated with RFdiffusion for each selected motif.

The combined output of all screening runs is found in working_dir/screening_results. You can have a closer look in pymol by running these commands:
```
cd /path/to/working_dir/screening_results
pymol align_results.pml
```
You will also find some plots containing data on the individual screening runs and a file called successful_input_motifs.json. This file contains all input motifs that yielded successful motifs, you can use it as input for subsequent screening runs (e.g. using a different prefix with the flag --screen_prefix).

It is generally recommended to have a look at the screening output before continuing to the next stage using the flag --skip_refinement. This way the script will not continue automatically.

## Stage 2: Refinement

In this stage, the screening output is iteratively refined by cycling sequence optimization-backbone optimization-sequence-optimization-structure prediction steps (LigandMPNN-Rosetta Relax-LigandMPNN-ESMFold). If you did not provide the --skip_refinement flag in the previous stage, it will start automatically. Otherwise, you can provide the output of the previous step (/path/to/working_dir/screening_results_all.json) via the flag --ref_input_json.
  - --ref_cycles: number of iterations for sequence optimization-backbone optimization-sequence-optimization-structure prediction.
  - --ref_input_poses: recommended to look at the screening output to determine how many structures are actually worth refining and select the number of input poses based on that.
The various filter steps can be adjusted via --ref_catres_bb_rmsd_cutoff_start/--ref_catres_bb_rmsd_cutoff_end, --ref_plddt_cutoff_start/--ref_plddt_cutoff_end. Filter values are ramped with each cycle, start and end values correspond to the first and last cycle.
You can again pause the script using --skip_evaluation. The refinement output is found in the refinement_results folder and can again be viewed in pymol by running the script align_results.pml. This folder also contains various plots for design parameters over the individual refinement cycles.

## Stage 3: Evaluation

The refined structures are predicted using AF2 and evaluated by various metrics. If --skip_evaluation was set in the previous stage, you can provide the refinement output with --eval_input_json.
  - --eval_input_poses: define based on number of promising refinement structures.
The output is in the folder evaluation_results and can be viewed via algin_results.pml.

## Stage 4: Diversification
This is an optional stage. It is used to diversify sequences for successful backbones and to fine-tune active sites. You can manually provide a list of mutations you want to introduce for each input structure (for instance, to open channels that are blocked by a sidechain). This can be done using the mutations_blank.csv in the evaluation_results folder. In the column omit_AAs, residue positions that should not have the selected amino acid can be provided. In the column allow_AAs, all allowed amino acids at the specified positions can be provided. e.g.: A37:R;A118:YWF in the omit_AAs column will prevent an arginine residue at position 37 and a tyrosine, tryptophane or phenylalanine at position 118. A12:AGST in the column allow_AAs will only allow an alanine, glycine, serine or threonine at position 12. By default, LigandMPNN is used to create diversified sequences. As an alternative, coupled moves can be employed (using the flag --variants_run_cm). The output of variants generation run can be found in the variants_results folder.


# DISCLAIMER
This repository contains code from https://github.com/RosettaCommons/RFdiffusion (potentials.py) and https://github.com/RosettaCommons/rosetta (molfile_to_params.py and dependencies)
