#imports
import json
import logging
import os
import sys
import copy
import itertools
import shutil
import numpy as np
import pandas as pd

from Bio.PDB import Structure

import protflow
import protflow.config
from protflow.jobstarters import SbatchArrayJobstarter
import protflow.poses
import protflow.residues
import protflow.tools
import protflow.tools.colabfold
import protflow.tools.esmfold
import protflow.tools.ligandmpnn
import protflow.tools.attnpacker
import protflow.metrics.tmscore
import protflow.tools.protein_edits
import protflow.tools.rfdiffusion
from protflow.poses import Poses
from protflow.residues import ResidueSelection
from protflow.metrics.generic_metric_runner import GenericMetric
from protflow.metrics.ligand import LigandClashes, LigandContacts
from protflow.metrics.rmsd import BackboneRMSD, MotifRMSD, MotifSeparateSuperpositionRMSD
import protflow.tools.rosetta
from protflow.utils.biopython_tools import renumber_pdb_by_residue_mapping, load_structure_from_pdbfile, save_structure_to_pdbfile, get_sequence_from_pose
import protflow.utils.plotting as plots


def write_pymol_alignment_script(df:pd.DataFrame, scoreterm: str, top_n:int, path_to_script: str, ascending=True,
                                 ref_motif_col: str = "template_motif", target_motif_col: str = "motif_residues",
                                 ref_catres_col: str = "template_fixedres", target_catres_col: str = "fixed_residues", rlx_output: bool = False
                                 ) -> str:
    '''
    Writes .pml script for automated pymol alignment.
    '''
    cmds = []
    for index in df.sort_values(scoreterm, ascending=ascending).head(top_n).index:
        cmd = write_align_cmds(
            input_data=df.loc[index],
            ref_motif_col=ref_motif_col,
            target_motif_col=target_motif_col,
            ref_catres_col=ref_catres_col,
            target_catres_col=target_catres_col,
            rlx_output=rlx_output
        )
        cmds.append(cmd)

    with open(path_to_script, 'w', encoding="UTF-8") as f:
        f.write("\n".join(cmds))
    return path_to_script

def write_align_cmds(input_data: pd.Series, ref_motif_col: str = "template_motif", target_motif_col: str = "motif_residues", ref_catres_col: str = "template_fixedres", target_catres_col: str = "fixed_residues", rlx_output: bool = False):
    '''AAA'''
    cmds = list()

    ref_pose = input_data["input_poses"].split("/")[-1].replace(".pdb", "")
    pose = input_data["poses_description"] + ".pdb"


    # load pose and reference
    cmds.append(f"load {pose}")
    ref_pose_name = input_data['poses_description'] + "_ref"
    cmds.append(f"load {ref_pose}.pdb, {ref_pose_name}")

    # basecolor
    cmds.append(f"color violetpurple, {input_data['poses_description']}")
    cmds.append(f"color yelloworange, {ref_pose_name}")

    # select inpaint_motif residues
    cmds.append(f"select temp_motif_res, {write_pymol_motif_selection(input_data['poses_description'], input_data[target_motif_col])}")
    cmds.append(f"select temp_ref_res, {write_pymol_motif_selection(ref_pose_name, input_data[ref_motif_col])}")

    # alignimpose inpaint_motif_residues:
    #cmds.append(f"align temp_ref_res, temp_motif_res")

    # select fixed residues, show sticks and color
    cmds.append(f"select temp_cat_res, {write_pymol_motif_selection(input_data['poses_description'], input_data[target_catres_col])}")
    cmds.append(f"select temp_refcat_res, {write_pymol_motif_selection(ref_pose_name, input_data[ref_catres_col])}")
    cmds.append("show sticks, temp_cat_res")
    cmds.append("show sticks, temp_refcat_res")

    if rlx_output:
        rlx = input_data["poses_description"] + "_rlx.pdb"
        rlx_pose_name = input_data['poses_description'] + "_rlx.pdb"
        cmds.append(f"load {rlx}, {rlx_pose_name}")
        cmds.append(f"color deepsalmon, {rlx_pose_name}")
        cmds.append(f"select temp_rlxcat_res, {write_pymol_motif_selection(rlx_pose_name, input_data[target_catres_col])}")
        cmds.append("show sticks, temp_rlxcat_res")
        cmds.append("delete temp_rlxcat_res")

    cmds.append("hide sticks, hydrogens")
    cmds.append("color atomic, (not elem C)")

    # store scene, delete selection and disable object:
    #cmds.append(f"center temp_motif_res")
    cmds.append(f"scene {input_data['poses_description']}, store")
    cmds.append(f"disable {input_data['poses_description']}")
    cmds.append(f"disable {ref_pose_name}")
    if rlx_output:
        cmds.append(f"disable {rlx_pose_name}")
    cmds.append("delete temp_cat_res")
    cmds.append("delete temp_refcat_res")
    cmds.append("delete temp_motif_res")
    cmds.append("delete temp_ref_res")

    return "\n".join(cmds)

def write_pymol_motif_selection(obj: str, motif: dict) -> str:
    '''AAA'''
    if isinstance(motif, ResidueSelection):
        motif = motif.to_dict()

    residues = [f"chain {chain} and resi {'+'.join([str(x) for x in res_ids])}" for chain, res_ids in motif.items()]
    pymol_selection = ' or '.join([f"{obj} and {resis}" for resis in residues])
    return pymol_selection


def update_and_copy_reference_frags(input_df: pd.DataFrame, ref_col:str, desc_col:str, prefix: str, out_pdb_path=None, keep_ligand_chain:str="") -> "list[str]":
    '''Updates reference fragments (input_pdbs) to the motifs that were set during diffusion.'''
    # create residue mappings {old: new} for renaming
    list_of_mappings = [protflow.tools.rfdiffusion.get_residue_mapping(ref_motif, inp_motif) for ref_motif, inp_motif in zip(input_df[f"{prefix}_con_ref_pdb_idx"].to_list(), input_df[f"{prefix}_con_hal_pdb_idx"].to_list())]

    # compile list of output filenames
    output_pdb_names_list = [f"{out_pdb_path}/{desc}.pdb" for desc in input_df[desc_col].to_list()]

    # renumber
    return [renumber_pdb_by_residue_mapping(ref_frag, res_mapping, out_pdb_path=pdb_output, keep_chain=keep_ligand_chain) for ref_frag, res_mapping, pdb_output in zip(input_df[ref_col].to_list(), list_of_mappings, output_pdb_names_list)]

def instantiate_trajectory_plotting(plot_dir: str, scores: list) -> dict[plots.PlottingTrajectory]:
    # instantiate plotting trajectories:
    trajectory_dict = {}
    for score in scores:
        trajectory_dict[score] = plots.PlottingTrajectory(y_label=score, location=os.path.join(plot_dir, f"trajectory_{score}.png"), title=f"{score} Trajectory")
    return trajectory_dict

def update_trajectory_plotting(trajectory_plots:dict, poses: Poses, prefix: str):
    for traj in trajectory_plots:
        trajectory_plots[traj].set_location(os.path.join(poses.plots_dir, f"trajectory_{traj}.png"))
        trajectory_plots[traj].add_and_plot(poses.df[f"{prefix}_{traj}"], prefix)
    return trajectory_plots

def create_results_dir(poses: Poses, dir: str, score_col: str, plot_cols: list = None, rlx_path_col: str = None, create_mutations_csv: bool = False):
    os.makedirs(dir, exist_ok=True)

    logging.info("Plotting final outputs.")
    plots.violinplot_multiple_cols(
        dataframe = poses.df,
        cols = plot_cols,
        y_labels = plot_cols,
        out_path = os.path.join(dir, "scores.png"),
        show_fig = False
    )

    poses.df.sort_values(score_col, ascending=True, inplace=True)
    poses.df.reset_index(drop=True, inplace=True)
    poses.save_poses(out_path=dir)
    poses.save_poses(out_path=dir, poses_col="input_poses")
    # copy top relaxed structures
    if rlx_path_col:
        for _, row in poses.df.iterrows():
            shutil.copy(row[rlx_path_col], os.path.join(dir, f"{row['poses_description']}_rlx.pdb"))
    poses.save_scores(out_path=dir)

    # write pymol alignment script
    logging.info(f"Writing pymol alignment script for backbones after evaluation at {dir}.")
    write_pymol_alignment_script(
        df = poses.df,
        scoreterm = score_col,
        top_n = np.min([25, len(poses.df.index)]),
        path_to_script = os.path.join(dir, "align_results.pml"),
        ref_motif_col = "template_fixedres",
        ref_catres_col = "template_fixedres",
        target_catres_col = "fixed_residues",
        target_motif_col = "fixed_residues",
        rlx_output=True if rlx_path_col else False
    )

    if create_mutations_csv:
        mut_df = pd.DataFrame(poses.df["poses_description"])
        mut_df["omit_AAs"] = None
        mut_df["allow_AAs"] = None
        mut_df.to_csv(os.path.join(dir, "mutations_blank.csv"))

def write_bbopt_opts(row: pd.Series, cycle: int, total_cycles: int, reference_location_col:str, motif_res_col: str, cat_res_col: str, ligand_chain: str) -> str:
    return f"-in:file:native {row[reference_location_col]} -parser:script_vars motif_res={row[motif_res_col].to_string(ordering='rosetta')} cat_res={row[cat_res_col].to_string(ordering='rosetta')} substrate_chain={ligand_chain} sd={0.8 - (0.4 * cycle/total_cycles)}"

def calculate_mean_scores(poses: Poses, scores: list, remove_layers: int=None):
    for score in scores:
        poses.calculate_mean_score(name=f"{score}_mean", score_col=score, remove_layers=remove_layers)
    return poses

def combine_screening_results(dir: str, prefixes: list, scoreterms: list, weights: list, residue_cols: list, input_motifs: Poses, screen_prefix: str = None,):

    logging.info("Combining results from all screening runs...")

    if len(prefixes) == 0:
        logging.error("No poses passed in any of the screening runs. Aborting!"); sys.exit(1)

    screen_dir = os.path.join(dir, f"{screen_prefix}_screening" if screen_prefix else "screening")

    # set up output dir
    out_dir = os.path.join(dir, f'{screen_prefix}_screening_results' if screen_prefix else 'screening_results')
    os.makedirs(out_dir, exist_ok=True)

    # combine all screening outputs into new poses
    pose_df = []
    for prefix in prefixes:
        df = pd.read_json(os.path.join(screen_dir, prefix, f"{prefix}_scores.json"))
        df['screen_passed_poses'] = len(df.index)
        pose_df.append(df)
    pose_df = pd.concat(pose_df).reset_index(drop=True)
    poses = Poses(poses=pose_df, work_dir=screen_dir)

    # recalculate composite score over all screening runs
    poses.calculate_composite_score(
        name="screen_composite_score",
        scoreterms=scoreterms,
        weights=weights,
        plot=True
    )
    shutil.copy(os.path.join(screen_dir, "plots", "screen_composite_score_comp_score.png"), os.path.join(out_dir, "screening_results_combined.png"))

    # convert columns to residues (else, pymol script writer and refinement crash)
    for residue_col in residue_cols:
        poses.df[residue_col] = [ResidueSelection(motif, from_scorefile=True) for motif in poses.df[residue_col].to_list()]

    poses.reindex_poses(prefix="reindexed_screening_poses", remove_layers=1, force_reindex=1)

    grouped_df = poses.df.groupby('screen', sort=True)
    # plot all scores
    df_names, dfs = zip(*[(name, df) for name, df in grouped_df])
    plot_scores = scoreterms + ['screen_composite_score', 'screen_decentralize_weight', 'screen_decentralize_distance', 'screen']
    for score in plot_scores:
        plots.violinplot_multiple_cols_dfs(dfs=dfs, df_names=df_names, cols=[score], y_labels=[score], out_path=os.path.join(out_dir, f'{score}_violin.png'), show_fig=False)

    # save poses dataframe as well
    poses.df.sort_values("screen_composite_score", ascending=True, inplace=True)
    poses.df.reset_index(drop=True, inplace=True)
    poses.save_scores(out_path=os.path.join(out_dir, 'screening_results_all.json'))

    logging.info(f"Writing pymol alignment script for screening results at {out_dir}")
    write_pymol_alignment_script(
        df=poses.df,
        scoreterm = "screen_composite_score",
        top_n = np.min([len(poses.df.index), 25]),
        path_to_script = os.path.join(out_dir, "align_results.pml"),
        ref_motif_col = "template_fixedres",
        ref_catres_col = "template_fixedres",
        target_catres_col = "fixed_residues",
        target_motif_col = "fixed_residues"
    )

    poses.save_poses(out_path=out_dir)
    poses.save_poses(out_path=out_dir, poses_col="input_poses")

    # save successful input motifs
    counts = poses.df["input_poses"].value_counts()
    poses.df["successful_screening_results"] = poses.df["input_poses"].map(counts)
    unique = poses.df.drop_duplicates(subset=["input_poses"], keep="first")
    if "successful_screening_results" in input_motifs.df.columns:
        input_motifs.df.drop("successful_screening_results", inplace=True, axis=1)
    successful_motifs = input_motifs.df.merge(unique[['input_poses', 'successful_screening_results']], on="input_poses")
    successful_motifs.sort_values("successful_screening_results", ascending=False, inplace=True)
    successful_motifs.reset_index(drop=True, inplace=True)
    successful_motifs.to_json(os.path.join(out_dir, "successful_input_motifs.json"))

    return poses

def ramp_cutoff(start_value, end_value, cycle, total_cycles):
    if total_cycles == 1:
        return end_value
    step = (end_value - start_value) / (total_cycles - 1)
    return start_value + (cycle - 1) * step


def split_combined_ligand(poses_dir:str, ligand_paths: list):
    logging.info('Replacing RFdiffusion combined ligand with separated ligands...')
    new_ligands = [[res for res in load_structure_from_pdbfile(new_lig).get_residues()][0] for new_lig in ligand_paths]
    poses = Poses(poses=poses_dir, glob_suffix="*.pdb")
    for pose_path in poses.df['poses'].to_list():
        pose = load_structure_from_pdbfile(pose_path)
        original_ligands = [lig for lig in pose["Z"].get_residues()]
        for old_lig in original_ligands:
            pose["Z"].detach_child(old_lig.id)
        for new_lig in new_ligands:
            pose["Z"].add(new_lig)
        save_structure_to_pdbfile(pose, pose_path)

def create_reduced_motif(fixed_res:ResidueSelection, motif_res:ResidueSelection):
    reduced_dict = {}
    fixed_res = fixed_res.to_dict()
    motif_res = motif_res.to_dict()
    for chain in fixed_res:
        res = []
        reduced_motif = []
        for residue in fixed_res[chain]:
            res.append(residue -1)
            res.append(residue)
            res.append(residue + 1)
        for i in res:
            if i in motif_res[chain]:
                reduced_motif.append(i)
        reduced_dict[chain] = reduced_motif
    return protflow.residues.from_dict(reduced_dict)

def aa_one_letter_code() -> str:
    return "ARDNCEQGHILKMFPSTWYV"

def omit_AAs(omitted_aas:str, allowed_aas:str, dir:str, name:str) -> str:
    mutations_dict = {}
    if not allowed_aas and not omitted_aas:
        return None
    if isinstance(omitted_aas, str):
        omitted_aas = omitted_aas.rstrip(";").split(";")
        for mutation in omitted_aas:
            position, omitted = mutation.split(":")
            if not position[0].isalpha():
                raise KeyError(f"Position for mutations have to include chain information (e.g. A{position}), not just {position}!")
            mutations_dict[position.strip()] = omitted.strip()
    if isinstance(allowed_aas, str):
        allowed_aas = allowed_aas.rstrip(";").split(";")
        for mutation in allowed_aas:
            position, allowed = mutation.split(":")
            if not position[0].isalpha():
                raise KeyError(f"Position for mutations have to include chain information (e.g. A{position}), not just {position}!")
            all_aas = aa_one_letter_code()
            for aa in allowed:
                all_aas = all_aas.replace(aa.upper(), "")
            mutations_dict[position.strip()] = all_aas.upper().strip()
    filename = os.path.join(dir, f"{name}_mutations.json")
    with open(filename, "w") as out:
        json.dump(mutations_dict, out)
    return f"--omit_AA_per_residue {filename}"



def log_cmd(arguments):
    cmd = ''
    for key, value in vars(arguments).items():
        cmd += f'--{key} {value} '
    cmd = f'{sys.argv[0]} {cmd}'
    logging.info(f"{sys.argv[0]} {cmd}")

def determine_last_ref_cycle(df: pd.DataFrame) -> int:
    cols = [col for col in df.columns if col.startswith("cycle_")]
    nums = [col.split("_")[1] for col in cols]
    last_cycle = "final" if "final" in nums else max([int(num) for num in nums])
    return last_cycle

def write_rfdiffusion_contig(frag_contigs:str, total_length:int, flanker_length:int, frag_sizes:str, flanker_type:str, channel_contig:str=None, sep=","):
    frag_contigs = [frag for frag in frag_contigs.split(sep)]
    frag_total_length = sum([int(size) for size in frag_sizes.split(sep)])
    num_linkers = len(frag_contigs) - 1
    linker_length = 5 + int((total_length - flanker_length - frag_total_length) / num_linkers)
    frag_contigs = f"/10-{linker_length}/".join(frag_contigs)
    if flanker_type == "split":
        contig = f"{flanker_length}/{frag_contigs}/{flanker_length}"
    elif flanker_type == "cterm":
        contig = f"{frag_contigs}/{flanker_length}"
    elif flanker_type == "nterm":
        contig = f"{flanker_length}/{frag_contigs}"
    else:
        logging.error(f"<flanker_type> must be one of 'split', 'cterm' or 'nterm', not {flanker_type}!")
        raise KeyError(f"<flanker_type> must be one of 'split', 'cterm' or 'nterm', not {flanker_type}!")
    if channel_contig:
        contig = f"{channel_contig}/0 {contig}"
    return f"'contigmap.contigs=[{contig}]' "

def write_rfdiffusion_inpaint_seq(motif_residues:ResidueSelection, fixed_residues:ResidueSelection):
    inpaint_seq = motif_residues - fixed_residues
    inpaint_seq = inpaint_seq.to_string(delim="/")
    return f"'contigmap.inpaint_seq=[{inpaint_seq}]'"

def add_covalent_bonds_info(poses:Poses, prefix:str, covalent_bonds_col:str) -> Poses:
    if poses.df[covalent_bonds_col].isna().all():
        return poses
    covalent_bonds = poses.df[covalent_bonds_col].to_list()
    os.makedirs(out_dir := os.path.join(poses.work_dir, prefix), exist_ok=True)
    paths = []
    for pose, cov_bonds in zip(poses.poses_list(), covalent_bonds):
        links = "\n".join([parse_link_from_cov_bond(cov_bond) for cov_bond in cov_bonds.split(",")]) + "\n"
        if not os.path.isfile(outfile := os.path.join(out_dir, os.path.basename(pose))):
            prefix_string_to_file(pose, links, outfile)
        paths.append(outfile)
    poses.df["poses"] = paths
    return poses

def prefix_string_to_file(file_path: str, prefix: str, save_path:str=None):
    '''Adds something to the beginning of a file.'''
    with open(file_path, 'r') as f: file_str = f.read()
    with open(save_path or file_path, 'w') as f: f.write(prefix + file_str)

def parse_link_from_cov_bond(covalent_bond: str) -> str:
    '''parses covalent bond into Rosetta formated link record for PDB headers.'''
    res_data, lig_data = covalent_bond.split(':')
    res_data, res_atom = res_data.split('-')
    lig_data, lig_atom = lig_data.split('-')
    res_data, res_id = res_data.split("_")
    lig_data, lig_id = lig_data.split("_")
    res_chain = res_data[-1]
    res_num = res_data[:-1]
    lig_chain = lig_data[-1]
    lig_num = lig_data[:-1]
    return f"LINK         {res_atom:<3} {res_id:<3} {res_chain:>1}{res_num:>4}                {lig_atom:<3}  {lig_id:>3} {lig_chain:>1}{lig_num:>4}                  0.00"

def update_covalent_bonds_info(bonds:str, original_fixedres:ResidueSelection, updated_fixedres:ResidueSelection) -> str:
    # NaN and None checker for covalent bonds
    if not bonds or bonds != bonds:
        return None
    original = original_fixedres.to_list(ordering="rosetta")
    new = updated_fixedres.to_list(ordering="rosetta")
    new_bonds = []
    for bond in bonds.split(","):
        resnum = bond.split("_")[0]
        idx = original.index(resnum)
        new_resnum = new[idx]
        new_bonds.append("_".join([new_resnum] + bond.split("_")[1:]))
    return ",".join(new_bonds)

def calculate_contact_score(df: pd.DataFrame, contact_col: str, score_col: str, target_value: float) -> pd.DataFrame:
    df[score_col] = abs(df[contact_col] - target_value)
    return df

def pool_all_cycle_results(ref_dict: dict, plddt_cutoff:float, catres_bb_rmsd_cutoff:float, motif_bb_rmsd_cutoff:float, comp_score_scoreterms:list, comp_score_weights:list, refinement_dir:str) -> Poses:
    poses_list = []
    for cycle in ref_dict:
        backbones = ref_dict[cycle]
        # save all cycle results as cycle_final
        final_columns = {
            f"cycle_final_{'_'.join(column.split('_')[2:])}": backbones.df[column]
            for column in backbones.df.columns if column.startswith(f"cycle_{cycle}")}
        backbones.df = pd.concat([backbones.df, pd.DataFrame(final_columns)], axis=1)

        # apply hard filters from last refinement cycle
        backbones.filter_poses_by_value(score_col="cycle_final_esm_plddt", value=plddt_cutoff, operator=">=", fail_on_empty=False)
        backbones.filter_poses_by_value(score_col="cycle_final_esm_catres_bb_rmsd", value=catres_bb_rmsd_cutoff, operator="<=", fail_on_empty=False)
        backbones.filter_poses_by_value(score_col="cycle_final_esm_motif_rmsd", value=motif_bb_rmsd_cutoff, operator="<=", fail_on_empty=False)
        if not backbones.df.empty:
            poses_list.append(backbones.df.copy())

    # combine all results that passed filters
    backbones.df = pd.concat(poses_list)
    backbones.df.reset_index(inplace=True, drop=True)
    backbones.set_work_dir(refinement_dir)

    final_scores = [f"cycle_final_{'_'.join(score.split('_')[2:])}" for score in comp_score_scoreterms]

    # calculate new composite score
    backbones.calculate_composite_score(
        name="cycle_final_refinement_composite_score",
        scoreterms=final_scores,
        weights=comp_score_weights)

    # sort
    backbones.df.sort_values("cycle_final_refinement_composite_score", ascending=True, inplace=True)

    # reindex to make sure no poses with identical names exist
    backbones.reindex_poses(prefix="pool_reindex", remove_layers=1, force_reindex=True)

    return backbones

def create_mutation_resfiles(omitted_aas:str, allowed_aas:str, name:str, dir:str):
    os.makedirs(dir, exist_ok=True)

    omitted = []
    allowed = []

    if isinstance(omitted_aas, str):
        omitted_aas = omitted_aas.rstrip(";").split(";")
        for mutation in omitted_aas:
            position, omitted_aas = mutation.split(":")
            if not position[0].isalpha():
                raise KeyError(f"Position for mutations have to include chain information (e.g. A{position}), not just {position}!")
            omitted.append(f"{position[1:]} {position[0]} NOTAA {omitted_aas}")

    if isinstance(allowed_aas, str):
        allowed = []
        allowed_aas = allowed_aas.rstrip(";").split(";")
        for mutation in allowed_aas:
            position, allowed_aas = mutation.split(":")
            if not position[0].isalpha():
                raise KeyError(f"Position for mutations have to include chain information (e.g. A{position}), not just {position}!")
            allowed.append(f"{position[1:]} {position[0]} PIKAA {allowed_aas}")
    mutations = omitted + allowed
    mutations = sorted(mutations, key=lambda x: int(x.split()[0]))
    filename = os.path.join(dir, f"{name}.resfile")
    resfile = "start\n" + "\n".join(mutations)
    with open(filename, "w") as out:
        out.write(resfile)
    return filename

def write_cm_opts(fixed_residues, motif_residues, native_path, design_shell, resfile: str = None):
    design_shell = design_shell.split(",")
    cm_script_vars = f"-parser:script_vars resfilepath={resfile} cat_res={fixed_residues.to_string(ordering='rosetta')} motif_res={motif_residues.to_string(ordering='rosetta')} cut1={design_shell[0]} cut2={design_shell[1]} cut3={design_shell[2]} cut4={design_shell[3]} -in:file:native {native_path}"
    return cm_script_vars


def extract_designpositions_from_resfile(resfile):
    design_positions = []
    with open(resfile, "r") as r:
        for line in r:
            if not line.startswith("NAT") and not line.startswith("start"):
                design_positions.append(int(line.split()[0]))
    #filter for unique elements in list
    #design_positions = sorted(list(set(design_positions)))
    return(design_positions)

def statsfile_to_df(statsfile: str):
    #reads in the .stats file output from a coupled-moves run and converts it to a dataframe
    df = pd.read_csv(statsfile, sep=None, engine='python', header=None, keep_default_na=False)
    df_scores = df[4].str.split(expand=True)
    columnheaders = df_scores[df_scores.columns[0::2]]
    columnheaders = columnheaders.loc[0, :].values.tolist()
    columnheaders = [i.replace(':','') for i in columnheaders]
    df_scores = df_scores[df_scores.columns[1::2]]
    df_scores.columns = columnheaders
    df_scores = df_scores.astype(float)
    df_scores["total_score"] = df_scores.sum(axis=1)
    df_scores["sequence"] = df[3]
    return(df_scores)

def statsfiles_to_json(input_dir: str, description:str, filename):

    if os.path.isfile(filename):
        with open(filename) as json_file:
            structdict = json.load(json_file)
            return(structdict)

    #gathers all coupled-moves statsfiles and converts to a single dictionary
    statsfiles = []
    resfiles = []
    for file in os.listdir(input_dir):
        # 6:-11 ignores protflow rosetta r0001_ prefixes and 0001.stats suffixes
        if file.endswith(".stats") and file[6:-11] == description:
            statsfiles.append(os.path.join(input_dir, file))
        elif file.endswith(".resfile") and file[6:-13] == description:
            resfiles.append(os.path.join(input_dir, file))

    statsfiles = sorted(statsfiles)
    resfiles = sorted(resfiles)
    stats_df_list = []
    for stats, res in zip(statsfiles, resfiles):
        statsdf = statsfile_to_df(stats)
        design_positions = extract_designpositions_from_resfile(res)
        design_positions = [design_positions for i in range(0, len(statsdf))]
        statsdf['design_positions'] = design_positions
        for index, row in statsdf.iterrows():
            for mut, pos in zip(row['sequence'], row['design_positions']):
                mut_row = row.copy()
                mut_row['mutation'] = mut
                mut_row['position'] = pos
                stats_df_list.append(mut_row)

    statsdf = pd.DataFrame(stats_df_list)
    #statsdf['total_score'] = statsdf['total_score'] - statsdf['res_type_constraint']

    structdict = {}
    for pos, df in statsdf.groupby('position'):
        posdict = {}
        for AA, pos_df in df.groupby('mutation'):
            posdict[AA] = {"pos": pos, "identity": AA, "count": len(pos_df), "ratio": len(pos_df)/len(df), "total_score": [round(score, 2) for score in pos_df['total_score'].to_list()], "total_score_average": round(pos_df['total_score'].mean(), 2), "coordinate_constraint": [round(score, 2) for score in pos_df['coordinate_constraint'].to_list()], "coordinate_constraint_average": round(pos_df['coordinate_constraint'].mean(), 2)}
        structdict[pos] = posdict

    with open(filename, "w") as outfile:
        json.dump(structdict, outfile)
    return(structdict)

def generate_mutations_dict(datadict, occurence_cutoff):
    '''
    only accepts mutations that show up in at least <occurence_cutoff> of coupled moves runs. if no mutation is above cutoff, picks the lowest energy one.
    '''
    mutations = {}
    for pos in datadict:
        df = pd.DataFrame(datadict[pos]).transpose().sort_values('ratio')
        df_filtered = df[df['ratio'] >= occurence_cutoff]
        if df_filtered.empty:
            df_filtered = df[df['ratio'] >= 0.1]
            if df_filtered.empty:
                df_filtered = df
            df_filtered = df_filtered.sort_values('coordinate_constraint_average', ascending=True).head(int(1 + len(df_filtered) / 2))
            df_filtered = df_filtered.sort_values('total_score_average', ascending=True).head(1)
        mutations[pos] = df_filtered['identity'].to_list()
    return mutations

def generate_variants(mutation_dict, seq):
    mutlist = []
    poslist = []
    for pos in mutation_dict:
        poslist.append(pos)
        mutlist.append(mutation_dict[pos])
    combs = list(itertools.product(*mutlist))
    variants = []
    for comb in combs:
        var = copy.deepcopy(seq)
        for index, AA in enumerate(comb):
            var[int(poslist[index]) - 1] = AA
        variants.append(''.join(var))
    return variants


def create_coupled_moves_sequences(output_dir, cm_resultsdir, poses_df, occurence_cutoff):

    os.makedirs(output_dir, exist_ok=True)
    out_df = []
    if not os.path.isfile(scorefile := os.path.join(output_dir, "cm_results.json")):
        for _, row in poses_df.iterrows():
            statsdict = statsfiles_to_json(cm_resultsdir, row['poses_description'], os.path.join(cm_resultsdir, f"{row['poses_description']}.json"))
            mutations = generate_mutations_dict(statsdict, occurence_cutoff)
            seq = list(get_sequence_from_pose(load_structure_from_pdbfile(row["poses"]))) # has to be loaded directly, otherwise running cm on cm output again will not work
            #seq = list(load_sequence_from_fasta(row["eval_fasta_conversion_fasta_location"]).seq)
            variants_df = pd.DataFrame(generate_variants(mutations, seq), columns=[f'cm_sequences'])
            variants_df["poses_description"] = row["poses_description"]
            variants_df = variants_df.merge(poses_df, on="poses_description", how="left")
            logging.info(f"Generated {len(variants_df.index)} variants for pose {row['poses_description']}.")
            for seqnum, var in variants_df.iterrows():
                var['poses_description'] = f"{row['poses_description']}_{seqnum+1:04d}"
                with open(pose_path := os.path.join(output_dir, f"{var['poses_description']}.fa"), "w") as fasta:
                    fasta.write(f">{var['poses_description']}\n{var['cm_sequences']}")
                var['poses'] = pose_path
                out_df.append(var)

        out_df = pd.DataFrame(out_df).reset_index(drop=True)
        out_df.to_json(scorefile)

    else:
        out_df = pd.read_json(scorefile)
    return out_df

def weighted_average(df, group_col, score_col, weight_col, inverse: bool = False, weight_exponent: float = 1):
    """
    Calculate the weighted average of a score column within groups,
    using a specified column where the weight is determined based on its squared values.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    group_col (str): Column name to group by.
    score_col (str): Column to calculate the weighted average for.
    weight_col (str): Column used for weight calculation.
    inverse (bool): If True, lower values in weight_col result in higher weights.
                    If False, higher values in weight_col result in higher weights.
    weight_exponent (float): Exponent for weight calculation.

    Returns:
    pd.DataFrame: A DataFrame with the group column and weighted averages.
    """
    # Compute weight based on the square of the weight_col values
    if inverse:
        df['weight'] = 1 / (df[weight_col] ** weight_exponent)  # Lower values → Higher weights
    else:
        df['weight'] = df[weight_col] ** weight_exponent  # Higher values → Higher weights

    # Calculate weighted average per group
    weighted_avg = df.groupby(group_col).apply(lambda g: (g[score_col] * g['weight']).sum() / g['weight'].sum())

    # Rename and reset index for merging
    return weighted_avg.rename(f'{score_col}_weighted').reset_index()

def top_fraction_avg(df: pd.DataFrame, group_col: str, value_col: str, fraction: float = 0.1, inverse: bool = False):
    """
    Groups a DataFrame by `group_col` and calculates the average of the top or
    bottom fraction of `value_col` within each group.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    group_col (str): The column to group by.
    value_col (str): The column to calculate the fraction-based average from.
    fraction (float): The fraction of values to consider (default is 10%).
    mode (str): 'top' for highest values, 'bottom' for lowest values.

    Returns:
    pd.DataFrame: A DataFrame with the group column and the calculated average.
    """
    def fraction_avg(group):
        n = max(1, int(len(group) * fraction))  # Ensure at least one row
        if inverse:
            return group.nsmallest(n, value_col)[value_col].mean()
        else:
            return group.nlargest(n, value_col)[value_col].mean()

    return df.groupby(group_col).apply(fraction_avg).reset_index(name=f'{value_col}_top_{fraction}_avg')



def sort_and_combine_relaxed_df(df: pd.DataFrame, group_name, out_dir, path_col, score_col, fraction, ascending):
    df_sorted = df.sort_values(score_col, ascending=ascending)
    df_top = df_sorted.head(max(1, round(len(df_sorted) * fraction)))

    out_structure = Structure.Structure("combined")
    for i, path in enumerate(df_top[path_col].to_list()):
        model = load_structure_from_pdbfile(path)
        model.id = i
        out_structure.add(model)

    out_path = os.path.join(out_dir, f"{group_name}_combined.pdb")
    save_structure_to_pdbfile(out_structure, out_path)

    return os.path.abspath(out_path)


def combine_relax_output(df: pd.DataFrame, path_col: str, out_dir: str, out_col: str, group_col: str, score_col:str, fraction: float = 0.1, ascending: bool = True) -> pd.DataFrame:
    """
    combines top <fraction> of models according to <score_col> into a single multimodel PDB.
    """
    os.makedirs(out_dir, exist_ok=True)
    # group dataframe
    result = (
        df.groupby(group_col)
              .apply(lambda g: pd.Series({group_col: g.name, out_col: sort_and_combine_relaxed_df(g, g.name, out_dir, path_col, score_col, fraction, ascending)}))
              .reset_index(drop=True))
    return result


def main(args):
    '''executes everyting (duh)'''
    ################################################# SET UP #########################################################
    # logging and checking of inputs
    if not os.path.isdir(args.working_dir):
        os.makedirs(args.working_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(args.working_dir, "riffdiff.log"),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    log_cmd(args)

    if not sum(bool(v) for v in [args.screen_input_json, args.ref_input_json, args.eval_input_json, args.variants_input_json]) == 1:
        raise ValueError("One (but not more) of :screen_input_json:, :ref_input_json:, :eval_input_json: and :variants_input_json: must be set!")

    if args.screen_input_json:
        backbones = protflow.poses.load_poses(args.screen_input_json)
        backbones.df["template_motif"] = backbones.df["motif_residues"]
        backbones.df["template_fixedres"] = backbones.df["fixed_residues"]
    elif args.ref_input_json:
        backbones = protflow.poses.load_poses(args.ref_input_json)
    elif args.eval_input_json:
        backbones = protflow.poses.load_poses(args.eval_input_json)
    elif args.variants_input_json:
        backbones = protflow.poses.load_poses(args.variants_input_json)

    # create residue selections
    residue_cols = ["fixed_residues", "motif_residues", "template_motif", "template_fixedres", "ligand_motif"]
    for res_col in residue_cols:
        backbones.df[res_col] = [ResidueSelection(motif, from_scorefile=True) for motif in backbones.df[res_col].to_list()]

    # check params files
    if args.params_files:
        params = args.params.split(",")
    elif "params_path" in backbones.df.columns:
        params = backbones.df["params_path"].unique()
        if len(params) > 1:
            logging.error("Poses with different params files found! Did you mix up input poses?")
            raise RuntimeError("Poses with different params files found! Did you mix up input poses?")
        params = params[0].split(",")
    else:
        params = None

    # check ligands
    if "ligand_path" in backbones.df.columns:
        ligand_paths = backbones.df["ligand_path"].unique()
        if len(ligand_paths) > 1:
            logging.error("Poses with different ligand paths found! Did you mix up input poses?")
            raise RuntimeError("Poses with different ligand paths found! Did you mix up input poses?")
        ligand_paths = ligand_paths[0].split(",")
    else:
        ligand_paths = None

    # setup jobstarters
    cpu_jobstarter = SbatchArrayJobstarter(max_cores=args.max_cpus, batch_cmds=args.max_cpus)
    small_cpu_jobstarter = SbatchArrayJobstarter(max_cores=10, batch_cmds=10)
    gpu_jobstarter = cpu_jobstarter if args.prefer_cpu else SbatchArrayJobstarter(max_cores=args.max_gpus, gpus=1, batch_cmds=args.max_gpus)
    real_gpu_jobstarter = SbatchArrayJobstarter(max_cores=args.max_gpus, gpus=1, batch_cmds=args.max_gpus) # esmfold does not work on cpu

    # set up runners
    logging.info("Settung up runners.")
    rfdiffusion = protflow.tools.rfdiffusion.RFdiffusion(jobstarter = gpu_jobstarter)
    chain_adder = protflow.tools.protein_edits.ChainAdder(jobstarter = small_cpu_jobstarter)
    chain_remover = protflow.tools.protein_edits.ChainRemover(jobstarter = small_cpu_jobstarter)
    bb_rmsd = BackboneRMSD(chains="A", jobstarter = small_cpu_jobstarter)
    fragment_motif_bb_rmsd = MotifRMSD(ref_col = "updated_reference_frags_location", target_motif = "motif_residues", ref_motif = "motif_residues", atoms=["N", "CA", "C"], jobstarter=small_cpu_jobstarter)
    catres_motif_bb_rmsd = MotifRMSD(ref_col = "updated_reference_frags_location", target_motif = "fixed_residues", ref_motif = "fixed_residues", atoms=["N", "CA", "C"], jobstarter=small_cpu_jobstarter)
    catres_motif_heavy_rmsd = MotifRMSD(ref_col = "updated_reference_frags_location", target_motif = "fixed_residues", ref_motif = "fixed_residues", jobstarter=small_cpu_jobstarter)
    ligand_clash = LigandClashes(ligand_chain="Z", factor=args.ligand_clash_factor, atoms=['N', 'CA', 'C', 'O'], jobstarter=small_cpu_jobstarter)
    ligand_contacts = LigandContacts(ligand_chain="Z", min_dist=0, max_dist=8, atoms=['CA'], jobstarter=small_cpu_jobstarter)
    rog_calculator = GenericMetric(module="protflow.utils.metrics", function="calc_rog_of_pdb", jobstarter=small_cpu_jobstarter)
    tm_score_calculator = protflow.metrics.tmscore.TMalign(jobstarter = small_cpu_jobstarter)
    ligand_mpnn = protflow.tools.ligandmpnn.LigandMPNN(jobstarter = cpu_jobstarter)
    rosetta = protflow.tools.rosetta.Rosetta(jobstarter = cpu_jobstarter, fail_on_missing_output_poses=True)
    esmfold = protflow.tools.esmfold.ESMFold(jobstarter = real_gpu_jobstarter) # esmfold does not work on cpu
    ligand_rmsd = MotifSeparateSuperpositionRMSD(
        ref_col="updated_reference_frags_location",
        super_target_motif="fixed_residues",
        super_ref_motif="fixed_residues",
        super_atoms=['N', 'CA', 'C'],
        rmsd_target_motif="ligand_motif",
        rmsd_ref_motif="ligand_motif",
        rmsd_atoms=None,
        rmsd_include_het_atoms=True,
        jobstarter = small_cpu_jobstarter)
    colabfold = protflow.tools.colabfold.Colabfold(jobstarter=real_gpu_jobstarter)
    if args.attnpacker_repack:
        attnpacker = protflow.tools.attnpacker.AttnPacker(jobstarter=gpu_jobstarter)


    # set up general ligandmpnn options
    ligandmpnn_options = f"--ligand_mpnn_use_side_chain_context 1 {args.ligandmpnn_options if args.ligandmpnn_options else ''}"

    # set up general rosetta options
    bb_opt_options = f"-parser:protocol {os.path.abspath(os.path.join(args.riff_diff_dir, 'utils', 'fr_constrained.xml'))} -beta -ignore_zero_occupancy false -flip_HNQ true -use_input_sc true -ignore_waters false"
    fr_options = f"-parser:protocol {os.path.abspath(os.path.join(protflow.config.AUXILIARY_RUNNER_SCRIPTS_DIR, 'fastrelax_sap.xml'))} -beta -ignore_zero_occupancy false -flip_HNQ true -use_input_sc true -ignore_waters false"

    if params:
        fr_options = fr_options + f" -extra_res_fa {' '.join(params)}"
        bb_opt_options = bb_opt_options + f" -extra_res_fa {' '.join(params)}"

    trajectory_scoreterms = ["plddt", "catres_bb_rmsd", "catres_heavy_rmsd", "motif_rmsd", "contacts_score", "ligand_clashes", "tm_TM_score_ref"]
    ############################################## SCREENING ######################################################


    if args.screen_input_json:
        screen_prefix = f"{args.screen_prefix}_" if args.screen_prefix else ""
        screening_dir = os.path.join(args.working_dir, f"{screen_prefix}screening")
        os.makedirs(screening_dir, exist_ok=True)
        # load poses
        input_poses_path = os.path.join(screening_dir, 'screening_input_poses', 'screening_input_poses.json')
        backbones.set_work_dir(screening_dir)
        if os.path.isfile(input_poses_path):
            logging.info(f"Reading in previously selected poses from {input_poses_path}")
            previous_selection = Poses(poses=input_poses_path)
            backbones.df = previous_selection.df.loc[:, ["poses_description"]].merge(backbones.df, on="poses_description")
        else:
            if args.screen_input_selection == "weighted":
                logging.info("Selecting screening input poses randomly weighted by path score.")
                backbones.df["selection_weights"] = backbones.df['path_score'] + 0.1 # to make sure each motif can be sampled, even if path score is 0
                backbones.df = backbones.df.sample(n=min(args.screen_input_poses, len(backbones.df.index)), weights=backbones.df['selection_weights'])
            elif args.screen_input_selection == "top":
                logging.info("Selecting screening input poses according to path score.")
                backbones.filter_poses_by_rank(n=min(args.screen_input_poses, len(backbones.df.index)), score_col='path_score', ascending=False, prefix='screening_input', plot=True)
            elif args.screen_input_selection == "random":
                logging.info("Selecting screening input poses randomly.")
                backbones.df = backbones.df.sample(n=min(args.screen_input_poses, len(backbones.df.index)))
            else:
                logging.error(f"<screen_input_selection> must be one of 'weighted', 'top' or 'random', not {args.screen_input_selection}!")
                raise KeyError(f"<screen_input_selection> must be one of 'weighted', 'top' or 'random', not {args.screen_input_selection}!")
            backbones.save_poses(os.path.join(screening_dir, 'screening_input_poses'))
            backbones.save_scores(input_poses_path)

        logging.info(f"Selected {len(backbones.df.index)} poses as input for screening.")
        if backbones.df.empty:
            logging.error(f"Input dataframe is empty. Are you sure you set up a correct working directory that does not contain input poses from previous runs at {input_poses_path}?")
            raise RuntimeError(f"Input dataframe is empty. Are you sure you set up a correct working directory that does not contain input poses from previous runs at {input_poses_path}?")

        # save input backbones for later
        starting_motifs = copy.deepcopy(backbones)

        # save run name in df, makes it easier to identify where poses come from when merging results with other runs
        backbones.df["run_name"] = os.path.basename(screening_dir)

        # setup rfdiffusion options:
        backbones.df["rfdiffusion_contigs"] = backbones.df.apply(lambda row: write_rfdiffusion_contig(row['motif_contigs'], args.total_length, args.flanker_length, row["frag_length"], args.flanking, row["channel_contig"] if "channel_contig" in backbones.df.columns else None, sep=","), axis=1)
        backbones.df["rfdiffusion_inpaint_seq"] = backbones.df.apply(lambda row: write_rfdiffusion_inpaint_seq(row['motif_residues'], row['fixed_residues']), axis=1)
        backbones.df["rfdiffusion_pose_opts"] = backbones.df["rfdiffusion_contigs"] + backbones.df["rfdiffusion_inpaint_seq"]

        if args.recenter:
            logging.info(f"Parameter --recenter specified. Setting direction for custom recentering during diffusion towards {args.recenter}")
            if len(args.recenter.split(";")) != 3:
                raise ValueError("--recenter needs to be semicolon separated coordinates. E.g. --recenter=31.123;-12.123;-0.342")
            recenter = f",recenter_xyz:{args.recenter}"
        else:
            recenter = ""

        input_backbones = copy.deepcopy(backbones)
        settings = tuple(itertools.product(args.screen_decentralize_weights, args.screen_decentralize_distances))
        prefixes = [f"screen_{i+1}" for i, s in enumerate(settings)]

        for prefix, setting in zip(prefixes, settings):
            logging.info(f"Running {prefix} with settings: 'decentralize_weight: {setting[0]}, decentralize_distance: {setting[1]}")
            backbones = copy.deepcopy(input_backbones)
            backbones.df['screen_decentralize_weight'] = float(setting[0])
            backbones.df['screen_decentralize_distance'] = float(setting[1])

            # save screen number in poses
            backbones.df['screen'] = int(prefix.split('_')[1])

            # set different directory for each screen
            backbones.set_work_dir(os.path.join(screening_dir, prefix))

            # run diffusion
            diffusion_options = f"diffuser.T=50 potentials.guide_scale=5 potentials.guiding_potentials=[\\'type:substrate_contacts,weight:0\\',\\'type:custom_recenter_ROG,weight:{setting[0]},rog_weight:0,distance:{setting[1]}{recenter}\\'] potentials.guide_decay=quadratic contigmap.length={args.total_length}-{args.total_length} potentials.substrate=LIG {args.rfdiffusion_options}"
            rfdiffusion.run(
                poses=backbones,
                prefix="rfdiffusion",
                num_diffusions=args.screen_num_rfdiffusions,
                options=diffusion_options,
                pose_options=backbones.df["rfdiffusion_pose_opts"].to_list(),
                update_motifs=["fixed_residues", "motif_residues"],
                fail_on_missing_output_poses=False
            )

            # remove channel chain (chain B)
            logging.info("Diffusion completed, removing channel chain from diffusion outputs.")
            if "channel_contig" in backbones.df.columns:
                chain_remover.run(
                    poses = backbones,
                    prefix = "channel_removed",
                    chains = "B"
                )
            else:
                backbones.df["channel_removed_location"] = backbones.df["rfdiffusion_location"]

            # create updated reference frags:
            if not os.path.isdir((updated_ref_frags_dir := os.path.join(backbones.work_dir, "updated_reference_frags"))):
                os.makedirs(updated_ref_frags_dir)

            logging.info("Channel chain removed, now renumbering reference fragments.")
            backbones.df["updated_reference_frags_location"] = update_and_copy_reference_frags(
                input_df = backbones.df,
                ref_col = "input_poses",
                desc_col = "poses_description",
                prefix = "rfdiffusion",
                out_pdb_path = updated_ref_frags_dir,
                keep_ligand_chain = "Z"
            )

            # replace combined ligand with original ligands (do it for single ligands as well, to change the ligand name from LIG to original name (important for covalent bonds))
            split_combined_ligand(updated_ref_frags_dir, ligand_paths)

            # update covalent bonds info
            backbones.df["covalent_bonds"] = backbones.df.apply(lambda row: update_covalent_bonds_info(row['covalent_bonds'], row["template_fixedres"], row["fixed_residues"]), axis=1)

            # calculate ROG after RFDiffusion, when channel chain is already removed:
            logging.info("Calculating rfdiffusion_rog and rfdiffusion_catres_rmsd")
            rog_calculator.run(poses=backbones, prefix="rfdiffusion_rog")

            # calculate rmsds
            catres_motif_bb_rmsd.run(
                poses = backbones,
                prefix = "rfdiffusion_catres"
            )
            fragment_motif_bb_rmsd.run(
                poses = backbones,
                prefix = "rfdiffusion_motif"
            )

            # add back the ligand:
            logging.info("Metrics calculated, now adding Ligand chain back into backbones.")
            chain_adder.superimpose_add_chain(
                poses = backbones,
                prefix = "post_rfdiffusion_ligand",
                ref_col = "updated_reference_frags_location",
                target_motif = "fixed_residues",
                copy_chain = "Z"
            )

            # calculate ligand stats
            logging.info("Calculating Ligand Statistics")
            ligand_clash.run(poses=backbones, prefix="rfdiffusion_ligand")
            ligand_contacts.run(poses=backbones, prefix="rfdiffusion_lig")
            backbones.df = calculate_contact_score(df=backbones.df, contact_col="rfdiffusion_lig_contacts", score_col="rfdiffusion_contacts_score", target_value=args.contacts_target_value)

            # plot rfdiffusion_stats
            results_dir = os.path.join(backbones.work_dir, "results")
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir, exist_ok=True)
            plots.violinplot_multiple_cols(
                dataframe = backbones.df,
                cols = ["rfdiffusion_catres_rmsd", "rfdiffusion_rog_data", "rfdiffusion_lig_contacts", "rfdiffusion_ligand_clashes", "rfdiffusion_plddt"],
                titles = ["Template RMSD", "ROG", "Ligand Contacts", "Ligand Clashes", "RFdiffusion pLDDT"],
                y_labels = ["RMSD [\u00C5]", "ROG [\u00C5]", "#CA", "#Clashes", "pLDDT"],
                dims = [(0,5), None, None, None, (0.8,1)],
                out_path = os.path.join(results_dir, "rfdiffusion_statistics.png"),
                show_fig = False
            )

            if args.rfdiffusion_max_clashes:
                backbones.filter_poses_by_value(score_col="rfdiffusion_ligand_clashes", value=args.rfdiffusion_max_clashes, operator="<=", prefix="rfdiffusion_ligand_clashes", plot=True)
            if args.rfdiffusion_max_rog:
                backbones.filter_poses_by_value(score_col="rfdiffusion_rog_data", value=args.rfdiffusion_max_rog, operator="<=", prefix="rfdiffusion_rog", plot=True)
            if args.min_contacts:
                backbones.filter_poses_by_value(score_col="rfdiffusion_lig_contacts", value=args.min_contacts, operator=">=", prefix="rfdiffusion_lig_contacts", plot=True)
            if args.rfdiffusion_catres_bb_rmsd:
                backbones.filter_poses_by_value(score_col="rfdiffusion_catres_rmsd", value=args.rfdiffusion_catres_bb_rmsd, operator="<=", prefix="rfdiffusion_catres_bb_rmsd", plot=True)
            if args.rfdiffusion_motif_bb_rmsd:
                backbones.filter_poses_by_value(score_col="rfdiffusion_motif_rmsd", value=args.rfdiffusion_motif_bb_rmsd, operator="<=", prefix="rfdiffusion_motif_bb_rmsd", plot=True)

            if len(backbones.df) == 0:
                logging.warning(f"No poses passed RFdiffusion filtering steps during {prefix}")
                prefixes.remove(prefix)
                continue

            # run LigandMPNN
            if args.screen_skip_mpnn_rlx_mpnn:
                logging.info(f"Running LigandMPNN on {len(backbones)} poses. Designing {args.screen_num_mpnn_sequences} sequences per pose.")
                ligand_mpnn.run(
                    poses = backbones,
                    prefix = "postdiffusion_ligandmpnn",
                    nseq = args.screen_num_mpnn_sequences,
                    options = ligandmpnn_options,
                    model_type = "ligand_mpnn",
                    fixed_res_col = "fixed_residues",
                    return_seq_threaded_pdbs_as_pose= False
                )

                # calculate composite score
                backbones.calculate_composite_score(
                    name="pre_esm_comp_score",
                    scoreterms=["rfdiffusion_lig_contacts", "rfdiffusion_ligand_clashes", "rfdiffusion_rog_data", "postdiffusion_ligandmpnn_overall_confidence"],
                    weights=[-1, 1, 1, -1,],
                    plot=True
                )

                backbones.filter_poses_by_rank(n=args.screen_esm_input_poses, score_col="pre_esm_comp_score", prefix="esm_input_filter", plot=True, plot_cols=["rfdiffusion_lig_contacts", "rfdiffusion_ligand_clashes", "rfdiffusion_rog_data", "postdiffusion_ligandmpnn_overall_confidence"])


            else:
                ligand_mpnn.run(
                    poses = backbones,
                    prefix = "postdiffusion_ligandmpnn",
                    nseq = args.screen_num_seq_thread_sequences,
                    options = ligandmpnn_options,
                    model_type = "ligand_mpnn",
                    fixed_res_col = "fixed_residues",
                    return_seq_threaded_pdbs_as_pose = True
                )

                # add covalent bonds info to poses pre-relax
                backbones = add_covalent_bonds_info(poses=backbones, prefix="bbopt_cov_info", covalent_bonds_col="covalent_bonds")

                # optimize backbones
                backbones.df['screen_bbopt_opts'] = [write_bbopt_opts(row=row, cycle=1, total_cycles=5, reference_location_col="updated_reference_frags_location", cat_res_col="fixed_residues", motif_res_col="motif_residues", ligand_chain="Z") for _, row in backbones.df.iterrows()]
                rosetta.run(
                    poses = backbones,
                    prefix = "bbopt",
                    rosetta_application="rosetta_scripts.default.linuxgccrelease",
                    nstruct = 1,
                    options = bb_opt_options,
                    pose_options='screen_bbopt_opts'
                )

                # filter backbones down to starting backbones
                backbones.filter_poses_by_rank(n=3, score_col="bbopt_total_score", remove_layers=2)

                # calculate ligand contacts and clashes again
                ligand_clash.run(poses=backbones, prefix="bbopt_ligand")
                ligand_contacts.run(poses=backbones, prefix="bbopt_lig")
                backbones.df = calculate_contact_score(df=backbones.df, contact_col="bbopt_lig_contacts", score_col="bbopt_contacts_score", target_value=9)

                # run ligandmpnn on relaxed poses
                ligand_mpnn.run(
                    poses = backbones,
                    prefix = "mpnn",
                    nseq = args.screen_num_mpnn_sequences,
                    model_type = "ligand_mpnn",
                    options = ligandmpnn_options,
                    fixed_res_col = "fixed_residues",
                )

                # filter backbones down to starting backbones
                backbones.filter_poses_by_rank(n=args.screen_num_mpnn_sequences, score_col="mpnn_overall_confidence", ascending=False, remove_layers=3)

                # calculate composite score
                backbones.calculate_composite_score(
                    name="pre_esm_comp_score",
                    scoreterms=["bbopt_total_score", "bbopt_contacts_score", "bbopt_ligand_clashes", "rfdiffusion_rog_data", "mpnn_overall_confidence"],
                    weights=[1, 1, 1, 1, -1],
                    plot=True
                )

                # filter esm input poses
                backbones.filter_poses_by_rank(n=args.screen_esm_input_poses, score_col="pre_esm_comp_score", prefix="esm_input_filter", plot=True, plot_cols=["bbopt_total_score", "bbopt_lig_contacts", "bbopt_ligand_clashes", "rfdiffusion_rog_data", "mpnn_overall_confidence"])

            # predict with ESMFold
            logging.info(f"LigandMPNN finished, now predicting {len(backbones)} sequences using ESMFold.")
            esmfold.run(
                poses = backbones,
                prefix = "esm"
            )

            # calculate ROG
            rog_calculator.run(poses=backbones, prefix="esm_rog")

            # calculate RMSDs (backbone, motif, fixedres)
            logging.info(f"Prediction of {len(backbones.df.index)} sequences completed. Calculating RMSDs to rfdiffusion backbone and reference fragment.")
            catres_motif_bb_rmsd.run(poses = backbones, prefix = "esm_catres_bb")
            bb_rmsd.run(poses = backbones, ref_col="rfdiffusion_location", prefix = "esm_backbone")
            catres_motif_heavy_rmsd.run(poses = backbones, prefix = "esm_catres_heavy")
            fragment_motif_bb_rmsd.run(poses = backbones, prefix = "esm_motif")

            # calculate TM-Score and get sc-tm score:
            tm_score_calculator.run(
                poses = backbones,
                prefix = "esm_tm",
                ref_col = "channel_removed_location",
            )

            # filter poses:
            backbones.filter_poses_by_value(score_col="esm_plddt", value=70, operator=">=", prefix="screen_esm_plddt", plot=True)
            backbones.filter_poses_by_value(score_col="esm_tm_TM_score_ref", value=0.9, operator=">=", prefix="screen_esm_TMscore", plot=True)
            backbones.filter_poses_by_value(score_col="esm_catres_bb_rmsd", value=1.5, operator="<=", prefix="screen_esm_catres_bb_rmsd", plot=True)
            backbones.filter_poses_by_value(score_col="esm_motif_rmsd", value=1.5, operator="<=", prefix="esm_motif_rmsd", plot=True)
            backbones.filter_poses_by_value(score_col="esm_rog_data", value=args.rfdiffusion_max_rog, operator="<=", prefix="esm_rog", plot=True)

            # add back ligand and determine pocket-ness!
            logging.info("Adding Ligand back into the structure for ligand-based pocket prediction.")
            chain_adder.superimpose_add_chain(
                poses = backbones,
                prefix = "post_prediction_ligand",
                ref_col = "updated_reference_frags_location",
                target_motif = "fixed_residues",
                copy_chain = "Z"
            )

            ligand_clash.run(poses=backbones, prefix="esm_ligand")
            ligand_contacts.run(poses=backbones, prefix="esm_lig")
            backbones.df = calculate_contact_score(df=backbones.df, contact_col="esm_lig_contacts", score_col="esm_contacts_score", target_value=args.contacts_target_value)

            backbones.filter_poses_by_value(score_col="esm_lig_contacts", value=args.min_contacts, operator=">=", prefix="esm_lig_contacts", plot=True)


            # calculate multi-scorerterm score for the final backbone filter:
            screen_scoreterms = ["esm_plddt", "esm_tm_TM_score_ref", "esm_catres_bb_rmsd", "esm_motif_rmsd", "esm_contacts_score", "esm_ligand_clashes", "esm_rog_data"]
            screen_weights = [-1, -1, 4, 1, 1, 1, 1]
            backbones.calculate_composite_score(
                name="screen_composite_score",
                scoreterms=screen_scoreterms,
                weights=screen_weights,
                plot=True
            )

            # filter down to rfdiffusion backbones
            backbones.filter_poses_by_rank(
                n=1,
                score_col="screen_composite_score",
                prefix=f"{prefix}_backbone_filter",
                plot=True,
                plot_cols=screen_scoreterms,
                remove_layers=1 if args.screen_skip_mpnn_rlx_mpnn else 3
            )

            # plot outputs
            logging.info("Plotting outputs.")
            cols = ["rfdiffusion_catres_rmsd", "esm_plddt", "esm_backbone_rmsd", "esm_catres_heavy_rmsd", "esm_motif_rmsd", "esm_tm_sc_tm", "esm_rog_data", "esm_lig_contacts", "esm_ligand_clashes"]
            titles = ["RFDiffusion Motif\nBackbone RMSD", "ESMFold pLDDT", "ESMFold BB-Ca RMSD", "ESMFold Sidechain\nRMSD", "ESMFold Motif RMSD", "SC-TM Score", "Radius of Gyration", "Ligand Contacts", "Ligand Clashes"]
            y_labels = ["Angstrom", "pLDDT", "Angstrom", "Angstrom", "Angstrom", "TM Score", "Angstrom", "#", "#"]
            dims = [None for _ in cols]

            # plot results
            plots.violinplot_multiple_cols(
                dataframe = backbones.df,
                cols = cols,
                titles = titles,
                y_labels = y_labels,
                dims = dims,
                out_path = os.path.join(results_dir, "design_results.png"),
                show_fig = False
            )

            #logging.info(f"Fraction of RFdiffusion design-successful backbones that contain active-site pocket: {pocket_containing_fraction}")
            backbones.df.sort_values("screen_composite_score", ascending=True, inplace=True)
            backbones.df.reset_index(drop=True, inplace=True)
            backbones.reindex_poses(prefix="reindex", remove_layers=2 if args.screen_skip_mpnn_rlx_mpnn else 4, force_reindex=True)

            # copy filtered poses to new location
            backbones.save_poses(out_path=results_dir)
            backbones.save_poses(out_path=results_dir, poses_col="input_poses")
            backbones.save_scores(out_path=results_dir)
            backbones.save_scores()

            # write pymol alignment script?
            logging.info(f"Created results/ folder and writing pymol alignment script for best backbones at {results_dir}")
            write_pymol_alignment_script(
                df=backbones.df,
                scoreterm="screen_composite_score",
                top_n=np.min([len(backbones), 25]),
                path_to_script=os.path.join(results_dir, "align_results.pml"),
                ref_motif_col = "template_fixedres",
                ref_catres_col = "template_fixedres",
                target_catres_col = "fixed_residues",
                target_motif_col = "fixed_residues"
            )

            # write successfull motifs to file, so that they can be read in again
            backbones.df.drop_duplicates(subset=["input_poses"], keep="first", inplace=True)
            successfull_motifs = starting_motifs.df.merge(backbones.df['input_poses'], on="input_poses")
            successfull_motifs.reset_index(drop=True, inplace=True)
            successfull_motifs.to_json(os.path.join(results_dir, "successful_input_motifs.json"))

        backbones = combine_screening_results(dir=args.working_dir, prefixes=prefixes, screen_prefix=args.screen_prefix, scoreterms=screen_scoreterms, weights=screen_weights, residue_cols=residue_cols, input_motifs=starting_motifs)
        backbones.set_work_dir(args.working_dir)
        backbones.save_scores()

        if args.skip_refinement:
            logging.info(f"Skipping refinement. Run concluded, output can be found in {results_dir}")
            sys.exit(1)
        else:
            args.ref_input_json = backbones.scorefile

    ############################################# REFINEMENT ########################################################
    if args.ref_input_json:
        if args.ref_prefix:
            ref_prefix = f"{args.ref_prefix}_"
        elif args.screen_prefix:
            ref_prefix = f"{args.screen_prefix}_"
        else: ref_prefix = ""

        refinement_dir = os.path.join(args.working_dir, f"{ref_prefix}refinement")
        os.makedirs(refinement_dir, exist_ok=True)

        backbones.set_work_dir(refinement_dir)

        if args.ref_input_poses_per_bb:
            logging.info("Filtering refinement input poses on per backbone level according to screen_composite_score...")
            backbones.filter_poses_by_rank(n=args.ref_input_poses_per_bb, score_col='screen_composite_score', remove_layers=1, prefix='refinement_input_bb', plot=True, plot_cols=["esm_plddt", "esm_tm_TM_score_ref", "esm_catres_bb_rmsd", "esm_catres_heavy_rmsd", "esm_lig_contacts", "esm_ligand_clashes", "esm_rog_data"])
        if args.ref_input_poses:
            logging.info("Filtering refinement input according to screen_composite_score...")
            backbones.filter_poses_by_rank(n=args.ref_input_poses, score_col='screen_composite_score', prefix='refinement_input', plot=True, plot_cols=["esm_plddt", "esm_tm_TM_score_ref", "esm_catres_bb_rmsd", "esm_catres_heavy_rmsd", "esm_lig_contacts", "esm_ligand_clashes", "esm_rog_data"])

        # use reduced motif if specified
        if args.use_reduced_motif:
            backbones.df["motif_residues"] = backbones.df.apply(lambda row: create_reduced_motif(row['fixed_residues'], row['motif_residues']), axis=1)

        # create refinement input poses dir
        refinement_input_dir = os.path.join(refinement_dir, "refinement_input_poses")
        os.makedirs(refinement_input_dir, exist_ok=True)
        backbones.save_scores(out_path=os.path.join(refinement_input_dir, "refinement_input_scores.json"), out_format="json")
        backbones.save_poses(out_path=refinement_input_dir)
        backbones.save_poses(out_path=refinement_input_dir, poses_col="input_poses")
        write_pymol_alignment_script(
            df=backbones.df,
            scoreterm="screen_composite_score",
            top_n=np.min([len(backbones), 25]),
            path_to_script=os.path.join(refinement_input_dir, "align_poses.pml"),
            ref_motif_col = "template_fixedres",
            ref_catres_col = "template_fixedres",
            target_catres_col = "fixed_residues",
            target_motif_col = "fixed_residues"
        )

        logging.info("Plotting refinement input data.")
        cols = ["esm_plddt", "esm_backbone_rmsd", "esm_catres_bb_rmsd", "esm_catres_heavy_rmsd", "esm_tm_sc_tm", "esm_rog_data", "esm_lig_contacts", "esm_ligand_clashes", "screen", "screen_decentralize_weight", "screen_decentralize_distance"]
        titles = ["ESMFold pLDDT", "ESMFold BB-Ca RMSD", "ESMFold fixed res\nBB-Ca RMSD", "ESMFold Sidechain\nRMSD", "SC-TM Score", "Radius of Gyration", "Ligand Contacts", "Ligand Clashes", "screen number", "decentralize weight", "decentralize distance"]
        y_labels = ["pLDDT", "Angstrom", "Angstrom", "Angstrom", "TM Score", "Angstrom", "#", "#", "#", "AU", "Angstrom"]
        dims = [None for _ in cols]

        plots.violinplot_multiple_cols(
            dataframe = backbones.df,
            cols = cols,
            titles = titles,
            y_labels = y_labels,
            dims = dims,
            out_path = os.path.join(backbones.plots_dir, f"{ref_prefix}refinement_input_poses.png"),
            show_fig = False
        )

        shutil.copy(os.path.join(backbones.plots_dir, f"{ref_prefix}refinement_input_poses.png"), refinement_input_dir)

        # combine all cycle results in a dict
        ref_dict = {}

        for cycle in range(args.ref_start_cycle, args.ref_cycles+1):
            cycle_work_dir = os.path.join(refinement_dir, f"cycle_{cycle}")
            backbones.set_work_dir(cycle_work_dir)
            logging.info(f"Starting refinement cycle {cycle} in directory {cycle_work_dir}")

            logging.info("Threading sequences on poses with LigandMPNN...")
            # run ligandmpnn, return pdbs as poses
            ligand_mpnn.run(
                poses = backbones,
                prefix = f"cycle_{cycle}_seq_thread",
                nseq = args.ref_seq_thread_num_mpnn_seqs,
                model_type = "ligand_mpnn",
                options = ligandmpnn_options,
                fixed_res_col = "fixed_residues",
                return_seq_threaded_pdbs_as_pose=True
            )

            # optimize backbones
            logging.info("Optimizing backbones with Rosetta...")
            backbones.df[f'cycle_{cycle}_bbopt_opts'] = [write_bbopt_opts(row=row, cycle=cycle, total_cycles=args.ref_cycles, reference_location_col="updated_reference_frags_location", cat_res_col="fixed_residues", motif_res_col="motif_residues", ligand_chain="Z") for _, row in backbones.df.iterrows()]

            # add covalent bonds info to poses pre-relax
            backbones = add_covalent_bonds_info(poses=backbones, prefix=f"cycle_{cycle}_bbopt_cov_info", covalent_bonds_col="covalent_bonds")

            rosetta.run(
                poses = backbones,
                prefix = f"cycle_{cycle}_bbopt",
                rosetta_application="rosetta_scripts.default.linuxgccrelease",
                nstruct = 1,
                options = bb_opt_options,
                pose_options=f'cycle_{cycle}_bbopt_opts'
            )

            # filter backbones down to starting backbones
            logging.info("Selecting poses with lowest total score for each input backbone...")
            backbones.filter_poses_by_rank(n=1, score_col=f"cycle_{cycle}_bbopt_total_score", remove_layers=2)

            # run ligandmpnn on optimized poses
            logging.info("Generating sequences for each pose...")
            ligand_mpnn.run(
                poses = backbones,
                prefix = f"cycle_{cycle}_mpnn",
                nseq = args.ref_num_mpnn_seqs,
                model_type = "ligand_mpnn",
                options = ligandmpnn_options,
                fixed_res_col = "fixed_residues",
            )

            # predict structures using ESMFold
            logging.info("Predicting sequences with ESMFold...")
            esmfold.run(
                poses = backbones,
                prefix = f"cycle_{cycle}_esm",
            )

            # calculate rmsds, TMscores and clashes
            logging.info("Calculating post-ESMFold RMSDs...")
            catres_motif_heavy_rmsd.run(poses = backbones, prefix = f"cycle_{cycle}_esm_catres_heavy")
            catres_motif_bb_rmsd.run(poses = backbones, prefix = f"cycle_{cycle}_esm_catres_bb")
            fragment_motif_bb_rmsd.run(poses = backbones, prefix = f"cycle_{cycle}_esm_motif")
            bb_rmsd.run(poses = backbones, ref_col=f"cycle_{cycle}_bbopt_location", prefix = f"cycle_{cycle}_esm_backbone")
            tm_score_calculator.run(poses = backbones, prefix = f"cycle_{cycle}_esm_tm", ref_col = f"cycle_{cycle}_bbopt_location")

            # calculate cutoff & filter
            logging.info("Applying post-ESMFold backbone filters...")
            plddt_cutoff = ramp_cutoff(args.ref_plddt_cutoff_start, args.ref_plddt_cutoff_end, cycle, args.ref_cycles)
            catres_bb_rmsd_cutoff = ramp_cutoff(args.ref_catres_bb_rmsd_cutoff_start, args.ref_catres_bb_rmsd_cutoff_end, cycle, args.ref_cycles)
            motif_bb_rmsd_cutoff = ramp_cutoff(args.ref_motif_rmsd_cutoff_start, args.ref_motif_rmsd_cutoff_end, cycle, args.ref_cycles)
            backbones.filter_poses_by_value(score_col=f"cycle_{cycle}_esm_plddt", value=plddt_cutoff, operator=">=", prefix=f"cycle_{cycle}_esm_plddt", plot=True)
            backbones.filter_poses_by_value(score_col=f"cycle_{cycle}_esm_tm_TM_score_ref", value=0.9, operator=">=", prefix=f"cycle_{cycle}_esm_TM_score", plot=True)
            backbones.filter_poses_by_value(score_col=f"cycle_{cycle}_esm_catres_bb_rmsd", value=catres_bb_rmsd_cutoff, operator="<=", prefix=f"cycle_{cycle}_esm_catres_bb", plot=True)
            backbones.filter_poses_by_value(score_col=f"cycle_{cycle}_esm_motif_rmsd", value=motif_bb_rmsd_cutoff, operator="<=", prefix=f"cycle_{cycle}_esm_motif_bb", plot=True)

            # add ligand to poses
            logging.info("Adding ligand to ESMFold predictions...")
            chain_adder.superimpose_add_chain(
                poses = backbones,
                prefix = f"cycle_{cycle}_ligand",
                ref_col = "updated_reference_frags_location",
                target_motif = "fixed_residues",
                copy_chain = "Z"
            )

            # calculate ligand clashes and ligand contacts
            ligand_clash.run(poses=backbones, prefix=f"cycle_{cycle}_esm_ligand")
            ligand_contacts.run(poses=backbones, prefix=f"cycle_{cycle}_esm_lig")
            backbones.df = calculate_contact_score(df=backbones.df, contact_col=f"cycle_{cycle}_esm_lig_contacts", score_col=f"cycle_{cycle}_esm_contacts_score", target_value=args.contacts_target_value)

            backbones.filter_poses_by_value(score_col=f"cycle_{cycle}_esm_lig_contacts", value=args.min_contacts, operator=">=", prefix=f"cycle_{cycle}_esm_lig_contacts", plot=True)

            # calculate multi-scoreterm score:
            logging.info("Calculating composite score for post-esm evaluation...")
            ref_comp_scoreterms = [f"cycle_{cycle}_esm_plddt", f"cycle_{cycle}_esm_catres_bb_rmsd", f"cycle_{cycle}_esm_catres_heavy_rmsd", f"cycle_{cycle}_esm_motif_rmsd", f"cycle_{cycle}_esm_contacts_score", f"cycle_{cycle}_esm_ligand_clashes"]
            ref_comp_weights = [-2, 4, 1, 1, 2, 4]
            backbones.calculate_composite_score(
                name=f"cycle_{cycle}_refinement_composite_score",
                scoreterms=ref_comp_scoreterms,
                weights=ref_comp_weights,
                plot=True
            )

            # define number of index layers added during refinement so far
            layers = 3

            # filter down to refinement input poses
            logging.info("Filtering poses according to composite score...")
            backbones.filter_poses_by_rank(
                n=5,
                score_col=f"cycle_{cycle}_refinement_composite_score",
                prefix=f"cycle_{cycle}_refinement_composite_score",
                plot=True,
                plot_cols=ref_comp_scoreterms,
                remove_layers=layers
            )

            # add covalent bonds info to poses
            backbones = add_covalent_bonds_info(poses=backbones, prefix=f"cycle_{cycle}_cov_info", covalent_bonds_col="covalent_bonds")

            # define number of index layers that were added during refinement cycle (higher in subsequent cycles because reindexing adds a layer)
            if cycle > 1:
                layers += 1
            else:
                trajectory_plots = instantiate_trajectory_plotting(backbones.plots_dir, scores = trajectory_scoreterms)

            # manage ref output
            backbones.reindex_poses(prefix=f"cycle_{cycle}_reindex", remove_layers=layers, force_reindex=True)

            # update trajectory plots
            trajectory_plots = update_trajectory_plotting(trajectory_plots=trajectory_plots, poses=backbones, prefix=f"cycle_{cycle}_esm")

            # save unfiltered backbones
            ref_dict[cycle] = copy.deepcopy(backbones)

            # filter down to rfdiffusion backbones
            logging.info("Filtering poses according to composite score...")
            backbones.filter_poses_by_rank(
                n=args.ref_num_cycle_poses,
                score_col=f"cycle_{cycle}_refinement_composite_score",
                prefix=f"cycle_{cycle}_refinement_composite_score",
                plot=True,
                plot_cols=ref_comp_scoreterms,
                remove_layers=1
            )

            results_dir = os.path.join(backbones.work_dir, f"cycle_{cycle}_results")
            create_results_dir(poses=backbones, dir=results_dir, score_col=f"cycle_{cycle}_refinement_composite_score", plot_cols=ref_comp_scoreterms)

        # combine results from all refinement cycles, if set
        if args.ref_skip_all_cycle_pooling:
            backbones = ref_dict[cycle]
            for score in [scoreterm for scoreterm in backbones.df.columns if scoreterm.startswith(f"cycle_{cycle}_")]:
                raw_score = "_".join(score.split("_")[2:])
                backbones.df[f"cycle_final_{raw_score}"] = backbones.df[score]
        else:
            cycle = "final"
            backbones = pool_all_cycle_results(ref_dict, plddt_cutoff, catres_bb_rmsd_cutoff, motif_bb_rmsd_cutoff, ref_comp_scoreterms, ref_comp_weights, refinement_dir)

        ref_comp_scoreterms = [f"cycle_final_{'_'.join(score.split('_')[2:])}" for score in ref_comp_scoreterms]

        # sort output
        backbones.df.sort_values("cycle_final_refinement_composite_score", ascending=True, inplace=True)
        backbones.df.reset_index(drop=True, inplace=True)

        # update trajectory plots
        trajectory_plots = update_trajectory_plotting(trajectory_plots=trajectory_plots, poses=backbones, prefix="cycle_final_esm")

        # create results directory
        refinement_results_dir = os.path.join(args.working_dir, f"{ref_prefix}refinement_results")
        create_results_dir(poses=backbones, dir=refinement_results_dir, score_col="cycle_final_refinement_composite_score", plot_cols=ref_comp_scoreterms)
        for file in os.listdir(backbones.plots_dir):
            if file.startswith("trajectory") and file.endswith(".png"):
                shutil.copy(os.path.join(backbones.plots_dir, file), refinement_results_dir)
        backbones.save_scores(out_path=os.path.join(refinement_results_dir, f"results_{ref_prefix}refinement"))
        backbones.set_work_dir(args.working_dir)
        backbones.save_scores()

        if args.skip_evaluation:
            logging.info(f"Skipping evaluation. Run concluded, per-backbone output can be found in {os.path.join(backbones.work_dir, f'cycle_{cycle}_results')}. Overall results can be found in {refinement_results_dir}.")
            sys.exit(1)
        else:
            args.eval_input_json = backbones.scorefile

    ########################### EVALUATION ###########################
    if args.eval_input_json:
        if args.eval_prefix:
            eval_prefix = f"{args.eval_prefix}_"
        elif args.ref_prefix:
            eval_prefix = f"{args.ref_prefix}_"
        elif args.screen_prefix:
            eval_prefix = f"{args.screen_prefix}_"
        else: eval_prefix = ""

        backbones.set_work_dir(os.path.join(args.working_dir, f"{eval_prefix}evaluation"))

        score_cols = ["cycle_final_esm_plddt", "cycle_final_esm_catres_bb_rmsd", "cycle_final_esm_catres_heavy_rmsd", "cycle_final_esm_motif_rmsd", "cycle_final_esm_contacts_score", "cycle_final_esm_ligand_clashes"]
        if args.eval_input_poses_per_bb:
            logging.info("Filtering evaluation input poses on per backbone level according to cycle_final_refinement_composite_score...")
            backbones.filter_poses_by_rank(n=args.eval_input_poses_per_bb, score_col="cycle_final_refinement_composite_score", remove_layers=1, prefix="evaluation_input_per_bb", plot=True, plot_cols=score_cols)
        if args.eval_input_poses:
            logging.info("Filtering evaluation input poses according to cycle_final_refinement_composite_score...")
            backbones.filter_poses_by_rank(n=args.eval_input_poses, score_col="cycle_final_refinement_composite_score", prefix="evaluation_input", plot=True, plot_cols=score_cols)

        evaluation_input_poses_dir = os.path.join(backbones.work_dir, "evaluation_input_poses")
        os.makedirs(evaluation_input_poses_dir, exist_ok=True)
        backbones.save_poses(out_path=evaluation_input_poses_dir)
        backbones.save_poses(out_path=evaluation_input_poses_dir, poses_col="input_poses")
        backbones.save_scores(out_path=evaluation_input_poses_dir)

        # write pymol alignment script
        logging.info(f"Writing pymol alignment script for evaluation input poses at {evaluation_input_poses_dir}.")
        write_pymol_alignment_script(
            df = backbones.df,
            scoreterm = "cycle_final_refinement_composite_score",
            top_n = np.min([len(backbones.df.index), 25]),
            path_to_script = os.path.join(evaluation_input_poses_dir, "align_input_poses.pml"),
            ref_motif_col = "template_fixedres",
            ref_catres_col = "template_fixedres",
            target_catres_col = "fixed_residues",
            target_motif_col = "fixed_residues"
        )

        # if run was interrupted, create new trajectories
        try:
            if not trajectory_plots:
                trajectory_plots = instantiate_trajectory_plotting(backbones.plots_dir, scores = trajectory_scoreterms)
                trajectory_plots = update_trajectory_plotting(trajectory_plots=trajectory_plots, poses=backbones, prefix="cycle_final_esm")
        except:
            trajectory_plots = instantiate_trajectory_plotting(backbones.plots_dir, scores = trajectory_scoreterms)
            trajectory_plots = update_trajectory_plotting(trajectory_plots=trajectory_plots, poses=backbones, prefix="cycle_final_esm")

        backbones.convert_pdb_to_fasta(prefix="eval_fasta_conversion", update_poses=True)

        # set for easier merging later
        backbones.df["eval_pre_af2_description"] = backbones.df["poses_description"]

        # run af2
        colabfold.run(
            poses=backbones,
            prefix="eval_af2",
            return_top_n_poses=5,
            options="--msa-mode single_sequence"
        )

        # calculate backbone rmsds
        catres_motif_bb_rmsd.run(poses=backbones, prefix="eval_af2_catres_bb")
        bb_rmsd.run(poses=backbones, prefix="eval_af2_backbone", ref_col="cycle_final_bbopt_location")
        bb_rmsd.run(poses=backbones, prefix="eval_af2_ESM_bb", ref_col="cycle_final_esm_location")
        tm_score_calculator.run(poses=backbones, prefix="eval_af2_tm", ref_col="cycle_final_bbopt_location")
        tm_score_calculator.run(poses=backbones, prefix="eval_af2_ESM_tm", ref_col="cycle_final_esm_location")
        fragment_motif_bb_rmsd.run(poses = backbones, prefix = "eval_af2_motif")

        # calculate average plddt
        backbones = calculate_mean_scores(backbones, scores=["eval_af2_plddt", "eval_af2_catres_bb_rmsd", "eval_af2_backbone_rmsd", "eval_af2_ESM_bb_rmsd", "eval_af2_tm_TM_score_ref", "eval_af2_ESM_tm_TM_score_ref", "eval_af2_motif_rmsd"], remove_layers=1)

        # filter for af2 top model
        backbones.filter_poses_by_rank(n=1, score_col="eval_af2_plddt", ascending=False, remove_layers=1)

        # apply rest of the filters
        backbones.filter_poses_by_value(score_col="eval_af2_plddt", value=args.eval_plddt_cutoff, operator=">=", prefix="eval_af2_plddt", plot=True)
        backbones.filter_poses_by_value(score_col="eval_af2_tm_TM_score_ref", value=0.9, operator=">=", prefix="eval_af2_TM_score", plot=True)
        backbones.filter_poses_by_value(score_col="eval_af2_catres_bb_rmsd", value=args.eval_catres_bb_rmsd_cutoff, operator="<=", prefix="eval_af2_catres_bb_rmsd", plot=True)

        # repack with attnpacker
        if args.attnpacker_repack:
            attnpacker.run(
                poses=backbones,
                prefix="eval_packing"
            )

        # calculate sc rmsd
        catres_motif_heavy_rmsd.run(poses=backbones, prefix="eval_af2_catres_heavy")

        # add ligand chain
        chain_adder.superimpose_add_chain(
            poses = backbones,
            prefix = "eval_ligand",
            ref_col = "updated_reference_frags_location",
            target_motif = "fixed_residues",
            copy_chain = "Z"
        )

        # calculate ligand clashes and ligand contacts
        ligand_clash.run(poses=backbones, prefix="eval_af2_ligand")
        ligand_contacts.run(poses=backbones, prefix="eval_af2_lig")
        backbones.df = calculate_contact_score(df=backbones.df, contact_col="eval_af2_lig_contacts", score_col="eval_af2_contacts_score", target_value=args.contacts_target_value)

        # filter for ligand contacts
        backbones.filter_poses_by_value(score_col="eval_af2_lig_contacts", value=args.min_contacts, operator=">=", prefix="eval_af2_lig_contacts", plot=True)

        # add covalent bonds info
        backbones = add_covalent_bonds_info(poses=backbones, prefix="eval_af2_cov_info", covalent_bonds_col="covalent_bonds")

        # create identifier for easier merging
        backbones.df["eval_pre_relax_description"] = backbones.df["poses_description"]
        rlx_poses = copy.deepcopy(backbones)

        # relax predictions with ligand present
        rosetta.run(
            poses = rlx_poses,
            prefix = "eval_af2_fastrelax",
            rosetta_application="rosetta_scripts.default.linuxgccrelease",
            nstruct = 15,
            options = fr_options
        )

        # calculate RMSDs of relaxed poses
        catres_motif_heavy_rmsd.run(poses = rlx_poses, prefix = "eval_af2_postrelax_catres_heavy")
        catres_motif_bb_rmsd.run(poses = rlx_poses, prefix = "eval_af2_postrelax_catres_bb")
        ligand_rmsd.run(poses = rlx_poses, prefix = "eval_af2_postrelax_ligand")

        # calculate average scores
        rlx_poses = calculate_mean_scores(rlx_poses, scores=["eval_af2_postrelax_catres_heavy_rmsd", "eval_af2_postrelax_catres_bb_rmsd", "eval_af2_postrelax_ligand_rmsd", "eval_af2_fastrelax_total_score", "eval_af2_fastrelax_sap_score"], remove_layers=1)

        # combine relaxed poses into a single multimodel pdb
        combined = combine_relax_output(rlx_poses.df, "poses", os.path.join(rlx_poses.work_dir, "relaxed_combined"), "eval_relaxed_combined_path", "eval_pre_relax_description", "eval_af2_fastrelax_total_score", 1)
        rlx_poses.df = rlx_poses.df.merge(combined, on="eval_pre_relax_description", how="left")

        # filter for top structure
        rlx_poses.filter_poses_by_rank(n=1, score_col="eval_af2_fastrelax_total_score", ascending=True, remove_layers=1)

        # merge with original poses
        backbones.df = backbones.df.merge(rlx_poses.df[["eval_pre_relax_description", "eval_relaxed_combined_path", "eval_af2_postrelax_catres_heavy_rmsd_mean", "eval_af2_postrelax_catres_bb_rmsd_mean", "eval_af2_postrelax_ligand_rmsd_mean", "eval_af2_fastrelax_total_score", "eval_af2_fastrelax_sap_score_mean"]], on="eval_pre_relax_description")

        # calculate eval composite score
        eval_comp_scoreterms = ["eval_af2_plddt", "eval_af2_catres_bb_rmsd", "eval_af2_catres_heavy_rmsd", "eval_af2_motif_rmsd", "eval_af2_contacts_score", "eval_af2_ligand_clashes", "eval_af2_postrelax_catres_heavy_rmsd_mean", "eval_af2_postrelax_ligand_rmsd_mean", "eval_af2_fastrelax_sap_score_mean", "eval_af2_fastrelax_total_score"]
        eval_comp_weights = [-1, 1, 4, 1, 1, 1, 4, 1, 2, 1]
        backbones.calculate_composite_score(
            name="eval_composite_score",
            scoreterms=eval_comp_scoreterms,
            weights=eval_comp_weights,
            plot=True
        )

        # update trajectory plots
        trajectory_plots = update_trajectory_plotting(trajectory_plots=trajectory_plots, poses=backbones, prefix="eval_af2")

        backbones.reindex_poses(prefix="eval_reindex", remove_layers=1 if not args.attnpacker_repack else 2, force_reindex=True)

        eval_results_dir = os.path.join(args.working_dir, f"{eval_prefix}evaluation_results")
        create_results_dir(poses=backbones, dir=eval_results_dir, score_col="eval_composite_score", plot_cols=eval_comp_scoreterms, rlx_path_col="eval_relaxed_combined_path", create_mutations_csv=True)
        for file in os.listdir(backbones.plots_dir):
            if file.startswith("trajectory") and file.endswith(".png"):
                shutil.copy(os.path.join(backbones.plots_dir, file), eval_results_dir)
        backbones.save_scores()

    ########################### VARIANT GENERATION ###########################

    if args.variants_input_json:

        if args.variants_prefix:
            variants_prefix = f"{args.variants_prefix}_"
        else: variants_prefix = ""

        backbones.set_work_dir(os.path.join(args.working_dir, f"{variants_prefix}variants"))

        eval_trajectory_plots = instantiate_trajectory_plotting(backbones.plots_dir, scores = ["postrelax_catres_heavy_rmsd_mean", "postrelax_ligand_rmsd_mean", "fastrelax_sap_score_mean"])
        eval_trajectory_plots = update_trajectory_plotting(trajectory_plots=eval_trajectory_plots, poses=backbones, prefix="eval_af2")

        trajectory_plots = instantiate_trajectory_plotting(backbones.plots_dir, scores = trajectory_scoreterms)
        trajectory_plots = update_trajectory_plotting(trajectory_plots=trajectory_plots, poses=backbones, prefix="cycle_final_esm")
        trajectory_plots = update_trajectory_plotting(trajectory_plots=trajectory_plots, poses=backbones, prefix="eval_af2")

        # drop all columns containing information about variant runs (to be able to create variants from previous ones)
        if len(var_cols := [col for col in backbones.df.columns if col.startswith("variants")]) > 0:
            backbones.df.drop(var_cols, axis=1, inplace=True)

        if args.variants_mutations_csv:
            mutations = pd.read_csv(args.variants_mutations_csv)
            mutations.replace({np.nan: None}, inplace=True)
            mutations.rename(columns={"omit_AAs": "variants_omit_AAs", "allow_AAs": "variants_allow_AAs"}, inplace=True)
            backbones.df = backbones.df.merge(mutations, on="poses_description", how="right")
            backbones.df.reset_index(drop=True, inplace=True)
            mutations_dir = os.path.join(backbones.work_dir, "mutations")
            os.makedirs(mutations_dir, exist_ok=True)
            backbones.df["variants_pose_opts"] = backbones.df.apply(lambda row: omit_AAs(row['variants_omit_AAs'], row['variants_allow_AAs'], mutations_dir, row["poses_description"]), axis=1)
        else:
            backbones.df["variants_omit_AAs"] = None
            backbones.df["variants_allow_AAs"] = None

        if args.variants_input_poses_per_bb:
            backbones.filter_poses_by_rank(n=args.variants_input_poses_per_bb, score_col="eval_composite_score", remove_layers=1)

        if args.variants_input_poses:
            backbones.filter_poses_by_rank(n=args.variants_input_poses, score_col="eval_composite_score")

        # create residue selections
        residue_cols = ["fixed_residues", "motif_residues", "template_motif", "template_fixedres", "ligand_motif"]
        for res_col in residue_cols:
            if not backbones.df[res_col].apply(lambda x: isinstance(x, ResidueSelection)).all():
                backbones.df[res_col] = [ResidueSelection(motif, from_scorefile=True) for motif in backbones.df[res_col].to_list()]

        # add covalent bonds info to poses pre-relax
        backbones = add_covalent_bonds_info(poses=backbones, prefix="variants_bbopt_cov_info", covalent_bonds_col="covalent_bonds")

        # create pose-specific options
        backbones.df['variants_bbopt_opts'] = [write_bbopt_opts(row=row, cycle=1, total_cycles=1, reference_location_col="updated_reference_frags_location", cat_res_col="fixed_residues", motif_res_col="motif_residues", ligand_chain="Z") for _, row in backbones.df.iterrows()]

        # optimize backbones (constrained)
        rosetta.run(
            poses = backbones,
            prefix = "variants_bbopt",
            rosetta_application="rosetta_scripts.default.linuxgccrelease",
            nstruct = 1,
            options = bb_opt_options,
            pose_options='variants_bbopt_opts'
        )

        if args.variants_run_cm:
            backbones.df["cm_resfile"] = backbones.df.apply(lambda row: create_mutation_resfiles(row['variants_omit_AAs'], row['variants_allow_AAs'], row['poses_description'], os.path.join(backbones.work_dir, "resfiles")), axis=1)
            cm_options =  f"-parser:protocol {os.path.abspath(os.path.join(args.riff_diff_dir, 'utils', 'coupled_moves.xml'))} -coupled_moves:ligand_mode true -coupled_moves:ligand_weight 2 -beta -ignore_zero_occupancy false -flip_HNQ true -ignore_waters false"
            if params:
                cm_options = cm_options + f" -extra_res_fa {' '.join(params)}"
            backbones.df["cm_pose_opts"] = backbones.df.apply(lambda row: write_cm_opts(row["fixed_residues"], row["motif_residues"], row["updated_reference_frags_location"], args.variants_cm_design_shell, row['cm_resfile']), axis=1)
            # coupled moves outputs in current directory --> change to another one, this is problematic when restarting runs
            original_dir = os.getcwd()
            cm_dir = os.path.join(backbones.work_dir, "coupled_moves")
            os.makedirs(cm_dir := os.path.join(backbones.work_dir, "coupled_moves"), exist_ok=True)
            os.chdir(cm_dir)
            pre_cm_poses = copy.deepcopy(backbones)

            if not os.path.isfile(os.path.abspath(os.path.join(cm_dir, "coupled_moves_rosetta_scores.json"))):
                rosetta.run(
                    poses=backbones,
                    prefix="coupled_moves",
                    rosetta_application="rosetta_scripts.default.linuxgccrelease",
                    nstruct=50,
                    options=cm_options,
                    pose_options="cm_pose_opts")

            os.chdir(original_dir)
            os.makedirs(cm_results_dir := os.path.join(backbones.work_dir, "cm_results"), exist_ok=True)
            backbones = pre_cm_poses

            backbones.df = create_coupled_moves_sequences(cm_results_dir, cm_dir, backbones.df, args.variants_cm_occurence_cutoff)

        else:
            if args.variants_activate_conservation_bias:
                # write distance conservation bias cmds
                shell_distances = [float(i) for i in args.variants_shell_distances.split(",")]
                shell_biases = [float(i) for i in args.variants_shell_biases.split(",")]
                protflow.tools.ligandmpnn.create_distance_conservation_bias_cmds(poses=backbones, prefix="conservation_bias", center="ligand_motif", shell_distances=shell_distances, shell_biases=shell_biases, jobstarter=small_cpu_jobstarter)
                ligandmpnn_options = ligandmpnn_options + f" --temperature {args.variants_bias_mpnn_temp}"
                # combine with previous pose opts:
                if args.variants_mutations_csv:
                    backbones.df["variants_pose_opts"] = backbones.df.apply(lambda row: (row['variants_pose_opts'] or '') + ' ' + (row['conservation_bias']),
        axis=1
    )
                else:
                    backbones.df["variants_pose_opts"] = backbones.df["conservation_bias"]

            # optimize sequences
            ligand_mpnn.run(
                poses = backbones,
                prefix = "variants_mpnn",
                nseq = args.variants_mpnn_sequences,
                model_type = "ligand_mpnn",
                options = ligandmpnn_options,
                pose_options = "variants_pose_opts" if args.variants_mutations_csv or args.variants_activate_conservation_bias else None,
                fixed_res_col = "fixed_residues"
            )

        # predict structures using ESMFold
        esmfold.run(
            poses = backbones,
            prefix = "variants_esm",
        )

        # TODO: no idea why, but residue columns seem to be lost when repeating run. hope this fixes it.
        # create residue selections
        residue_cols = ["fixed_residues", "motif_residues", "template_motif", "template_fixedres", "ligand_motif"]
        for res_col in residue_cols:
            if not backbones.df[res_col].apply(lambda x: isinstance(x, ResidueSelection)).all():
                backbones.df[res_col] = [ResidueSelection(motif, from_scorefile=True) for motif in backbones.df[res_col].to_list()]

        # calculate rmsds, TMscores and clashes
        catres_motif_heavy_rmsd.run(poses = backbones, prefix = f"variants_esm_catres_heavy")
        catres_motif_bb_rmsd.run(poses = backbones, prefix = f"variants_esm_catres_bb")
        bb_rmsd.run(poses = backbones, ref_col=f"variants_bbopt_location", prefix = f"variants_esm_backbone")
        fragment_motif_bb_rmsd.run(poses = backbones, prefix = "variants_esm_motif")
        tm_score_calculator.run(poses = backbones, prefix = f"variants_esm_tm", ref_col = f"variants_bbopt_location")

        backbones.filter_poses_by_value(score_col=f"variants_esm_plddt", value=args.ref_plddt_cutoff_end, operator=">=", prefix=f"variants_esm_plddt", plot=True)
        backbones.filter_poses_by_value(score_col=f"variants_esm_tm_TM_score_ref", value=0.9, operator=">=", prefix=f"variants_esm_TM_score", plot=True)
        backbones.filter_poses_by_value(score_col=f"variants_esm_catres_bb_rmsd", value=args.ref_catres_bb_rmsd_cutoff_end, operator="<=", prefix=f"variants_esm_catres_bb", plot=True)

        # add ligand to poses
        chain_adder.superimpose_add_chain(
            poses = backbones,
            prefix = f"variants_esm_ligand",
            ref_col = "updated_reference_frags_location",
            target_motif = "fixed_residues",
            copy_chain = "Z"
        )

        # calculate ligand clashes and ligand contacts
        ligand_clash.run(poses=backbones, prefix="variants_esm_ligand")
        ligand_contacts.run(poses=backbones, prefix="variants_esm_lig")
        backbones.df = calculate_contact_score(df=backbones.df, contact_col="variants_esm_lig_contacts", score_col="variants_esm_contacts_score", target_value=args.contacts_target_value)

        backbones.filter_poses_by_value(score_col="variants_esm_lig_contacts", value=args.min_contacts, operator=">=", prefix="variants_esm_lig_contacts", plot=True)

        # copy description column for merging with data later
        backbones.df[f'variants_post_esm_description'] = backbones.df['poses_description']

        # calculate multi-scoreterm score:
        logging.info("Calculating composite score for post-esm evaluation...")
        variants_esm_scoreterms = ["variants_esm_plddt", "variants_esm_catres_bb_rmsd", "variants_esm_catres_heavy_rmsd", "variants_esm_motif_rmsd", "variants_esm_contacts_score", "variants_esm_ligand_clashes"]
        variants_esm_weights = [-2, 4, 1, 1, 2, 4]
        backbones.calculate_composite_score(
            name="variants_esm_composite_score",
            scoreterms=variants_esm_scoreterms,
            weights=variants_esm_weights,
            plot=True
        )

        # update trajectory plots
        trajectory_plots = update_trajectory_plotting(trajectory_plots=trajectory_plots, poses=backbones, prefix="variants_esm")

        # define number of index layers that were added during variants generation
        layers = 2

        # manage screen output
        backbones.reindex_poses(prefix=f"variants_esm_reindex", remove_layers=layers, force_reindex=True)

        # filter down to rfdiffusion backbones
        backbones.filter_poses_by_rank(
            n=args.variants_evaluation_input_poses_per_bb,
            score_col="variants_esm_composite_score",
            prefix="variants_esm_composite_score_per_bb",
            plot=True,
            remove_layers=3
        )

        backbones.filter_poses_by_rank(
            n=args.variants_evaluation_input_poses,
            score_col="variants_esm_composite_score",
            prefix="variants_esm_composite_score",
            plot=True,
            remove_layers=None
        )

        backbones.convert_pdb_to_fasta(prefix="variants_fasta_conversion", update_poses=True)

        # set for easier merging later
        backbones.df["variants_pre_af2_description"] = backbones.df["poses_description"]

        # run af2
        colabfold.run(
            poses=backbones,
            prefix="variants_af2",
            return_top_n_poses=1,
            options="--msa-mode single_sequence"
        )

        # calculate backbone rmsds
        catres_motif_bb_rmsd.run(poses=backbones, prefix=f"variants_af2_catres_bb")
        bb_rmsd.run(poses=backbones, prefix="variants_af2_backbone", ref_col="variants_bbopt_location")
        bb_rmsd.run(poses=backbones, prefix="variants_af2_ESM_bb", ref_col="variants_esm_location")
        tm_score_calculator.run(poses=backbones, prefix="variants_af2_tm", ref_col="variants_bbopt_location")
        tm_score_calculator.run(poses=backbones, prefix="variants_af2_ESM_tm", ref_col="variants_esm_location")
        fragment_motif_bb_rmsd.run(poses = backbones, prefix = "variants_af2_motif")

        # calculate average plddt
        backbones = calculate_mean_scores(backbones, scores=["variants_af2_plddt", "variants_af2_catres_bb_rmsd", "variants_af2_backbone_rmsd", "variants_af2_ESM_bb_rmsd", "variants_af2_tm_TM_score_ref", "variants_af2_ESM_tm_TM_score_ref", "variants_af2_motif_rmsd"], remove_layers=1)

        # filter for af2 top model
        backbones.filter_poses_by_rank(n=1, score_col="variants_af2_plddt", ascending=False, remove_layers=1)

        # apply rest of the filters
        backbones.filter_poses_by_value(score_col="variants_af2_plddt", value=args.eval_plddt_cutoff, operator=">=", prefix="variants_af2_plddt", plot=True)
        backbones.filter_poses_by_value(score_col="variants_af2_tm_TM_score_ref", value=0.9, operator=">=", prefix=f"variants_af2_TM_score", plot=True)
        backbones.filter_poses_by_value(score_col="variants_af2_catres_bb_rmsd", value=args.eval_catres_bb_rmsd_cutoff, operator="<=", prefix=f"variants_af2_catres_bb_rmsd", plot=True)

        # repack with attnpacker
        if args.attnpacker_repack:
            attnpacker.run(
                poses=backbones,
                prefix=f"eval_af2_packing"
            )

        # calculate sc rmsd
        catres_motif_heavy_rmsd.run(poses=backbones, prefix=f"variants_af2_catres_heavy")

        # add ligand chain
        chain_adder.superimpose_add_chain(
            poses = backbones,
            prefix = f"variants_af2_ligand",
            ref_col = "updated_reference_frags_location",
            target_motif = "fixed_residues",
            copy_chain = "Z"
        )

        # calculate ligand clashes and ligand contacts
        ligand_clash.run(poses=backbones, prefix="variants_af2_ligand")
        ligand_contacts.run(poses=backbones, prefix="variants_af2_lig")
        backbones.df = calculate_contact_score(df=backbones.df, contact_col="variants_af2_lig_contacts", score_col="variants_af2_contacts_score", target_value=args.contacts_target_value)

        # filter for ligand contacts
        backbones.filter_poses_by_value(score_col="variants_af2_lig_contacts", value=args.min_contacts, operator=">=", prefix="variants_af2_lig_contacts", plot=True)

        # add covalent bonds info
        backbones = add_covalent_bonds_info(poses=backbones, prefix="variants_af2_cov_info", covalent_bonds_col="covalent_bonds")

        # create identifier for easier merging
        backbones.df["variants_af2_pre_relax_description"] = backbones.df["poses_description"]
        rlx_poses = copy.deepcopy(backbones)

        # relax predictions with ligand present
        rosetta.run(
            poses = rlx_poses,
            prefix = "variants_af2_fastrelax",
            rosetta_application="rosetta_scripts.default.linuxgccrelease",
            nstruct = 15,
            options = fr_options
        )

        # calculate RMSDs of relaxed poses
        catres_motif_heavy_rmsd.run(poses = rlx_poses, prefix = "variants_af2_postrelax_catres_heavy")
        catres_motif_bb_rmsd.run(poses = rlx_poses, prefix = "variants_af2_postrelax_catres_bb")
        ligand_rmsd.run(poses = rlx_poses, prefix = "variants_af2_postrelax_ligand")

        # calculate average scores, filter
        rlx_poses = calculate_mean_scores(rlx_poses, scores=["variants_af2_postrelax_catres_heavy_rmsd", "variants_af2_postrelax_catres_bb_rmsd", "variants_af2_postrelax_ligand_rmsd", "variants_af2_fastrelax_total_score", "variants_af2_fastrelax_sap_score"], remove_layers=1)

        # combine relaxed poses into a single multimodel pdb
        combined = combine_relax_output(rlx_poses.df, "poses", os.path.join(rlx_poses.work_dir, "relaxed_combined"), "variants_relaxed_combined_path", "variants_af2_pre_relax_description", "variants_af2_fastrelax_total_score", 1)
        rlx_poses.df = rlx_poses.df.merge(combined, on="variants_af2_pre_relax_description", how="left")

        # filter for top structure
        rlx_poses.filter_poses_by_rank(n=1, score_col="variants_af2_fastrelax_total_score", ascending=True, remove_layers=1)

        # merge with original poses
        backbones.df = backbones.df.merge(rlx_poses.df[["variants_af2_pre_relax_description", "variants_af2_postrelax_catres_heavy_rmsd_mean", "variants_af2_postrelax_catres_bb_rmsd_mean", "variants_af2_postrelax_ligand_rmsd_mean", "variants_af2_fastrelax_total_score", "variants_af2_fastrelax_sap_score_mean", "variants_relaxed_combined_path"]], on="variants_af2_pre_relax_description")

        # calculate variants_af2 composite score
        variants_af2_scoreterms = ["variants_af2_plddt", "variants_af2_catres_bb_rmsd", "variants_af2_catres_heavy_rmsd", "variants_af2_motif_rmsd", "variants_af2_contacts_score", "variants_af2_ligand_clashes", "variants_af2_postrelax_catres_heavy_rmsd_mean", "variants_af2_postrelax_ligand_rmsd_mean", "variants_af2_fastrelax_sap_score_mean", "variants_af2_fastrelax_total_score"]
        variants_af2_comp_weights = [-1, 1, 4, 1, 1, 1, 4, 1, 2, 1]
        backbones.calculate_composite_score(
            name="variants_af2_composite_score",
            scoreterms=variants_af2_scoreterms,
            weights=variants_af2_comp_weights,
            plot=True
        )

        # update trajectory plots
        trajectory_plots = update_trajectory_plotting(trajectory_plots=trajectory_plots, poses=backbones, prefix="variants_af2")
        eval_trajectory_plots = update_trajectory_plotting(trajectory_plots=eval_trajectory_plots, poses=backbones, prefix="variants_af2")

        backbones.reindex_poses(prefix="variants_af2_reindex", remove_layers=2 if not args.attnpacker_repack else 3, force_reindex=True)

        # filter for unique diffusion backbones
        backbones.filter_poses_by_rank(n=5, score_col="variants_af2_composite_score", remove_layers=3, plot=True)

        # create output directory
        create_results_dir(poses=backbones, dir=os.path.join(args.working_dir, f"{variants_prefix}variants_results"), score_col="variants_af2_composite_score", plot_cols=variants_af2_scoreterms, rlx_path_col="variants_relaxed_combined_path", create_mutations_csv=True)
        backbones.save_scores()



if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--riff_diff_dir", type=str, default=".", help="Directory that contains the Riff-Diff repository.")
    argparser.add_argument("--working_dir", type=str, required=True, help="output directory.")

    # general optionals
    argparser.add_argument("--skip_refinement", action="store_true", help="Skip refinement and evaluation, only run screening.")
    argparser.add_argument("--skip_evaluation", action="store_true", help="Skip evaluation, only run screening and refinement.")
    argparser.add_argument("--params_files", type=str, default=None, help="Path to alternative params file. Can also be multiple paths separated by ','.")
    argparser.add_argument("--attnpacker_repack", action="store_true", help="Run attnpacker and af2 predictions")
    argparser.add_argument("--use_reduced_motif", action="store_true", help="Instead of using the full fragments during backbone optimization, just use residues directly adjacent to fixed_residues. Also affects motif_bb_rmsd etc.")

    # jobstarter
    argparser.add_argument("--prefer_cpu", action="store_true", help="Use CPUs instead of GPUs, where possible (ESMFold will only work with GPU).")
    argparser.add_argument("--max_gpus", type=int, default=10, help="How many GPUs do you want to use at once?")
    argparser.add_argument("--max_cpus", type=int, default=1000, help="How many CPUs do you want to use at once?")

    # screening
    argparser.add_argument("--screen_input_json", type=str, default=None, help="Read in a poses json file containing input poses for screening (e.g. the selected_paths.json from motif generation or the successful_input_motifs.json from a previous screening run).")
    argparser.add_argument("--screen_decentralize_weights", type=str, nargs="+", default=[30], help="Decentralize weights that should be tested during screening.")
    argparser.add_argument("--screen_decentralize_distances", type=str, nargs="+", default=[2], help="Decentralize distances that should be tested during screening.")
    argparser.add_argument("--screen_input_poses", type=int, default=200, help="Number of input poses for screening. Poses will be selected according to <screen_input_selection>.")
    argparser.add_argument("--screen_input_selection", default="weighted", help="Can be either 'top' (default), 'random' or 'weighted'. Defines if motif library input poses are chosen based on score, at random or random weighted by score.")
    argparser.add_argument("--screen_num_rfdiffusions", type=int, default=5, help="Number of backbones to generate per input path during screening.")
    argparser.add_argument("--screen_skip_mpnn_rlx_mpnn", action="store_true", help="Skip LigandMPNN-RELAX-LigandMPNN steps and just run LigandMPNN once before prediction with ESMFold. Faster, but lower success rates (only recommended for initial testing purposes).")
    argparser.add_argument("--screen_num_mpnn_sequences", type=int, default=30, help="Number of LigandMPNN sequences per backbone that should be generated and predicted with ESMFold post-RFdiffusion.")
    argparser.add_argument("--screen_num_seq_thread_sequences", type=int, default=3, help="Number of LigandMPNN sequences that should be generated during the sequence threading phase (input for backbone optimization). Only used if <screen_skip_mpnn_rlx_mpnn> is not set.")
    argparser.add_argument("--screen_prefix", type=str, default=None, help="Prefix for screening runs for testing different settings. Will be reused for subsequent steps if not specified otherwise.")
    argparser.add_argument("--screen_esm_input_poses", type=int, default=5000, help="Maximum total number of poses that should be predicted with ESMFold.")

    # refinement optionals
    argparser.add_argument("--ref_input_json", type=str, default=None, help="Read in a poses json file containing input poses for refinement (e.g. screening_results_all.json in the screening results directory). Screening will be skipped.")
    argparser.add_argument("--ref_prefix", type=str, default=None, help="Prefix for refinement runs for testing different settings.")
    argparser.add_argument("--ref_cycles", type=int, default=5, help="Number of Rosetta-MPNN-ESM refinement cycles.")
    argparser.add_argument("--ref_input_poses_per_bb", default=None, help="Filter the number of refinement input poses on an input-backbone level. This filter is applied before the ref_input_poses filter.")
    argparser.add_argument("--ref_input_poses", type=int, default=100, help="Maximum number of input poses for refinement cycles after initial RFDiffusion-MPNN-ESM-Rosetta run. Poses will be filtered by screen_composite_score.")
    argparser.add_argument("--ref_num_mpnn_seqs", type=int, default=25, help="Number of sequences that should be created per pose with LigandMPNN during refinement.")
    argparser.add_argument("--ref_catres_bb_rmsd_cutoff_start", type=float, default=1.2, help="Start value for catalytic residue backbone rmsd filter after each refinement cycle. Filter will be ramped from start to end during refinement.")
    argparser.add_argument("--ref_catres_bb_rmsd_cutoff_end", type=float, default=0.7, help="End value for catalytic residue backbone rmsd filter after each refinement cycle. Filter will be ramped from start to end during refinement.")
    argparser.add_argument("--ref_motif_rmsd_cutoff_start", type=float, default=1.5, help="Start value for motif backbone rmsd filter after each refinement cycle. Filter will be ramped from start to end during refinement.")
    argparser.add_argument("--ref_motif_rmsd_cutoff_end", type=float, default=1.0, help="End value for motif backbone rmsd filter after each refinement cycle. Filter will be ramped from start to end during refinement.")
    argparser.add_argument("--ref_plddt_cutoff_end", type=float, default=85, help="End value for ESMFold plddt filter after each refinement cycle. Filter will be ramped from start to end during refinement.")
    argparser.add_argument("--ref_plddt_cutoff_start", type=float, default=75, help="Start value for ESMFold plddt filter after each refinement cycle. Filter will be ramped from start to end during refinement.")
    argparser.add_argument("--ref_num_cycle_poses", type=int, default=3, help="Number of poses per unique diffusion backbone that should be passed on to the next refinement cycle.")
    argparser.add_argument("--ref_seq_thread_num_mpnn_seqs", type=float, default=3, help="Number of LigandMPNN output sequences during the initial, sequence-threading phase (pre-relax).")
    argparser.add_argument("--ref_start_cycle", type=int, default=1, help="Number from which to start cycles. Useful if adding additional refinement cycles after a run has completed.")
    argparser.add_argument("--ref_skip_all_cycle_pooling", action="store_true", help="Do not combine results of all refinement results. Instead, just use the last cycle.")

    # evaluation
    argparser.add_argument("--eval_prefix", type=str, default=None, help="Prefix for evaluation runs for testing different settings or refinement outputs.")
    argparser.add_argument("--eval_input_json", type=str, default=None, help="Read in a poses json containing input poses for evaluation.")
    argparser.add_argument("--eval_input_poses", type=int, default=500, help="Maximum number of input poses for evaluation with AF2 after refinement. Poses will be filtered by cycle_final_refinement_composite_score.")
    argparser.add_argument("--eval_input_poses_per_bb", type=int, default=30, help="Maximum number of input poses per unique diffusion backbone for evaluation with AF2 after refinement. Poses will be filtered by cycle_final_refinement_composite_score")
    argparser.add_argument("--eval_plddt_cutoff", type=float, default=85, help="Cutoff for plddt for the AF2 top model for each pose.")
    argparser.add_argument("--eval_catres_bb_rmsd_cutoff", type=float, default=0.7, help="Cutoff for catalytic residue backbone rmsd for the AF2 top model for each pose.")

    # variant generation
    argparser.add_argument("--variants_prefix", type=str, default=None, help="Prefix for variant generation runs for testing different variants.")
    argparser.add_argument("--variants_input_json", type=str, default=None, help="Read in a poses json containing poses from evaluation output.")
    argparser.add_argument("--variants_mutations_csv", type=str, default=None, help="Read in a custom csv containing poses description and mutation columns (as found in the evaluation_results directory).")
    argparser.add_argument("--variants_input_poses_per_bb", type=int, default=10, help="Number of poses per unique backbone that variant generation should be performed on.")
    argparser.add_argument("--variants_input_poses", type=int, default=50, help="Number of poses that variant generation should be performed on.")
    argparser.add_argument("--variants_evaluation_input_poses_per_bb", type=int, default=50, help="Number of poses per unique backbone that should go into the evaluation step of variant generation.")
    argparser.add_argument("--variants_evaluation_input_poses", type=int, default=400, help="Number of poses per unique backbone that should go into the evaluation step of variant generation.")
    argparser.add_argument("--variants_activate_conservation_bias", action="store_true", help="Add a bias for conservation of residues based on distance from ligand to LigandMPNN runs (to sample around ligand more efficiently)")
    argparser.add_argument("--variants_shell_distances", type=str, default="10,15,20,1000", help="Shell distances from the ligand. Only active if variants_activate_conservation_bias is set.")
    argparser.add_argument("--variants_shell_biases", type=str, default="0,0.25,0.5,1", help="Conservation bias strength for each shell. 0 means no bias. Only active if variants_activate_conservation_bias is set.")
    argparser.add_argument("--variants_bias_mpnn_temp", type=float, default=0.3, help="LigandMPNN temperature for conservation bias MPNN runs. Higher means more sequence diversity. Only active if variants_activate_conservation_bias is set.")
    argparser.add_argument("--variants_mpnn_sequences", type=int, default=50, help="Number of sequences that will be generated for each input structure. All of these will be predicted with ESMFold.")
    argparser.add_argument("--variants_run_cm", action="store_true", help="Run coupled moves protocol instead of LigandMPNN for active site optimization")
    argparser.add_argument("--variants_cm_design_shell", type=str, default="6,8,10,12", help="Design shells for coupled moves protocol.")
    argparser.add_argument("--variants_cm_occurence_cutoff", type=int, default=0.25, help="Cutoff for coupled moves mutations to be accepted.")
    argparser.add_argument("--variants_drop_previous_results", action="store_true", help="Drop all variants columns from poses dataframe (useful if running variant generation again on output of variant generation, e.g. to test new mutations)")


    # rfdiffusion optionals
    argparser.add_argument("--recenter", type=str, default=None, help="Point (xyz) in input pdb towards the diffusion should be recentered. example: --recenter=-13.123;34.84;2.3209")
    argparser.add_argument("--flanking", type=str, default="split", help="How flanking should be set. nterm or cterm also valid options.")
    argparser.add_argument("--flanker_length", type=int, default=30, help="Set Length of flanking regions.")
    argparser.add_argument("--total_length", type=int, default=200, help="Total length of protein to diffuse. This includes flanker, linkers and input fragments.")

    # ligandmpnn optionals
    argparser.add_argument("--ligandmpnn_options", type=str, default=None, help="Options for ligandmpnn runs.")

    # filtering options
    argparser.add_argument("--rfdiffusion_max_clashes", type=int, default=None, help="Filter rfdiffusion output for ligand-backbone clashes before passing poses to LigandMPNN.")
    argparser.add_argument("--rfdiffusion_max_rog", type=float, default=18, help="Filter rfdiffusion output for radius of gyration before passing poses to LigandMPNN.")
    argparser.add_argument("--ligand_clash_factor", type=float, default=0.9, help="Factor for determining clashes. Set to 0 if ligand clashes should be ignored.")
    argparser.add_argument("--rfdiffusion_catres_bb_rmsd", type=float, default=1, help="Filter RFdiffusion output for catalytic residue backbone rmsd.")
    argparser.add_argument("--rfdiffusion_motif_bb_rmsd", type=float, default=1, help="Filter RFdiffusion output for fragment motif backbone rmsd.")
    argparser.add_argument("--rfdiffusion_options", type=str, default="", help="Additional options for RFdiffusion runs.")
    argparser.add_argument("--min_contacts", type=int, default=5, help="Mininum number of backbone atoms within 8 A per ligand heavy atom after prediction.")
    argparser.add_argument("--contacts_target_value", type=int, default=9, help="Target number of backbone atoms within 8 A per ligand heavy atom after prediction.")

    arguments = argparser.parse_args()


    main(arguments)
