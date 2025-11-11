import logging
import os
import sys
import copy
import json
import string
import itertools
import time
import math
from collections import Counter


# import dependencies
import pandas as pd
import numpy as np
import Bio
import Bio.PDB
from Bio.PDB import Structure, Model, Chain, Residue, Atom
import matplotlib.pyplot as plt

# protflow
from protflow.utils.biopython_tools import load_structure_from_pdbfile, save_structure_to_pdbfile
from protflow.utils.utils import vdw_radii
from protflow.jobstarters import SbatchArrayJobstarter, LocalJobStarter
from protflow.config import PROTFLOW_ENV
from protflow.utils.plotting import violinplot_multiple_cols
from protflow.utils.openbabel_tools import openbabel_fileconverter
from protflow.poses import description_from_path
from protflow.residues import ResidueSelection, from_dict


def split_pdb_numbering(pdbnum):
    '''
    splits residue-chain (or chain-residue) into separate entities
    '''
    resnum = ""
    chain = ""
    for char in pdbnum:
        if char.isdigit():
            resnum += char
        else:
            chain += char
    resnum = int(resnum)
    if not chain:
        chain = "A"
    return [resnum, chain]

def tip_symmetric_residues():
    '''
    3-letter code for residues with symmetric functional groups
    '''
    symres = ["ARG", "ASP", "GLU", "LEU", "PHE", "TYR", "VAL"]
    return symres

def return_residue_rotamer_library(library_folder:str, residue_identity:str):
    '''
    finds the correct library for a given amino acid and drops not needed chi angles
    '''
    prefix = residue_identity.lower()
    rotlib = pd.read_csv(os.path.join(library_folder, f'{prefix}.bbdep.rotamers.lib'))
    if residue_identity in aa_up_to_chi3():
        rotlib.drop(['chi4', 'chi4sig'], axis=1, inplace=True)
    elif residue_identity in aa_up_to_chi2():
        rotlib.drop(['chi3', 'chi3sig', 'chi4', 'chi4sig'], axis=1, inplace=True)
    elif residue_identity in aa_up_to_chi1():
        rotlib.drop(['chi2', 'chi2sig', 'chi3', 'chi3sig', 'chi4', 'chi4sig'], axis=1, inplace=True)

    return rotlib

def aa_up_to_chi1():
    '''
    tuple of amino acids with a single chi angle
    '''
    return ('CYS', 'SER', 'THR', 'VAL')

def aa_up_to_chi2():
    '''
    tuple of amino acids with two chi angles
    '''
    return ('ASP', 'ASN', 'HIS', 'ILE', 'LEU', 'PHE', 'PRO', 'TRP', 'TYR')

def aa_up_to_chi3():
    '''
    tuple of amino acids with three chi angles
    '''
    return ('GLN', 'GLU', 'MET')

def aa_up_to_chi4():
    '''
    tuple of amino acids with four chi angles
    '''
    return ('ARG', 'LYS')

def num_chis_for_residue_id(res_id):
    '''
    identify number of chi angles for residue
    '''
    if res_id in aa_up_to_chi4():
        return 4
    if res_id in aa_up_to_chi3():
        return 3
    elif res_id in aa_up_to_chi2():
        return 2
    elif res_id in aa_up_to_chi1():
        return 1
    else:
        return 0


def rama_plot(df, x_col, y_col, color_col, size_col, save_path=None):
    '''
    Ramachandran plot
    '''
    df_list = []
    for _, df in df.groupby([x_col, y_col]):
        top = df.sort_values(color_col, ascending=False).head(1)
        df_list.append(top)
    df = pd.concat(df_list)
    df = df[df[size_col] > 0]
    df[size_col] = df[size_col] * 100
    fig, ax = plt.subplots()
    norm_color = plt.Normalize(0, df[color_col].max())
    cmap = plt.cm.Blues # pylint: disable=no-member
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_color)
    sm.set_array([])
    #norm_size = plt.Normalize(0, df[size_col].max())
    ax.scatter(df[x_col], df[y_col], c=df[color_col], cmap=cmap, s=df[size_col], norm=norm_color)
    fig.colorbar(sm, label="probability", ax=ax)
    ax.set_xlabel("phi [degrees]")
    ax.set_ylabel("psi [degrees]")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-180, 181, 60))
    fig.gca().set_aspect('equal', adjustable='box')
    if save_path:
        plt.savefig(save_path, dpi=300)

def identify_backbone_angles_suitable_for_rotamer(residue_identity:str, rotlib:pd.DataFrame, output_dir: str, output_prefix:str=None, limit_sec_struct:str=None, occurrence_cutoff:int=5, max_output:int=None, rotamer_diff_to_best=0.05, rotamer_chi_binsize=None, rotamer_phipsi_binsize=None, prob_cutoff=None):
    '''
    finds phi/psi angles most common for a given set of chi angles from a rotamer library
    chiX_bin multiplies the chiXsigma that is used to check if a rotamer fits to the given set of chi angles --> increasing leads to higher number of hits, decreasing leads to rotamers that more closely resemble input chis. Default=1
    if score_cutoff is provided, returns only phi/psi angles above score_cutoff
    if fraction is provided, returns only the top rotamer fraction ranked by score
    if max_output is provided, returns only max_output phi/psi combinations
    '''

    filename = os.path.join(output_dir, output_prefix + f'{residue_identity}_rama_pre_filtering')

    rama_plot(rotlib, 'phi', 'psi', 'probability', 'phi_psi_occurrence', filename)

    if limit_sec_struct:
        rotlib = filter_rotamers_by_sec_struct(rotlib, limit_sec_struct)

    if occurrence_cutoff:
        rotlib = rotlib.loc[rotlib['phi_psi_occurrence'] > occurrence_cutoff / 100]
        if rotlib.empty:
            log_and_print(f"No rotamers passed occurrence cutoff of {occurrence_cutoff}")
            logging.error(f"No rotamers passed occurrence cutoff of {occurrence_cutoff}")

    if prob_cutoff:
        rotlib = rotlib[rotlib['probability'] >= prob_cutoff]
        if rotlib.empty:
            log_and_print(f"No rotamers passed probability cutoff of {prob_cutoff}")
            logging.error(f"No rotamers passed probability cutoff of {prob_cutoff}")

    rotlib = rotlib.sort_values('rotamer_score', ascending=False)

    rotlib = filter_rotlib_for_rotamer_diversity(rotlib, rotamer_chi_binsize, rotamer_phipsi_binsize)

    if rotamer_diff_to_best:
        rotlib = rotlib[rotlib['probability'] >= rotlib['probability'].max() * (1 - rotamer_diff_to_best)]

    rotlib = rotlib.sort_values('rotamer_score', ascending=False)

    if max_output:
        rotlib = rotlib.head(max_output)

    if rotlib.empty:
        raise RuntimeError('Could not find any rotamers that fit. Try setting different filter values!')

    rotlib.reset_index(drop=True, inplace=True)

    filename = os.path.join(output_dir, output_prefix + f'{residue_identity}_rama_post_filtering')
    rama_plot(rotlib, 'phi', 'psi', 'probability', 'phi_psi_occurrence', filename)

    return rotlib

def angle_difference(angle1, angle2):
    '''
    calculate difference between angles (considers negative values and +360)
    '''
    return min([abs(angle1 - angle2), abs(angle1 - angle2 + 360), abs(angle1 - angle2 - 360)])

def filter_rotlib_for_rotamer_diversity(rotlib:pd.DataFrame, rotamer_chi_binsize:float=None, rotamer_phipsi_binsize:float=None):
    '''
    filters rotamer library for more diversity in rotamers
    '''
    accepted_rotamers = []
    chi_columns = [column for column in rotlib.columns if column.startswith('chi') and not column.endswith('sig')]

    for _, row in rotlib.iterrows():
        rotamer_accept_list = []
        if len(accepted_rotamers) == 0:
            accepted_rotamers.append(row)
            continue
        for accepted_rot in accepted_rotamers:
            angle_accept_list = []
            if rotamer_phipsi_binsize: # check for backbone angle difference
                phi_difference = angle_difference(row['phi'], accepted_rot['phi'])
                psi_difference = angle_difference(row['psi'], accepted_rot['psi'])
                if sum([phi_difference, psi_difference]) >= rotamer_phipsi_binsize:
                    angle_accept_list.append(True)
                else:
                    angle_accept_list.append(False)
            if rotamer_chi_binsize:
                for column in chi_columns:
                    # only accept rotamers that are different from already accepted ones
                    if angle_difference(row[column], accepted_rot[column]) >= rotamer_chi_binsize:
                        angle_accept_list.append(True)
                    else:
                        # if chi angles are similar to accepted one, set False
                        angle_accept_list.append(False)
            if not rotamer_chi_binsize and not rotamer_phipsi_binsize: # set true if no filter was set
                angle_accept_list.append(True)
            if True in angle_accept_list:
                rotamer_accept_list.append(True) # if any angle was different enough, accept it
            else:
                rotamer_accept_list.append(False) # if no angle was different enough, discard it
        if set(rotamer_accept_list) == {True}: # check if difference to all accepted rotamers was ok
            accepted_rotamers.append(row)
    rotlib = pd.DataFrame(accepted_rotamers)
    return rotlib


def filter_rotamers_by_sec_struct(rotlib:pd.DataFrame, secondary_structure:str):
    '''
    filters rotamers according to secondary structure (via phi/psi angles that are typical for a given secondary structure)
    '''
    filtered_list = []
    sec_structs = [*secondary_structure]
    #phi and psi angle range was determined from fragment library
    if "-" in sec_structs:
        phi_range = list(range(-170, -39, 10)) + list(range(60, 81, 10))
        psi_range = list(range(-180, -159, 10)) + list(range(-40, 181, 10))
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(filtered)
    if "B" in sec_structs:
        phi_range = list(range(-170, -49, 10))
        psi_range = list(range(-180, -169, 10)) + list(range(80, 181, 10))
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(filtered)
    if "E" in sec_structs:
        phi_range = list(range(-170, -59, 10))
        psi_range = list(range(-180, -169, 10)) + list(range(90, 181, 10))
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(filtered)
    if "G" in sec_structs:
        phi_range = list(range(-130, -39, 10)) + list(range(50, 71, 10))
        psi_range = list(range(-50, 41, 10))
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(copy.deepcopy(filtered))
    if "H" in sec_structs:
        phi_range = list(range(-100, -39, 10))
        psi_range = list(range(-60, 1, 10))
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(filtered)
    if "I" in sec_structs:
        phi_range = list(range(-140, -49, 10))
        psi_range = list(range(-80, 1, 10))
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(filtered)
    if "S" in sec_structs:
        phi_range = list(range(-170, -49, 10)) + list(range(50, 111, 10))
        psi_range = list(range(-180, -149, 10)) + list(range(-60, 181, 10))
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(filtered)
    if "T" in sec_structs:
        phi_range = list(range(-130, -40, 10)) + list(range(40, 111, 10))
        psi_range = list(range(-60, 61, 10)) + list(range(120, 151, 10))
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(filtered)
    rotlib = pd.concat(filtered_list)
    if rotlib.empty:
        log_and_print(f"No rotamers passed secondary structure filtering for secondary structure {secondary_structure}.")
        logging.error(f"No rotamers passed secondary structure filtering for secondary structure {secondary_structure}.")
    return rotlib

def import_fragment_library(library_path:str):
    '''
    reads in a fragment library
    '''
    library = pd.read_pickle(library_path)
    #library.drop(library.columns[[0]], axis=1, inplace=True)
    return library


def check_for_chainbreaks(df, columname, fragsize):
    '''
    returns true if dataframe column is consistently numbered
    '''
    if df[columname].diff().sum() + 1 == fragsize:
        return True
    return False

def filter_frags_df_by_secondary_structure_content(frags_df, frag_sec_struct_fraction):
    '''
    filter fragment dataframe to specified fraction of secondary structure content
    '''
    frags_df_list = []
    for _, df in frags_df.groupby('frag_num', sort=False):
        for sec_struct, fraction in frag_sec_struct_fraction.items():
            if df['ss'].str.contains(sec_struct, regex=False).sum() / len(df.index) >= fraction:
                frags_df_list.append(df)
                break
    if len(frags_df_list) > 0:
        frags_df = pd.concat(frags_df_list)
        return frags_df
    return pd.DataFrame()

def filter_frags_df_by_score(frags_df, score_cutoff, scoretype, mode):
    '''
    filter fragment dataframe according to scoretype and score_cutoff, supports different filtering modes
    '''
    frags_df_list = []
    for _, df in frags_df.groupby('frag_num', sort=False):
        #only accepts fragments where mean value is above threshold
        if mode == 'mean_max_cutoff':
            if df[scoretype].mean() < score_cutoff:
                frags_df_list.append(df)
        if mode == 'mean_min_cutoff':
            if df[scoretype].mean() > score_cutoff:
                frags_df_list.append(df)
        #only accepts fragments if none of the residues is above threshold
        elif mode == 'max_cutoff':
            if df[scoretype].max() < score_cutoff:
                frags_df_list.append(df)
        elif mode == 'min_cutoff':
            if df[scoretype].min() > score_cutoff:
                frags_df_list.append(df)

    if len(frags_df_list) > 0:
        frags_df = pd.concat(frags_df_list)
        return frags_df
    return pd.DataFrame()


def check_fragment(frag, check_dict, frag_df, ligand, channel, rotamer_position, rmsd_cutoff, bb_lig_clash_vdw_multiplier, rot_lig_clash_vdw_multiplier, channel_frag_clash_vdw_multiplier):
    '''
    checks fragment for clashes and RMSD to all other accepted fragments
    '''
    frag_bb_atoms = [atom for atom in frag.get_atoms() if atom.id in ["N", "CA", "C", "O"]]
    ligand_atoms = [atom for atom in ligand.get_atoms() if not atom.element == "H"]
    # select all sidechain heavy atoms, but ignore functional groups
    frag_sc_atoms = [atom for atom in frag[rotamer_position].get_atoms() if not atom.element == "H" and not atom.id in atoms_of_functional_groups() and not atom.id in ["N", "CA", "C", "O"]]

    # check for channel clashes
    if channel:
        channel_bb_atoms = [atom for atom in channel.get_atoms() if atom.id in ["N", "CA", "C", "O"]]
        channel_clashing_atoms = clash_detection(frag_bb_atoms, channel_bb_atoms, channel_frag_clash_vdw_multiplier)
        if channel_clashing_atoms:
            check_dict["channel_clashes"] += 1
            return check_dict

    # check for fragment-bb ligand clashes
    frag_ligand_clashes = clash_detection(frag_bb_atoms, ligand_atoms, bb_lig_clash_vdw_multiplier)
    if frag_ligand_clashes:
        check_dict["bb_clashes"] += 1
        return check_dict

    # check for rotamer ligand clashes
    sc_ligand_clashes = clash_detection(frag_sc_atoms, ligand_atoms, rot_lig_clash_vdw_multiplier)
    if sc_ligand_clashes:
        clash_ids = [f"{clash[0].get_parent().get_resname()} {clash[0].id} - {clash[1].get_parent().get_resname()} {clash[1].id}" for clash in sc_ligand_clashes]
        check_dict["sc_clashes"] = check_dict["sc_clashes"] + clash_ids
        return check_dict

    # check rmsd to all other fragments
    if len(check_dict["selected_frags"]) == 0:
        check_dict["selected_frags"].append(frag)
        check_dict["selected_frag_dfs"].append(frag_df)
    else:
        rmsdlist = [calculate_rmsd_bb(picked_frag, frag) for picked_frag in check_dict["selected_frags"]]
        if min(rmsdlist) >= rmsd_cutoff:
            check_dict["selected_frags"].append(frag)
            check_dict["selected_frag_dfs"].append(frag_df)
        else:
            check_dict["rmsd_fails"] += 1

    return check_dict



def clash_detection(entity1, entity2, vdw_multiplier):
    '''
    checks for clashes by comparing VanderWaals radii. If clashes with ligand should be detected, set ligand to true. Ligand chain must be added as second entity.
    bb_only: only detect backbone clashes between to proteins or a protein and a ligand.
    clash_detection_vdw_multiplier: multiply Van der Waals radii with this value to set clash detection limits higher/lower
    database: path to database directory
    '''

    entity1_coords = np.array([atom.get_coord() for atom in entity1])
    entity2_coords = np.array([atom.get_coord() for atom in entity2])


    entity1_vdw = np.array([vdw_radii()[atom.element.lower()] for atom in entity1])
    entity2_vdw = np.array([vdw_radii()[atom.element.lower()] for atom in entity2])

    if np.any(np.isnan(entity1_vdw)) or np.any(np.isnan(entity2_vdw)):
        raise RuntimeError("Could not find Van der Waals radii for all elements in ligand. Check protflow.utils.vdw_radii and add it, if applicable!")

    # Compute pairwise distances using broadcasting
    dgram = np.linalg.norm(entity1_coords[:, np.newaxis] - entity2_coords[np.newaxis, :], axis=-1)

    # calculate distance cutoff for each atom pair, considering VdW radii
    distance_cutoff = entity1_vdw[:, np.newaxis] + entity2_vdw[np.newaxis, :]

    # multiply distance cutoffs with set parameter
    distance_cutoff = distance_cutoff * vdw_multiplier

    # compare distances to distance_cutoff
    check = dgram - distance_cutoff

    # Find the indices where the distances are smaller than the cutoff (clashing)
    clashing_pairs = np.argwhere(check < 0)

    # check if any clashes have occurred
    if len(clashing_pairs) >= 1:
        clashing_atoms = [[entity1[i], entity2[j]] for i, j in clashing_pairs]
        return clashing_atoms
    return None


def atoms_of_functional_groups():
    '''
    PDB names of functional group atoms for each amino acid
    '''
    return ["NH1", "NH2", "OD1", "OD2", "ND2", "NE", "SG", "OE1", "OE2", "NE2", "ND1", "NZ", "SD", "OG", "OG1", "NE1", "OH"]


def sort_frags_df_by_score(frags_df, backbone_score_weight, rotamer_score_weight, frag_length):
    '''
    creates a single fragment out of fragments with same identifier, sorts fragments according to a composite score (comprised of backbone and rotamer score) 
    '''
    # calculate number of fragments
    total_fragments = frags_df['frag_num'].nunique()

    # correct angles for better median calculations
    for col in ['phi', 'psi', 'omega']:
        frags_df.loc[frags_df[col] <= -175, col] += 360

    cols = frags_df.columns
    AA_i     = cols.get_loc('AA')
    phi_i    = cols.get_loc('phi')
    psi_i    = cols.get_loc('psi')
    omega_i  = cols.get_loc('omega')
    frag_i   = cols.get_loc('frag_num')
    rotpos_i = cols.get_loc('rotamer_pos')
    rotid_i  = cols.get_loc('rotamer_id')

    df_list = []
    frag_num = 0
    # assume frag_num is defined before the loop
    for _, unique_df in frags_df.groupby('frag_identifier', sort=False):
        # read rotamer info
        rotamer_pos = int(unique_df.iat[0, rotpos_i])
        rotamer_id  = unique_df.iat[0, rotid_i]

        # backbone count (fast path if guaranteed layout: len(unique_df)//frag_length)
        backbone_count = len(unique_df)//frag_length

        # compute medians for each residue position
        # (your existing lists phis/psis/omegas work; vectorized shown here)
        n = len(unique_df)
        bc = n // frag_length

        phi_v   = unique_df.iloc[:, phi_i].to_numpy()
        psi_v   = unique_df.iloc[:, psi_i].to_numpy()
        omega_v = unique_df.iloc[:, omega_i].to_numpy()

        phis   = np.nanmedian(phi_v.reshape(frag_length, bc),   axis=1)
        psis   = np.nanmedian(psi_v.reshape(frag_length, bc),   axis=1)
        omegas = np.nanmedian(omega_v.reshape(frag_length, bc), axis=1)

        head = slice(0, frag_length)

        # set AA for the first fragment
        unique_df.iloc[head, AA_i] = 'GLY'
        unique_df.iat[rotamer_pos - 1, AA_i] = rotamer_id  # still OK (scalar)

        # write medians only into the first fragment rows
        unique_df.iloc[head, [phi_i, psi_i, omega_i]] = np.column_stack((phis, psis, omegas))

        # and only tag those rows
        unique_df.iloc[head, frag_i] = frag_num
        unique_df.loc[unique_df.index[head], 'backbone_count'] = backbone_count

        df_list.append(unique_df.iloc[head])
        frag_num += 1

    frags_df = pd.concat(df_list, ignore_index=True)

    frags_df.drop(['pdb', 'ss', 'frag_identifier'], axis=1, inplace=True)
    frags_df['backbone_probability'] = frags_df['backbone_count'] / total_fragments

    #sort frags by fragment score
    grouped = frags_df[['frag_num', 'backbone_probability', 'rotamer_score']].groupby('frag_num', sort=False).mean(numeric_only=True)
    grouped['log_backbone_probability'] = np.log(grouped['backbone_probability'])
    grouped = normalize_col(grouped, 'log_backbone_probability', True, "backbone_score")
    grouped = combine_normalized_scores(grouped, 'fragment_score', ['backbone_score', 'rotamer_score'], [backbone_score_weight, rotamer_score_weight], False, False)
    grouped = grouped[['log_backbone_probability', 'backbone_score', 'fragment_score']].sort_values('fragment_score', ascending=False)
    frags_df = grouped.merge(frags_df, left_index=True, right_on="frag_num").reset_index(drop=True)
    return frags_df

def calculate_rmsd_bb(entity1, entity2):
    '''
    calculates rmsd of 2 structures considering CA atoms. does no superposition!
    '''
    bb_atoms = ["CA"]
    entity1_atoms = [atom for atom in entity1.get_atoms() if atom.id in bb_atoms]
    entity2_atoms = [atom for atom in entity2.get_atoms() if atom.id in bb_atoms]

    rmsd = math.sqrt(sum([(atom1 - atom2) ** 2 for atom1, atom2 in zip(entity1_atoms, entity2_atoms)]) / len(entity1_atoms))

    return rmsd

def create_fragment_from_df(df:pd.DataFrame, rotamer_position, aa_alphabet):
    '''
    creates a biopython chain from dataframe containing angles and coordinates
    '''
    chain = Chain.Chain('A')
    df.reset_index(drop=True, inplace=True)

    serial_num = 1
    for index, row in df.iterrows():
        res = Residue.Residue((' ', index + 1, ' '), row['AA'], ' ')
        for atom in ["N", "CA", "C", "O"]:
            coords = np.array([row[f'{atom}_x'], row[f'{atom}_y'], row[f'{atom}_z']])
            bfactor = 0 if math.isnan(row['probability']) else round(row['probability'] * 100, 2)
            bb_atom = Atom.Atom(name=atom, coord=coords, bfactor=bfactor, occupancy=1.0, altloc=' ', fullname=f' {atom} ', serial_number=serial_num, element=atom[0])
            serial_num += 1
            res.add(bb_atom)
        if index + 1 == rotamer_position:
            chi_angles = [None if math.isnan(row[chi]) else row[chi] for chi in ['chi1', 'chi2', 'chi3', 'chi4']]
            rotamer_id = row["AA"]
            prob = row["probability"]
        chain.add(res)

    chain.atom_to_internal_coordinates()

    for index, row in df.iterrows():
        chain[index + 1].internal_coord.set_angle('phi', row['phi'])
        chain[index + 1].internal_coord.set_angle('psi', row['psi'])
        chain[index + 1].internal_coord.set_angle('omega', row['omega'])
    chain.internal_to_atom_coordinates()

    fragment = attach_rotamer_to_fragments(chain, rotamer_position, rotamer_id, chi_angles, prob, aa_alphabet) # pylint: disable=E0606

    for res in chain:
        if hasattr(res, "internal_coord"):
            delattr(res, "internal_coord")

    delattr(fragment, "internal_coord")

    return fragment

def attach_rotamer_to_fragments(frag, rot_pos, rotamer_id, chi_angles, probability, aa_alphabet):
    '''
    generates a rotamer based on angles and attaches it to a backbone fragment
    '''
    to_mutate = frag[rot_pos]
    resid = to_mutate.id
    backbone_angles = extract_backbone_angles(frag, rot_pos)
    backbone_bondlengths = extract_backbone_bondlengths(frag, rot_pos)
    res = generate_rotamer(aa_alphabet, rotamer_id, resid, backbone_angles["phi"], backbone_angles["psi"], backbone_angles["omega"], backbone_angles["carb_angle"], backbone_angles["tau"], backbone_bondlengths["N_CA"], backbone_bondlengths["CA_C"], backbone_bondlengths["C_O"], chi_angles[0], chi_angles[1], chi_angles[2], chi_angles[3], probability)
    delattr(res, 'internal_coord')
    rotamer_on_fragments = attach_rotamer_to_backbone(frag, to_mutate, res)

    return rotamer_on_fragments

def attach_rotamer_to_backbone(fragment, fragment_residue, rotamer):
    '''
    attaches a rotamer to a backbone fragment using Biopython
    '''
    fragment.detach_child(fragment_residue.id)
    to_mutate_atoms = []
    res_atoms = []
    for atom in ["N", "CA", "C"]:
        to_mutate_atoms.append(fragment_residue[atom])
        res_atoms.append(rotamer[atom])
    sup = Bio.PDB.Superimposer()
    sup.set_atoms(to_mutate_atoms, res_atoms)
    sup.apply(rotamer)
    rotamer.detach_child("O")
    rotamer.add(fragment_residue["O"])
    fragment.insert(rotamer.id[1]-1, rotamer)

    return fragment

def check_if_angle_in_bin(df, phi, psi, phi_psi_bin) -> pd.DataFrame:
    '''
    calculates angle difference to specified angles and compares if value falls within bin size. filters based on that.
    '''
    df['phi_difference'] = df.apply(lambda row: angle_difference(row['phi'], phi), axis=1)
    df['psi_difference'] = df.apply(lambda row: angle_difference(row['psi'], psi), axis=1)

    df = df[(df['psi_difference'] < phi_psi_bin / 2) & (df['phi_difference'] < phi_psi_bin / 2)]
    df = df.drop(['phi_difference', 'psi_difference'], axis=1)
    return df


def identify_positions_for_rotamer_insertion(fraglib_path, rotlib, rot_sec_struct, phi_psi_bin, directory, script_path, prefix, chi_std_multiplier, jobstarter) -> pd.DataFrame:
    '''
    identifies positions in the library that fit to the selected rotamers
    '''
    os.makedirs(directory, exist_ok=True)
    out_pkl = os.path.join(directory, f"{prefix}_rotamer_positions_collected.pkl")
    if os.path.isfile(out_pkl):
        log_and_print(f"Found existing scorefile at {out_pkl}. Skipping step.")
        return pd.read_pickle(out_pkl)

    in_filenames = []
    out_filenames = []
    for index, row in rotlib.iterrows():
        in_file = os.path.join(directory, f"{prefix}_rotamer_{index}.json")
        out_file = os.path.join(directory, f"{prefix}_rotamer_positions_{index}.pkl")
        row.to_json(in_file)
        in_filenames.append(in_file)
        out_filenames.append(out_file)

    cmds = [f"{os.path.join(PROTFLOW_ENV, 'python')} {script_path} --input_json {in_file} --fraglib {fraglib_path} --output_pickle {out_file} --phi_psi_bin {phi_psi_bin} --chi_std_multiplier {chi_std_multiplier}" for in_file, out_file in zip(in_filenames, out_filenames)]
    if rot_sec_struct:
        cmds = [cmd + f" --rot_sec_struct {rot_sec_struct}" for cmd in cmds]

    jobstarter.start(cmds=cmds, jobname="position_identification", output_path=directory)

    rotamer_positions = []
    for index, out_file in enumerate(out_filenames):
        df = pd.read_pickle(out_file)
        df['rotamer_index'] = index
        os.remove(out_file)
        log_and_print(f"Found {len(df.index)} positions for rotamer {index}")
        rotamer_positions.append(df)

    rotamer_positions = pd.concat(rotamer_positions)
    # TODO: Find out why this is necessary, there should not be any duplicates in theory
    rotamer_positions = rotamer_positions.loc[~rotamer_positions.index.duplicated(keep='first')]
    rotamer_positions.to_pickle(out_pkl)

    return rotamer_positions


def normalize_col(df:pd.DataFrame, col:str, scale:bool=False, output_col_name:str=None) -> pd.DataFrame:
    '''
    normalize a dataframe column (median=zero if scale is False)
    '''
    median = df[col].median()
    std = df[col].std()
    if not output_col_name:
        output_col_name = f"{col}_normalized"
    if df[col].nunique() == 1:
        df[output_col_name] = 0
        return df
    df[output_col_name] = (df[col] - median) / std
    if scale:
        df = scale_col(df=df, col=output_col_name, inplace=True)
    return df

def scale_col(df:pd.DataFrame, col:str, inplace=False) -> pd.DataFrame:
    '''
    scale a dataframe column to values between 0 and 1
    '''
    factor = df[col].max() - df[col].min()
    df[f"{col}_scaled"] = df[col] / factor
    df[f"{col}_scaled"] = df[f"{col}_scaled"] + (1 - df[f"{col}_scaled"].max())
    if inplace:
        df[col] = df[f"{col}_scaled"]
        df.drop(f"{col}_scaled", axis=1, inplace=True)
    return df

def has_consecutive_numbers(df:pd.DataFrame, column: str, fragsize):
    '''
    check if numbers in df column are consecutive
    '''
    if df[column].diff().sum() + 1 == fragsize:
        return True
    return False

def assign_frag_identifier(df, pos):
    '''
    create a fragment identifier that contains information on backbone angles and rotamer position
    '''
    dihedrals = ['phi', 'psi', 'omega']

    # Round and clean dihedrals
    for angle in dihedrals:
        df[f"{angle}_rounded"] = df[angle].round(-1).replace(-180, 180).astype(int)

    # Function to create the identifier for each frag_num group
    def make_identifier(group):
        dihedral_seq = tuple(zip(group['phi_rounded'], group['psi_rounded'], group['omega_rounded']))  # preserve order
        rotamer = group['rotamer_index'].iloc[0]
        return (dihedral_seq, rotamer, pos)

    # Apply group-wise identifier creation
    identifiers = df.groupby('frag_num', sort=False).apply(make_identifier)
    identifiers.name = 'frag_identifier'
    identifiers = identifiers.reset_index()

    # Merge the identifier back to the full DataFrame
    df = df.merge(identifiers, on='frag_num', how='left')

    return df

def extract_fragments(rotamer_positions_df: pd.DataFrame, fraglib_path: str, frag_pos_to_replace: list, fragsize: int, working_dir: str, script_path:str, jobstarter):
    '''
    frag_pos_to_replace: the position in the fragment the future rotamer should be inserted. central position recommended.
    residue_identity: only accept fragments with the correct residue identity at that position (recommended)
    rotamer_secondary_structure: accepts a string describing secondary structure (B: isolated beta bridge residue, E: strand, G: 3-10 helix, H: alpha helix, I: pi helix, T: turn, S: bend, -: none (not in the sense of no filter --> use None instead!)). e.g. provide EH if central atom in fragment should be a helix or strand.
    '''
    def write_rotamer_extraction_cmd(script_path, rotamer_positions_path, fraglib_path, rotamer_positions, fragsize, out_path):
        cmd = f"{os.path.join(PROTFLOW_ENV, 'python')} {script_path} --rotpos_path {rotamer_positions_path} --fraglib_path {fraglib_path} --rotamer_positions {','.join([str(pos) for pos in rotamer_positions])} --fragsize {fragsize} --outfile {out_path}"
        return cmd

    #choose fragments from fragment library that contain the positions selected above
    frag_dict = {}
    rotamer_positions_df["temp_index_for_merge"] = rotamer_positions_df.index

    os.makedirs(working_dir, exist_ok=True)

    cmds = []
    out_paths = []
    # split rotamer positions df into several parts, create fragments for each part
    for i, split_df in enumerate(np.array_split(rotamer_positions_df, min(jobstarter.max_cores, len(rotamer_positions_df.index)))):
        split_df.to_pickle(positions_path := os.path.join(working_dir, f"rotamer_positions_{i}.pkl"))
        out_paths.append(out_path := os.path.join(working_dir, f"fragments_{i}.pkl"))
        cmds.append(write_rotamer_extraction_cmd(script_path, positions_path, fraglib_path, frag_pos_to_replace, fragsize, out_path))

    jobstarter.start(cmds=cmds, jobname="fragment_extraction", wait=True, output_path=working_dir) # distribute fragment extraction to cluster

    # combine all fragments
    num_frags = 0
    frags_dfs = []
    for out_path in out_paths:
        # read in results
        frags_df = pd.read_pickle(out_path)
        max_frags = frags_df["frag_num"].max()
        frags_df['frag_num'] = frags_df['frag_num'] + num_frags # update frag_num to have continuous fragnums over all positions
        num_frags = num_frags + max_frags + 1
        frags_dfs.append(frags_df)

    frags_dfs = pd.concat(frags_dfs)

    return frags_dfs


def extract_backbone_angles(chain, resnum:int):
    '''
    takes a biopython chain and extracts phi/psi/omega angles of specified residue
    '''
    #convert to internal coordinates, read phi/psi angles
    #chain = copy.deepcopy(chain)
    #chain.internal_coord
    chain.atom_to_internal_coordinates()
    phi = chain[resnum].internal_coord.get_angle("phi")
    psi = chain[resnum].internal_coord.get_angle("psi")
    omega = chain[resnum].internal_coord.get_angle("omg")
    carb_angle = round(chain[resnum].internal_coord.get_angle("N:CA:C:O"), 1)
    tau = round(chain[resnum].internal_coord.get_angle("tau"), 1)
    if phi is not None:
        phi = round(phi, 1)
    if psi is not None:
        psi = round(psi, 1)
    if omega is not None:
        omega = round(omega, 1)
    return {"phi": phi, "psi": psi, "omega": omega, "carb_angle": carb_angle, "tau": tau}

def extract_backbone_bondlengths(chain, resnum:int):
    '''
    takes a biopython chain and extracts phi/psi/omega angles of specified residue
    '''
    #convert to internal coordinates, read phi/psi angles
    #chain = copy.deepcopy(chain)
    #chain.internal_coord
    chain.atom_to_internal_coordinates()
    n_ca = round(chain[resnum].internal_coord.get_length("N:CA"), 3)
    ca_c = round(chain[resnum].internal_coord.get_length("CA:C"), 3)
    c_o = round(chain[resnum].internal_coord.get_length("C:O"), 3)
    return {"N_CA": n_ca, "CA_C": ca_c, "C_O": c_o}


def generate_rotamer(aa_alphabet_structure, residue_identity:str, res_id, phi:float=None, psi:float=None, omega:float=None, carb_angle:float=None, tau:float=None, N_CA_length:float=None, CA_C_length:float=None, C_O_length:float=None, chi1:float=None, chi2:float=None, chi3:float=None, chi4:float=None, rot_probability:float=None): # pylint: disable=C0103
    '''
    builds a rotamer from residue identity, phi/psi/omega/chi angles
    '''
    alphabet = copy.deepcopy(aa_alphabet_structure)
    for res in alphabet[0]["A"]:
        if res.get_resname() == residue_identity:
            #set internal coordinates
            alphabet[0]["A"].atom_to_internal_coordinates()
            #change angles to specified value
            if tau:
                res.internal_coord.set_angle("tau", tau)
            if carb_angle:
                res.internal_coord.bond_set("N:CA:C:O", carb_angle)
            if phi:
                res.internal_coord.set_angle("phi", phi)
            if psi:
                res.internal_coord.set_angle("psi", psi)
            if omega:
                res.internal_coord.set_angle("omega", omega)
            if N_CA_length:
                res.internal_coord.set_length("N:CA", N_CA_length)
            if CA_C_length:
                res.internal_coord.set_length("CA:C", CA_C_length)
            if C_O_length:
                res.internal_coord.set_length("C:O", C_O_length)

            max_chis = num_chis_for_residue_id(residue_identity)

            if max_chis > 0:
                res.internal_coord.bond_set("chi1", chi1)
            if max_chis > 1:
                res.internal_coord.bond_set("chi2", chi2)
            if max_chis > 2:
                res.internal_coord.bond_set("chi3", chi3)
            if max_chis > 3:
                res.internal_coord.set_angle("chi4", chi4)
            alphabet[0]["A"].internal_to_atom_coordinates()
            #change residue number to the one that is replaced (detaching is necessary because otherwise 2 res with same resid would exist in alphabet)
            alphabet[0]["A"].detach_child(res.id)
            res.id = res_id
            if rot_probability:
                for atom in res.get_atoms():
                    atom.bfactor = rot_probability * 100

            return res

def identify_rotamer_position_by_probability(df):
    '''
    drop all rows that do not contain the rotamer, return a series --> might cause issues downstream if more than one rotamer on a fragment
    '''
    rotamer_pos = df.dropna(subset = ['probability']).squeeze()
    return rotamer_pos

def align_to_sidechain(entity, entity_residue_to_align, sidechain, flip_symmetric:bool=True):
    '''
    aligns an input structure (bb_fragment_structure, resnum_to_align) to a sidechain residue (sc_structure, resnum_to_alignto)
    '''
    sc_residue_identity = sidechain.get_resname()

    #superimpose structures based on specified atoms
    bbf_atoms = atoms_for_func_group_alignment(entity_residue_to_align)
    sc_atoms = atoms_for_func_group_alignment(sidechain)
    if flip_symmetric and sc_residue_identity in tip_symmetric_residues():
        order = [1, 0, 2]
        sc_atoms = [sc_atoms[i] for i in order]
    sup = Bio.PDB.Superimposer()
    sup.set_atoms(sc_atoms, bbf_atoms)
    #sup.rotran
    sup.apply(entity)

    return entity

def identify_his_central_atom(histidine, ligand):
    '''
    identify the atom for histidine rotation by distance to ligand
    '''
    his_NE2 = histidine["NE2"] # pylint: disable=C0103
    his_ND1 = histidine["ND1"] # pylint: disable=C0103
    lig_atoms =  [atom for atom in ligand.get_atoms()]
    NE2_distance = min([his_NE2 - atom for atom in lig_atoms]) # pylint: disable=C0103
    ND1_distance = min([his_ND1 - atom for atom in lig_atoms]) # pylint: disable=C0103
    if NE2_distance < ND1_distance:
        his_central_atom = "NE2"
    else:
        his_central_atom = "ND1"
    return his_central_atom

def rotate_histidine_fragment(entity, degrees, theozyme_residue, his_central_atom, ligand):
    '''
    rotate histidine residue around central atom
    '''
    if his_central_atom == "auto":
        if not ligand:
            raise RuntimeError("Ligand is required if using <flip_histidines='auto'>!")
        his_central_atom = identify_his_central_atom(theozyme_residue, ligand)
    if his_central_atom == "NE2":
        half = theozyme_residue["ND1"].coord + 0.5 * (theozyme_residue["CG"].coord - theozyme_residue["ND1"].coord)
    elif his_central_atom == "ND1":
        half = theozyme_residue["NE2"].coord + 0.5 * (theozyme_residue["CD2"].coord - theozyme_residue["NE2"].coord)
    entity = rotate_entity_around_axis(entity, theozyme_residue[his_central_atom].coord, half, degrees) # pylint: disable=E0606

    return entity

def rotate_phenylalanine_fragment(entity, degrees, theozyme_residue):
    '''
    rotate phenylalanine residue around center
    '''
    center = theozyme_residue["CZ"].coord - 0.5 * (theozyme_residue["CZ"].coord - theozyme_residue["CG"].coord)
    a = theozyme_residue["CG"].coord - theozyme_residue["CZ"].coord
    b = theozyme_residue["CE1"].coord - theozyme_residue["CZ"].coord
    n = np.cross(a, b)
    n = n / np.linalg.norm(n)
    point = center + n

    entity = rotate_entity_around_axis(entity, center, point, degrees)
    return entity

def rotate_entity_around_axis(entity, coords_1, coords_2, angle):
    '''
    rotate a biopython entity around an axis
    '''
    rotation_axis = coords_1 - coords_2
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    angle_radians = np.radians(angle)
    for atom in entity.get_atoms():
        array = atom.coord - coords_1
        rotated_array = rotate_array_around_vector(array, rotation_axis, angle_radians)
        atom.coord = rotated_array + coords_1
    return entity

def rotation_matrix(v, x):
    '''
    calculate rotation matrix
    '''
    k = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    i = np.identity(3)
    r = i + np.sin(x)*k + (1-np.cos(x))*np.dot(k,k)
    return r

def rotate_array_around_vector(array, axis, angle):
    """
    Rotate a 3D NumPy array around a specified axis by a given angle.

    Parameters:
        array: A 3D NumPy array to be rotated.
        axis: A 3D NumPy array representing the rotation axis.
        angle: The angle (in radians) of rotation.

    Returns:
        The rotated 3D NumPy array.
    """
    rotation_matrix_3x3 = rotation_matrix(axis, angle)
    rotated_array = np.dot(rotation_matrix_3x3, array.T).T
    return rotated_array


def atoms_for_func_group_alignment(residue):
    '''
    return the atoms used for superposition via functional groups
    '''
    sc_residue_identity = residue.get_resname()

    if not sc_residue_identity in func_groups():
        raise RuntimeError(f'Unknown residue with name {sc_residue_identity}!')
    else:
        return [residue[atom] for atom in func_groups()[sc_residue_identity]]
    
def func_groups():
    '''
    dictionary for functional group atoms for each residue
    '''
    func_group_atoms = {
        "ALA": ["CB", "CA", "N"],
        "ARG": ["NH1", "NH2", "CZ"],
        "ASP": ["OD1", "OD2", "CG"],
        "ASN": ["OD1", "ND2", "CG"],
        "CYS": ["SG", "CB", "CA"],
        "GLU": ["OE1", "OE2", "CD"],
        "GLN": ["OE1", "NE2", "CD"],
        "GLY": ["CA", "N", "C"],
        "HIS": ["ND1", "NE2", "CG"],
        "ILE": ["CD1", "CG1", "CB"],
        "LEU": ["CD1", "CD2", "CG"],
        "LYS": ["NZ", "CE", "CD"],
        "MET": ["CE", "SD", "CG"],
        "PHE": ["CD1", "CD2", "CZ"],
        "PRO": ["CD", "CG", "CB"],
        "SER": ["OG", "CB", "CA"],
        "THR": ["OG1", "CG2", "CB"],
        "TRP": ["NE1", "CZ3", "CG"],
        "TYR": ["CE1", "CE2", "OH"],
        "VAL": ["CG1", "CG2", "CB"]
        }

    return func_group_atoms


def clean_input_backbone(entity: Structure):
    '''
    clean input backbone for chain names and resnums
    '''
    chains = [chain for chain in entity.get_chains()]
    models = [model for model in entity.get_models()]
    if len(chains) > 1 or len(models) > 1:
        logging.error("Input backbone fragment pdb must only contain a single chain and a single model!")
    for model in models:
        model.id = 0
    for chain in chains:
        chain.id = "A"
    for index, residue in enumerate(entity.get_residues()):
        residue.id = (residue.id[0], index + 1, residue.id[2])
    for atom in entity.get_atoms():
        atom.bfactor = 0
    return entity[0]["A"]

def identify_residues_with_equivalent_func_groups(residue, add_equivalent:bool=False):
    '''
    checks if residues with same functional groups exist, returns a list of these residues
    '''
    resname = residue.get_resname()
    if not add_equivalent:
        return [resname]
    else:
        if resname in ['ASP', 'GLU']:
            return ['ASP', 'GLU']
        elif resname in ['ASN', 'GLN']:
            return ['ASN', 'GLN']
        elif resname in ['VAL', 'ILE']:
            return ['VAL', 'LEU']
        else:
            return [resname]

def rotamers_for_backbone(resnames, rotlib_path, phi, psi, rot_prob_cutoff:float=0.05, prob_diff_to_best:float=0.5, max_rotamers:int=70, max_stdev:float=2, level:int=2):
    '''
    identify rotamers that fit to a given backbone
    '''
    rotlib_list = []
    for res in resnames:
        if res in ["ALA", "GLY"]:
            #TODO: assign proper scores for log prob and occurrence
            rotlib = pd.DataFrame([[res, phi, psi, float("nan"), 1, float("nan"), 0, 0]], columns=["AA", "phi", "psi", "count", "probability", "phi_psi_occurrence", "log_prob", "log_occurrence"])
            rotlib_list.append(rotlib)
        else:
            rotlib = return_residue_rotamer_library(rotlib_path, res)
            rotlib_list.append(identify_rotamers_suitable_for_backbone(phi, psi, rotlib, rot_prob_cutoff, prob_diff_to_best, max_rotamers, max_stdev, level))
    if len(rotlib_list) > 1:
        filtered_rotlib = pd.concat(rotlib_list)
        filtered_rotlib = filtered_rotlib.sort_values("probability", ascending=False)
        filtered_rotlib.reset_index(drop=True, inplace=True)
        return filtered_rotlib
    else:
        return rotlib_list[0]


def identify_rotamers_suitable_for_backbone(phi:float, psi:float, rotlib:pd.DataFrame, prob_cutoff:float=None, prob_diff_to_best:float=None, max_rotamers:int=None, max_stdev:float=2, level:int=3):
    '''
    identifies suitable rotamers by filtering for phi/psi angles
    if fraction is given, returns only the top rotamer fraction ranked by probability (otherwise returns all rotamers)
    if prob_cutoff is given, returns only rotamers more common than prob_cutoff
    '''
    rotlib.rename(columns={'identity': 'AA'}, inplace=True)
    #round dihedrals to the next tens place
    if phi is not None:
        phi = round(phi, -1)
    if psi is not None:
        psi = round(psi, -1)
    #extract all rows containing specified phi/psi angles from library
    if phi and psi:
        log_and_print(f"Searching for rotamers in phi/psi bin {phi}/{psi}.")
        rotlib = rotlib.loc[(rotlib['phi'] == phi) & (rotlib['psi'] == psi)].reset_index(drop=True)
    elif not phi or not psi:
        if not phi:
            rotlib = rotlib[rotlib['psi'] == psi].reset_index(drop=True)
        elif not psi:
            rotlib = rotlib[rotlib['phi'] == phi].reset_index(drop=True)
        rotlib = rotlib.loc[rotlib['phi_psi_occurrence'] >= 1]
        rotlib = rotlib.drop_duplicates(subset=['phi', 'psi'], keep='first')
        rotlib.sort_values("probability", ascending=False)
        rotlib = rotlib.head(5)
    #filter top rotamers
    rotlib = rotlib.sort_values("probability", ascending=False)
    if prob_cutoff:
        rotlib = rotlib.loc[rotlib['probability'] > prob_cutoff]
    if prob_diff_to_best:
        rotlib = rotlib[rotlib['probability'] >= rotlib['probability'].max() * (1 - prob_diff_to_best)]
    if level > 0:
        rotlib = diversify_chi_angles(rotlib, max_stdev, level)
        #filter again, since diversify_chi_angles produces rotamers with lower probability
        if prob_cutoff:
            rotlib = rotlib.loc[rotlib['probability'] > prob_cutoff]
    if prob_diff_to_best:
        rotlib = rotlib[rotlib['probability'] >= rotlib['probability'].max() * (1 - prob_diff_to_best)]
    if max_rotamers:
        rotlib = rotlib.head(max_rotamers)
    return rotlib


def diversify_chi_angles(rotlib, max_stdev:float=2, level:int=3):
    '''
    adds additional chi angles based on standard deviation.
    max_stdev: defines how far to stray from mean based on stdev. chi_new = chi_orig +- stdev * max_stdev
    level: defines how many chis should be sampled within max_stdev. if level = 1, mean, mean + stdev*max_stdev, mean - stdev*max_stdev will be returned. if level = 2, mean, mean + 1/2 stdev*max_stdev, mean + stdev*max_stdev, mean - 1/2 stdev*max_stdev, mean - stdev*max_stdev will be returned
    '''
    #check which chi angles exist in rotamer library
    columns = list(rotlib.columns)
    columns = [column for column in columns if column.startswith('chi') and not 'sig' in column]
    #generate deviation parameters
    devs = [max_stdev * i / level for i in range(-level, level +1)]
    #calculate chi angles
    for chi_angle in columns:
        new_chis_list = []
        for dev in devs:
            new_chis = alter_chi(rotlib, chi_angle, f'{chi_angle}sig', dev)
            new_chis_list.append(new_chis)
        rotlib = pd.concat(new_chis_list)
        rotlib.drop([f'{chi_angle}sig'], axis=1, inplace=True)
        rotlib[chi_angle] = round(rotlib[chi_angle], 1)
    rotlib.sort_values('probability', inplace=True, ascending=False)
    rotlib.reset_index(drop=True, inplace=True)
    return rotlib

def create_df_from_fragment(backbone, fragment_name):
    '''
    extract information from fragment to create a dataframe
    '''
    pdbnames = [fragment_name for res in backbone.get_residues()]
    resnames = [res.resname for res in backbone.get_residues()]
    pos_list = [res.id[1] for res in backbone.get_residues()]
    problist = [float("nan") for res in backbone.get_residues()]
    phi_angles = [extract_backbone_angles(backbone, res.id[1])['phi'] for res in backbone.get_residues()]
    psi_angles = [extract_backbone_angles(backbone, res.id[1])['psi'] for res in backbone.get_residues()]
    omega_angles = [extract_backbone_angles(backbone, res.id[1])['omega'] for res in backbone.get_residues()]
    CA_x_coords_list = [(round(res["CA"].get_coord()[0], 3)) for res in backbone.get_residues()] # pylint: disable=C0103
    CA_y_coords_list = [(round(res["CA"].get_coord()[1], 3)) for res in backbone.get_residues()] # pylint: disable=C0103
    CA_z_coords_list = [(round(res["CA"].get_coord()[2], 3)) for res in backbone.get_residues()] # pylint: disable=C0103
    C_x_coords_list = [(round(res["C"].get_coord()[0], 3)) for res in backbone.get_residues()] # pylint: disable=C0103
    C_y_coords_list = [(round(res["C"].get_coord()[1], 3)) for res in backbone.get_residues()] # pylint: disable=C0103
    C_z_coords_list = [(round(res["C"].get_coord()[2], 3)) for res in backbone.get_residues()] # pylint: disable=C0103
    N_x_coords_list = [(round(res["N"].get_coord()[0], 3)) for res in backbone.get_residues()] # pylint: disable=C0103
    N_y_coords_list = [(round(res["N"].get_coord()[1], 3)) for res in backbone.get_residues()] # pylint: disable=C0103
    N_z_coords_list = [(round(res["N"].get_coord()[2], 3)) for res in backbone.get_residues()] # pylint: disable=C0103
    O_x_coords_list = [(round(res["O"].get_coord()[0], 3)) for res in backbone.get_residues()] # pylint: disable=C0103
    O_y_coords_list = [(round(res["O"].get_coord()[1], 3)) for res in backbone.get_residues()] # pylint: disable=C0103
    O_z_coords_list = [(round(res["O"].get_coord()[2], 3)) for res in backbone.get_residues()] # pylint: disable=C0103

    df = pd.DataFrame(list(zip(pdbnames, resnames, pos_list, phi_angles, psi_angles, omega_angles, CA_x_coords_list, CA_y_coords_list, CA_z_coords_list, C_x_coords_list, C_y_coords_list, C_z_coords_list, N_x_coords_list, N_y_coords_list, N_z_coords_list, O_x_coords_list, O_y_coords_list, O_z_coords_list, problist)), columns=["pdb", "AA", "position", "phi", "psi", "omega", "CA_x", "CA_y", "CA_z", "C_x", "C_y", "C_z", "N_x", "N_y", "N_z", "O_x", "O_y", "O_z", "probability"])
    df[["chi1", "chi2", "chi3", "chi4"]] = float("nan")
    return df

def normal_dist_density(x):
    '''
    calculates y value for normal distribution from distance from mean TODO: check if it actually makes sense to do it this way
    '''
    y = math.e **(-(x)**2 / 2)
    return y


def alter_chi(rotlib, chi_column, chi_sig_column, dev):
    '''
    calculate deviations from input chi angle for rotamer library
    '''
    new_chis = copy.deepcopy(rotlib)
    new_chis[chi_column] = new_chis[chi_column] + new_chis[chi_sig_column] * dev
    new_chis['probability'] = new_chis['probability'] * normal_dist_density(dev)
    new_chis['log_prob'] = np.log(new_chis['probability'])
    return new_chis

def exchange_covalent(covalent_bond):
    '''
    exchange atoms involved in covalent bonds when using residues with same functional groups
    '''
    atom = covalent_bond[0]
    exchange_dict = {"OE1": "OD1", "OE2": "OD2", "CD1": "CG1", "CD2": "CG2", "NE2": "ND2", "OD1": "OE1", "OD2": "OE2", "CG1": "CD1", "CG2": "CD2", "ND2": "NE2"}
    new_atom = exchange_dict.get(atom, atom)
    covalent_bond[0] = new_atom
    return covalent_bond

def flip_covalent(covalent_bond, residue):
    '''
    flip atoms involved in covalent bonds when flipping a rotamer
    '''
    atom = covalent_bond[0]
    exchange_dict = {
        "GLU": {"OE1": "OE2", "OE2": "OE1"},
        "ASP": {"OD1": "OD2", "OD2": "OD1"},
        "VAL": {"CD1": "CD2", "CD2": "CD1"},
        "LEU": {"CG1": "CG2", "CG2": "CG1"},
        "ARG": {"NH1": "NH2", "NH2": "NH1"}     
        }
    inner = exchange_dict.get(residue, {})
    new_atom = inner.get(atom, atom)

    covalent_bond[0] = new_atom
    return covalent_bond

def log_and_print(info: str):
    '''
    log and print
    '''
    logging.info(info)
    print(info)
    return info

def combine_normalized_scores(df: pd.DataFrame, name:str, scoreterms:list, weights:list, normalize:bool=False, scale:bool=False):
    '''
    normalize scores and combine them to a single score according to weights
    '''
    if not len(scoreterms) == len(weights):
        raise RuntimeError(f"Number of scoreterms ({len(scoreterms)}) and weights ({len(weights)}) must be equal!")
    df[name] = sum([df[col]*weight for col, weight in zip(scoreterms, weights)]) / sum(weights)
    if df[name].nunique() == 1:
        df[name] = 0
        return df

    if normalize:
        df = normalize_col(df, name, False)
        df.drop(name, axis=1, inplace=True)
        df.rename(columns={f'{name}_normalized': name}, inplace=True)
    if scale:
        df[name] = df[name] / df[name].max()
        df = scale_col(df, name, True)
    return df

def define_rotamer_positions(rotamer_positions, fragsize):
    '''
    parse rotamer positions
    '''
    if rotamer_positions == "auto":
        rotamer_positions = list(range(2, fragsize))
        return rotamer_positions
    elif isinstance(rotamer_positions, list):
        if any(int(pos) > fragsize for pos in rotamer_positions):
            raise KeyError("Rotamer positions must be smaller than fragment size!")
        return [int(pos) for pos in rotamer_positions]
    elif isinstance(rotamer_positions, int):
        if rotamer_positions > fragsize:
            raise KeyError("Rotamer positions must be smaller than fragment size!")
        return [rotamer_positions]
    else:
        raise KeyError(f"rotamer_positions must be 'auto', a list of int or a single int value, not {type(rotamer_positions)}!")

def update_covalent_bonds(covalent_bonds:str, rotamer_id:str, rotamer_position:int, ligand_dict:dict):
    '''
    update covalent bonds with rotamer position information
    '''
    if not covalent_bonds:
        return None
    updated_bonds = []
    for cov_bond in covalent_bonds:
        res_atom, lig_chain, lig_resnum, lig_atom = cov_bond
        ligand = ligand_dict[lig_chain][int(lig_resnum)]
        _, lig_resnum, _ = ligand.id
        lig_id = ligand.get_resname()
        updated_bonds.append(f"{rotamer_position}A_{rotamer_id}-{res_atom}:{lig_resnum}Z_{lig_id}-{lig_atom}")
    return ",".join(updated_bonds)

def import_input_json(file_path):
    """Reads a JSON file and returns its contents as a dictionary."""
    if not os.path.isfile(file_path):
        raise KeyError(f"Could not find input json file at {file_path}!")
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def extract_covalent_bond_info(covalent_bonds, theozyme_residue, lig_dict, resnum, chain):
    '''
    parses covalent bond information
    '''
    cov_bonds = []
    for cov_bond in covalent_bonds:
        res_atom, lig_info = cov_bond.split(":")
        lig_resnum, lig_chain = split_pdb_numbering(lig_info.split("-")[0])
        lig_atom = lig_info.split("-")[1]
        # check if covalent bond atoms are present in theozyme residue and ligand
        if not res_atom in [atom.name for atom in theozyme_residue.get_atoms()]:
            raise KeyError(f"Could not find atom {res_atom} from covalent bond {cov_bond} in residue {chain}{resnum}!")
        if not lig_atom in [atom.name for atom in lig_dict[lig_chain][lig_resnum].get_atoms()]:
            raise KeyError(f"Could not find atom {lig_atom} from covalent bond {cov_bond} in ligand {lig_chain}{lig_resnum}!")
        # add covalent bond to list of covalent bonds for this residue
        cov_bonds.append([res_atom, lig_chain, lig_resnum, lig_atom])
    return cov_bonds

def check_residue_specific_config(argument, args, residue_config):
    '''
    compares general and residue specific arguments
    '''
    if argument in residue_config:
        return residue_config[argument]
    else:
        return getattr(args, argument)

def combine_general_and_residue_specific_config(args, residue_args):
    '''
    overwrites general settings with residue-specific settings, if defined
    '''
    # create new args from general and residue-specific dict
    res_args = copy.deepcopy(args)

    # check for wrong config input (e.g. typos etc)
    for res_key in residue_args:
        if not hasattr(res_args, res_key) and not res_key == "covalent_bonds" or res_key in ["ligands", "theozyme_resnums"]:
            raise KeyError(f"{res_key} in input json is not a valid input!")

    # Merge config and CLI arguments
    for res_key, res_value in residue_args.items():
        setattr(res_args, res_key, res_value)  # Add config values to args

    return res_args

def create_frag_sec_struct_fraction_dict(frag_sec_struct_fraction: str, fragsize: int, rot_sec_struct: str):
    '''parse secondary structure input'''
    if frag_sec_struct_fraction:
        sec_structs = frag_sec_struct_fraction.split(',')
        sec_dict = {}
        for i in sec_structs:
            sec, frac = i.split(':')
            frac = float(frac)
            if frac > 1 or frac < 0:
                logging.error(f'Fraction for secondary structure {sec} must be a value between 0 and 1, but it is {frac}!')
                raise ValueError(f'Fraction for secondary structure {sec} must be a value between 0 and 1, but it is {frac}!')
            if (fragsize - frac * fragsize) < 1 and sec != rot_sec_struct and rot_sec_struct is not None:
                logging.error(f"If limiting all fragment residues to secondary structure {sec}, it is not possible that the rotamer has secondary structure {rot_sec_struct}!")
                raise KeyError(f"If limiting all fragment residues to secondary structure {sec}, it is not possible that the rotamer has secondary structure {rot_sec_struct}!")
            elif (fragsize - frac * fragsize) < 1 and rot_sec_struct is None and len(sec_structs) == 1:
                log_and_print(f"Setting <rot_sec_struct> to {sec} because all residues in fragment have to have secondary structure {sec}!")
                rot_sec_struct = sec
            sec_dict[sec] = float(frac)
    else:
        sec_dict = None

    return sec_dict, rot_sec_struct

def replace_covalent_bonds_chain(chain:str, covalent_bonds:str=None) -> ResidueSelection:
    '''
    update covalent bonds chain information after assigning new chain names
    '''
    if not isinstance(covalent_bonds, str):
        return None
    covalent_bonds = covalent_bonds.split(",")
    new_cov_bonds = []
    for cov_bond in covalent_bonds:
        rot, lig = cov_bond.split(":")
        rechained = rot.split("_")[0][:-1] + chain + "_" + rot.split("_")[1]
        new_cov_bonds.append(":".join([rechained, lig]))
    return ",".join(new_cov_bonds)

def run_clash_detection(data, directory, bb_multiplier, sc_multiplier, script_path, jobstarter):
    '''
    run clash detection between fragments
    '''

    def write_clash_detection_cmd(pose1, pose2, bb_multiplier, sc_multiplier, script_path, directory, prefix):
        cmd = f"{os.path.join(PROTFLOW_ENV, 'python')} {script_path} --pose1 {pose1} --pose2 {pose2} --working_dir {directory} --bb_multiplier {bb_multiplier} --sc_multiplier {sc_multiplier} --output_prefix {prefix}"
        return cmd

    def generate_valid_combinations(n_sets, compat_maps, set_lengths):
        valid_combos = []

        def backtrack(current_indices, depth):
            if depth == n_sets:
                valid_combos.append(tuple(current_indices))
                return

            for idx in range(set_lengths[depth]):
                # Check compatibility of this new index with all previous ones
                is_valid = True
                for prev_set in range(depth):
                    prev_idx = current_indices[prev_set]
                    # Make sure there's an entry
                    if prev_idx not in compat_maps[prev_set][depth]:
                        is_valid = False
                        break
                    if idx not in compat_maps[prev_set][depth][prev_idx]:
                        is_valid = False
                        break
                if is_valid:
                    backtrack(current_indices + [idx], depth + 1)

        backtrack([], 0)
        return valid_combos

    in_files = []
    in_dfs = []
    for pose, df in data.groupby('poses', sort=False):
        filename = os.path.join(directory, f"{os.path.splitext(os.path.basename(pose))[0]}.json")
        df.reset_index(drop=True, inplace=True)
        df.to_json(filename)
        in_files.append(filename)
        in_dfs.append(df)

    set_lengths = [len(df.index) for df in in_dfs]
    n_sets = len(in_files)

    cmds = []
    prefixes = []
    prefix_map = {}
    for i, set1 in enumerate(in_files): # iterative over each set
        for j in range(i+1, n_sets):
            set2 = in_files[j] # define second set
            prefixes.append(prefix := f"{os.path.splitext(os.path.basename(set1))[0]}_{os.path.splitext(os.path.basename(set2))[0]}") # create a unique prefix for each pair
            cmds.append(write_clash_detection_cmd(set1, set2, bb_multiplier, sc_multiplier, script_path, directory, prefix)) # write clash detection cmds
            prefix_map[prefix] = (i, j)
    log_and_print('Calculating pairwise compatibility maps...')

    jobstarter.start(cmds=cmds, jobname="clash_detection", wait=True, output_path=directory) # distribute clash detection to cluster

    # Build compatibility maps using fast filtering (no row iteration)
    compat_maps = [[None] * n_sets for _ in range(n_sets)]

    log_and_print("Importing results...")
    # import results
    clash_dfs = []
    for prefix in prefixes:
        i, j = prefix_map[prefix]
        filepath = os.path.join(directory, f"{prefix}.json")
        clash_df = pd.read_json(filepath)
        clash_dfs.append(clash_df)
        filtered_df = clash_df[clash_df["clash"] == False]

        # Group each pose1_index by its non-clashing pose2_index set
        a_to_b = filtered_df.groupby("pose1_index")["pose2_index"].agg(set).to_dict()
        b_to_a = filtered_df.groupby("pose2_index")["pose1_index"].agg(set).to_dict()

        compat_maps[i][j] = a_to_b
        compat_maps[j][i] = b_to_a  # Optional bidirectional support

        # analyze number of clashes
        bb_bb_clashes = clash_df["bb_bb_clash"].sum()
        bb_sc_clashes = clash_df["bb_sc_clash"].sum()
        sc_sc_clashes = clash_df["sc_sc_clash"].sum()
        log_and_print(f"Number of clashes for combination {prefix}:\nbackbone-backbone clashes: {bb_bb_clashes}\nbackbone-sidechain clashes: {bb_sc_clashes}\nsidechain-sidechain clashes: {sc_sc_clashes}")


    clash_df = pd.concat(clash_dfs)
    bb_bb_clashes = clash_df["bb_bb_clash"].sum()
    bb_sc_clashes = clash_df["bb_sc_clash"].sum()
    sc_sc_clashes = clash_df["sc_sc_clash"].sum()
    log_and_print(f"Total number of clashes:\nbackbone-backbone clashes: {bb_bb_clashes}\nbackbone-sidechain clashes: {bb_sc_clashes}\nsidechain-sidechain clashes: {sc_sc_clashes}")
    log_and_print("If number of sidechain clashes is high, this is often a result of missing covalent bonds. Otherwise, <frag_frag_sc_clash_vdw_multiplier> can be reduced.")

    log_and_print("Generating valid combinations...")
    valid_combos = generate_valid_combinations(n_sets, compat_maps, set_lengths)

    if len(valid_combos) < 1:
        logging.error("No valid non-clashing combinations found. Adjust parameters like Van-der-Waals multiplier or pick different fragments!")

    log_and_print(f"Found {len(valid_combos)} valid combinations.")

    valid_combos_arr = np.array(valid_combos)  # shape (num_ensembles, n_sets)

    log_and_print("Extracting data for each pose...")
    flattened_dfs = []
    for i in range(n_sets):
        indices = valid_combos_arr[:, i].flatten()  # indices from set i
        df = in_dfs[i].iloc[indices]
        flattened_dfs.append(df)

    log_and_print("Combining data to ensemble dataframe...")
    # Combine all into final DataFrame (optimized for speed)
    ensemble_df = pd.concat(flattened_dfs, ignore_index=True)
    ensemble_df["ensemble_num"] = [i for i in range(len(valid_combos))] * n_sets
    ensemble_df.reset_index(drop=True, inplace=True)

    #log_and_print("Saving results...")
    #ensemble_df.to_pickle(os.path.join(directory, "ensembles.pkl"))

    return ensemble_df

def sort_dataframe_groups_by_column(df:pd.DataFrame, group_col:str, sort_col:str, method="mean", ascending:bool=True, filter_top_n:int=None, randomize_ties:bool=False) -> pd.DataFrame:
    '''group by group column and calculate mean values'''
    df_sorted = df.groupby(group_col, sort=False).agg({sort_col: method})
    if randomize_ties:
        df_sorted["temp_randomizer"] = np.random.rand(len(df_sorted))
        df_sorted.sort_values([sort_col, "temp_randomizer"], ascending=ascending, inplace=True)
    else:
        df_sorted.sort_values(sort_col, ascending=ascending, inplace=True)
    # filter
    if filter_top_n:
        df_sorted = df_sorted.head(filter_top_n)
    # merge back with original dataframe
    df = df_sorted.loc[:, []].merge(df, left_index=True, right_on=group_col)
    # reset index
    df.reset_index(drop=True, inplace=True)
    return df

def concat_columns(group):
    '''concat columns within a group to a single str'''
    non_null_elements = group.dropna().astype(str)
    return ','.join(non_null_elements) if not non_null_elements.empty else None

def split_str_to_dict(key_str, value_str, sep):
    '''split a str according to sep, convert to dict'''
    return dict(zip(key_str.split(sep), [list(i) for i in value_str.split(sep)]))

def create_motif_residues(chain_str, fragsize_str, sep:str=","):
    '''create motif residue dict'''
    motif_residues = [[i for i in range(1, int(fragsize)+1)] for fragsize in fragsize_str.split(sep)]
    return dict(zip(chain_str.split(sep), motif_residues))

def create_motif_contig(chain_str, fragsize_str, path_order, sep):
    '''create motif contig for diffusion'''
    chains = chain_str.split(sep)
    fragsizes = fragsize_str.split(sep)
    contig = [f"{chain}1-{length}" for chain, length in zip(chains, fragsizes)]
    contig = f"{sep}".join(sorted(contig, key=lambda x: path_order.index(x[0])))
    return contig

def create_pdbs(df:pd.DataFrame, output_dir, ligand, channel_path, preserve_channel_coordinates):
    '''write output pdbs'''
    filenames = []
    for _, row in df.iterrows():
        paths = row["poses"].split(",")
        models = row["model_num"].split(",")
        chains = row["chain_id"].split(",")
        frag_chains = [load_structure_from_pdbfile(path, all_models=True)[int(model)][chain] for path, model, chain in zip(paths, models, chains)]
        filename = ["".join([chain, model]) for chain, model in zip(chains, models)]
        filename = "-".join(sorted(filename, key=lambda x: row['path_order'].index(x[0]))) + ".pdb"
        filename = os.path.abspath(os.path.join(output_dir, filename))
        struct = Structure.Structure('out')
        struct.add(model := Model.Model(0))
        model.add(chain := Chain.Chain("Z"))
        for frag in frag_chains:
            model.add(frag)
        for lig in ligand:
            chain.add(lig)
        if channel_path and preserve_channel_coordinates:
            model.add(load_structure_from_pdbfile(channel_path)["Q"])
        elif channel_path:
            model = add_placeholder_to_pose(model, channel_path=channel_path, channel_chain="Q", ligand_chain='Z')
        save_structure_to_pdbfile(struct, filename)
        filenames.append(filename)
    return filenames

def fit_line_helix_centroids(points, window=4, trim_ends=0, outlier_z=2.5):
    """
    Helix-aware axis fit using sliding-window centroids (e.g., i..i+3 Cα).
    Returns:
      c : centroid on axis
      v : unit direction vector (N→C oriented)
      (s_min, s_max) : projection range of the ORIGINAL points along v
      r_est : RMS radial distance of ORIGINAL points to axis
    Args:
      points     : (N,3) array (ideally Cα atoms ordered from N to C)
      window     : sliding window length for centroids (4 is classic for α-helix)
      trim_ends  : optional number of residues to drop from each end before centroiding
      outlier_z  : remove centroid outliers farther than z * MAD from preliminary line
    """
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] < max(5, window+1):
        # fallback to your PCA if not enough points
        return fit_line_pca(P)

    # 0) optionally trim ragged ends before centroiding
    if trim_ends > 0:
        P_used = P[trim_ends: -trim_ends]
        if P_used.shape[0] < window+1:
            P_used = P  # not enough points to trim
    else:
        P_used = P

    # 1) sliding-window centroids (i..i+window-1)
    M = P_used.shape[0] - window + 1
    C = np.empty((M, 3), dtype=float)
    for i in range(M):
        C[i] = P_used[i:i+window].mean(axis=0)

    # 2) preliminary line from centroids via PCA
    Cc = C - C.mean(axis=0)
    _, _, Vt = np.linalg.svd(Cc, full_matrices=False)
    v0 = Vt[0]; v0 /= np.linalg.norm(v0)
    c0 = C.mean(axis=0)

    # 3) robust pass: remove centroid outliers (radial distance from preliminary line)
    #    distance to line through c0 with direction v0
    t = (C - c0) @ v0
    C_proj = c0 + np.outer(t, v0)
    resid = np.linalg.norm(C - C_proj, axis=1)
    med = np.median(resid)
    mad = np.median(np.abs(resid - med)) + 1e-9  # robust scale
    keep = resid <= (med + outlier_z * 1.4826 * mad)  # 1.4826 ≈ MAD→σ for Gauss

    if keep.sum() >= max(3, window):  # refit if we actually kept enough points
        Ck = C[keep]
        Ckc = Ck - Ck.mean(axis=0)
        _, _, Vt2 = np.linalg.svd(Ckc, full_matrices=False)
        v = Vt2[0]; v /= np.linalg.norm(v)
        c = Ck.mean(axis=0)
    else:
        v, c = v0, c0

    # 4) Orient axis N→C using original endpoints
    if ((P[-1] - P[0]) @ v) < 0:
        v = -v

    # 5) Project ORIGINAL points to get span and radius estimate
    X = P - c
    s_vals = X @ v

    s_first = float(s_vals[0])       # projection coordinate of N-terminal Cα
    p_first_proj = c + s_first * v   # 3D coords of projected first Cα


    return c, v, p_first_proj

def fit_line_pca(points):
    """
    Fit a 3D line (axis) through a set of coordinates using PCA (SVD).
    Returns:
      c      = centroid (a point on the axis)
      v      = unit direction vector along the helix axis
      (s_min, s_max) = scalar range of projections of the points along v
      r_est  = RMS radial distance of points to the fitted axis (≈ helix radius)
    """
    P = np.asarray(points, dtype=float)        # ensure input is a NumPy float array of shape (N,3)
    c = P.mean(axis=0)                         # compute centroid of all points
    X = P - c                                  # center the data around the centroid
    U, S, Vt = np.linalg.svd(X, full_matrices=False)  
    # perform SVD; right-singular vectors in Vt are principal directions
    
    v = Vt[0]                                  # take the first principal component (direction of max variance)
    v = v / np.linalg.norm(v)                  # normalize to unit vector
    
    s_vals = X @ v                             # project all centered points onto axis → scalar coords
    
    s_first = float(s_vals[0])       # projection coordinate of N-terminal Cα
    p_first_proj = c + s_first * v   # 3D coords of projected first Cα


    
    return c, v, p_first_proj         # return centroid, direction, span, and estimated radius

def add_placeholder_to_pose(pose: Structure, channel_path:str, channel_chain:str="Q", ligand_chain:str="Z", ignore_atoms:"list[str]"="H") -> Structure:
    '''
    add channel placehoder to pose
    '''
    if isinstance(ignore_atoms, str) or not ignore_atoms:
        ignore_atoms = [ignore_atoms]

    # load polyala:
    channel = load_structure_from_pdbfile(channel_path)

    channel_atoms = [atom for atom in channel.get_atoms() if atom.name not in ignore_atoms]


    frag_protein_atoms, frag_ligand_atoms = get_protein_and_ligand_atoms(pose, ligand_chain=ligand_chain, ignore_atoms=ignore_atoms)

    # calculate vector between fragment and ligand centroids
    frag_protein_centroid = np.mean(frag_protein_atoms, axis=0)
    frag_ligand_centroid = np.mean(frag_ligand_atoms, axis=0)
    vector_fragment = frag_ligand_centroid - frag_protein_centroid

    # calculate vector between CA of first and last residue of polyala
    channel_ca = [atom.get_coord() for atom in channel_atoms if atom.id == "CA"]

    channel_centroid, channel_vector, projected_Nterm = fit_line_helix_centroids(channel_ca)

    # calculate rotation between vectors
    r = Bio.PDB.rotmat(Bio.PDB.Vector(channel_vector), Bio.PDB.Vector(vector_fragment))

    # rotate polyala and translate into motif
    polyala_rotated = apply_rotation_to_pose(channel, projected_Nterm, r)
    polyala_translated = apply_translation_to_pose(polyala_rotated, frag_ligand_centroid - projected_Nterm)

    # change chain id of polyala and add into pose:
    if channel_chain in [chain.id for chain in pose.get_chains()]:
        raise KeyError(f"Chain {channel_chain} already found in pose. Try other chain name!")

    pose.add(polyala_translated[channel_chain])
    return pose

def apply_rotation_to_pose(pose: Structure, origin: "list[float]", r: "list[list[float]]") -> Structure:
    '''apply rotation'''
    for chain in pose:
        for residue in chain:
            for atom in residue:
                atom.coord = np.dot(r, atom.coord - origin) + origin
    return pose

def apply_translation_to_pose(pose: Structure, vector: "list[float]") -> Structure:
    '''apply translation'''
    for chain in pose:
        for residue in chain:
            for atom in residue:
                atom.coord += vector
    return pose

def get_protein_and_ligand_atoms(pose: Structure, ligand_chain, bb_atoms=None, ignore_atoms="H") -> "tuple[list]":
    '''AAA'''
    if isinstance(ignore_atoms, str) or not ignore_atoms:
        ignore_atoms = [ignore_atoms]

    if not bb_atoms:
        bb_atoms = ["CA", "C", "N", "O"]

    if isinstance(ligand_chain, str):
        # get all CA coords of protein:
        protein_atoms = np.array([atom.get_coord() for atom in get_protein_atoms(pose, ligand_chain) if atom.id in bb_atoms])

        # get Ligand Heavyatoms:
        ligand_atoms = np.array([atom.get_coord() for atom in pose[ligand_chain].get_atoms() if atom.id not in ignore_atoms])

    elif isinstance(ligand_chain, Bio.PDB.Chain.Chain):
        # get all CA coords of protein:
        protein_atoms = np.array([atom.get_coord() for atom in pose.get_atoms() if atom.id == "CA"])

        # get Ligand Heavyatoms:
        ligand_atoms = np.array([atom.get_coord() for atom in ligand_chain.get_atoms() if atom.id not in ignore_atoms])
    else: raise TypeError(f"Expected 'ligand' to be of type str or Bio.PDB.Chain.Chain, but got {type(ligand_chain)} instead.")
    return protein_atoms, ligand_atoms

def get_protein_atoms(pose: Structure, ligand_chain:str=None, atms:list=None) -> list:
    '''Selects atoms from a pose object. If ligand_chain is given, excludes all atoms in ligand_chain'''
    # define chains of pose
    chains = [x.id for x in pose.get_chains()]
    if ligand_chain:
        chains.remove(ligand_chain)

    # select specified atoms
    pose_atoms = [atom for chain in chains for atom in pose[chain].get_atoms()]
    if atms:
        pose_atoms = [atom for atom in pose_atoms if atom.id in atms]

    return pose_atoms

def create_ligand_dict(ligand_ids, theozyme):
    '''create a dictionary containing each ligand in the theozyme'''
    ligand = Chain.Chain('Z')
    lig_dict = {}
    if isinstance(ligand_ids, str):
        ligand_ids = [ligand_ids]
    for index, lig_id in enumerate(ligand_ids):
        resnum, chain = split_pdb_numbering(lig_id)
        if not chain in [chain.id for chain in theozyme.get_chains()] or not resnum in [res.id[1] for res in theozyme[0][chain].get_residues()]:
            logging.error(f'No ligand found in chain {chain} with residue number {resnum}. Please make sure the theozyme pdb is correctly formatted.')
            raise KeyError(f'No ligand found in chain {chain} with residue number {resnum}. Please make sure the theozyme pdb is correctly formatted.')
        else:
            # No idea why, but lig = theozyme[0][chain][resnum] does not work (I think because of heteroatoms)
            for res in theozyme[0][chain].get_residues():
                if res.id[1] == resnum:
                    lig = theozyme[0][chain][res.id]
            log_and_print(f"Found ligand in chain {chain} with residue number {resnum}.")
            lig.detach_parent() # pylint: disable=E0606
            # set occupancy to 1 (to prevent downstream issues)
            for atom in lig.get_atoms():
                atom.occupancy = 1
            lig.id = (lig.id[0], index+1, lig.id[2])
            ligand.add(lig)
            lig_dict.setdefault(chain, {})
            lig_dict[chain].update({resnum: lig})
    return lig_dict, ligand

def import_channel(path, chain, database):
    '''import channel placeholder'''
    if not path:
        path = os.path.join(database, "channel_placeholder", "helix_cone_long.pdb")
        chain = "Q"
    else:
        if not chain:
            raise KeyError("<channel_chain> must be provided if using a custom channel placeholder!")
    if os.path.isfile(path):
        channel = load_structure_from_pdbfile(path)
    else:
        raise RuntimeError(f"Could not find a PDB file at {path} to add as channel placeholder!")
    if not chain in [chain.id for chain in channel]:
        raise RuntimeError(f'No channel placeholder found in {path} on chain {chain}. Please make sure the channel pdb is correctly formatted.')
    channel = channel[chain]
    channel.detach_parent()
    channel.id = "Q"
    for index, residue in enumerate(channel.get_residues()):
        residue.id = (residue.id[0], index + 1, residue.id[2])

    return channel

def import_covalent_bonds(res_args, theozyme_residue, lig_dict, resnum, chain):
    '''import covalent bond information'''
    if hasattr(res_args, "covalent_bonds"):
        if not isinstance(res_args.covalent_bonds, list):
            setattr(res_args, "covalent_bonds", [res_args.covalent_bonds])
        covalent_bonds = extract_covalent_bond_info(res_args.covalent_bonds, theozyme_residue, lig_dict, resnum, chain)
    else:
        covalent_bonds = None
    return covalent_bonds


def build_frag_dict(rotlib, backbone_df):
    '''create a fragment dictionary'''
    frag_dict = {}
    relevant_columns = [col for col in rotlib.columns if col.startswith("chi") or col in ["probability", "phi_psi_occurrence", "AA"]]

    for pos, group in rotlib.groupby('rotamer_position'):
        pos_frags = []
        for idx, row in group.iterrows():
            df = pd.DataFrame(backbone_df.values, columns=backbone_df.columns)  # recreate backbone
            df.loc[pos - 1, relevant_columns] = row[relevant_columns].values
            df['frag_num'] = idx
            df['rotamer_pos'] = pos
            df['rotamer_score'] = row['rotamer_score']
            df['fragment_score'] = row['rotamer_score']
            df['backbone_score'] = 0
            df['backbone_probability'] = 0
            pos_frags.append(df)
        frag_dict[pos] = pd.concat(pos_frags, ignore_index=True)
        log_and_print(f"Created {len(pos_frags)} fragments for position {pos}.")

    return frag_dict


def main(args):
    '''run'''
    start = time.time()
    os.makedirs(args.working_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=os.path.join(args.working_dir, f"motif_library_{os.path.splitext(os.path.basename(args.theozyme_pdb))[0]}.log"))

    fragment_dir = os.path.join(args.working_dir, f"{args.output_prefix}_fragments" if args.output_prefix else "fragments")
    os.makedirs(fragment_dir, exist_ok=True)
    cmd = ''
    for key, value in vars(args).items():
        cmd += f'--{key} {value} '
    cmd = f'{sys.argv[0]} {cmd}'
    log_and_print(cmd)

    #import and prepare stuff
    riff_diff_dir = os.path.abspath(args.riff_diff_dir)
    database_dir = os.path.join(riff_diff_dir, "database")
    utils_dir = os.path.join(riff_diff_dir, "utils")
    rotlib_dir = os.path.join(database_dir, "bb_dep_rotlibs")

    theozyme = load_structure_from_pdbfile(args.theozyme_pdb, all_models=True)
    aa_alphabet = load_structure_from_pdbfile(os.path.join(database_dir, 'AA_alphabet.pdb'), all_models=True)

    # import ligands
    lig_dict, ligand = create_ligand_dict(args.ligands, theozyme)
    # import channel
    channel = import_channel(args.custom_channel_path, args.channel_chain, database_dir)
    channel_size = len([res for res in channel.get_residues()])
    save_structure_to_pdbfile(channel, channel_path := os.path.join(fragment_dir, "channel_placeholder.pdb"))

    # create output folders
    rotinfo_dir = os.path.join(fragment_dir, "rotamer_info")
    os.makedirs(rotinfo_dir, exist_ok=True)
    fraginfo_dir = os.path.join(fragment_dir, "fragment_info")
    os.makedirs(fraginfo_dir, exist_ok=True)

    assembly = []
    for resname in args.theozyme_resnums:

        # combine general and residue-specific args
        if hasattr(args, resname):
            res_args = combine_general_and_residue_specific_config(args, getattr(args, resname))
        else:
            res_args = args

        # check if residue exists in theozyme
        resnum, chain = split_pdb_numbering(resname)
        try:
            theozyme_residue = theozyme[0][chain][resnum]
        except Exception as exc:
            raise KeyError(f"Could not find residue {resnum} on chain {chain} in theozyme {args.theozyme_pdb}!") from exc

        # import covalent bonds
        covalent_bonds = import_covalent_bonds(res_args, theozyme_residue, lig_dict, resnum, chain)

        # define residue ids
        residue_identities = identify_residues_with_equivalent_func_groups(theozyme_residue, res_args.add_equivalent_func_groups)
        log_and_print(f"Looking for rotamers for these residues: {residue_identities}")


        if not res_args.pick_frags_from_db:


            #################################### BACKBONE ROTAMER FINDER ####################################
            # import backbone fragment
            fragment_path = res_args.fragment_pdb or os.path.join(database_dir, "backbone_frags", "7helix.pdb")
            backbone = load_structure_from_pdbfile(fragment_path, all_models=True)
            backbone = clean_input_backbone(backbone)
            # define positions for rotamer insertion
            if res_args.rotamer_positions == "auto":
                frag_pos_to_replace = [i+1 for i, _ in enumerate(backbone.get_residues())][1:-1]
            elif isinstance(res_args.rotamer_positions, list):
                frag_pos_to_replace = res_args.rotamer_positions
            elif isinstance(res_args.rotamer_positions, int):
                frag_pos_to_replace = [res_args.rotamer_positions]
            else:
                raise KeyError(f"<rotamer_position> for residue {resname} must be 'auto', a list of int, or a single int!")

            # extract data from backbone
            backbone_df = create_df_from_fragment(backbone, os.path.basename(fragment_path))

            # identify rotamers
            rotlibs = []
            log_and_print("Identifying rotamers...")
            for pos in frag_pos_to_replace:
                backbone_angles = extract_backbone_angles(backbone, pos)
                log_and_print(f"Position {pos} phi/psi angles: {backbone_angles['phi']} / {backbone_angles['psi']}.")
                rotlib = rotamers_for_backbone(residue_identities, rotlib_dir, backbone_angles["phi"], backbone_angles["psi"], res_args.prob_cutoff, res_args.rotamer_diff_to_best, 100, 2, 2)
                rotlib["rotamer_position"] = pos
                log_and_print(f"Found {len(rotlib.index)} rotamers for position {pos}.")
                rotlibs.append(rotlib)

            # combine rotamer libraries for each position
            rotlib = pd.concat(rotlibs).reset_index(drop=True)

            # rank rotamers
            rotlib = normalize_col(rotlib, 'log_prob', scale=False)
            rotlib = normalize_col(rotlib, 'log_occurrence', scale=False)
            rotlib = combine_normalized_scores(rotlib, 'rotamer_score', ['log_prob_normalized', 'log_occurrence_normalized'], [res_args.prob_weight, res_args.occurrence_weight], False, True)
            rotlib = rotlib.sort_values('rotamer_score', ascending=False).reset_index(drop=True)

            log_and_print(f"Found {len(rotlib.index)} rotamers in total.")
            rotlibcsv = os.path.join(rotinfo_dir, f'rotamers_{resname}_combined.csv')
            log_and_print(f"Writing phi/psi combinations to {rotlibcsv}.")
            rotlib.to_csv(rotlibcsv)

            # create dictionary containing dataframes for all fragments
            frag_dict = build_frag_dict(rotlib, backbone_df)

        else:

            #################################### FRAGMENT FINDER ####################################
            #sanity check command line input
            sec_dict, res_args.rot_sec_struct = create_frag_sec_struct_fraction_dict(res_args.frag_sec_struct_fraction, res_args.fragsize, res_args.rot_sec_struct)

            frag_pos_to_replace = define_rotamer_positions(res_args.rotamer_positions, res_args.fragsize)

            fraglib_path = os.path.join(database_dir, 'fraglib_noscore.pkl')
            if not os.path.isfile(fraglib_path):
                raise RuntimeError(f"Could not find fragment library at {fraglib_path}. Did you forget to download it?")

            rotlibs = []

            for residue_identity in residue_identities:
                #find rotamer library for given amino acid
                log_and_print(f"Importing backbone dependent rotamer library for residue {residue_identity} from {database_dir}")
                rotlib = return_residue_rotamer_library(rotlib_dir, residue_identity)
                rotlib = normalize_col(rotlib, 'log_prob', scale=False)
                rotlib = normalize_col(rotlib, 'log_occurrence', scale=False)
                rotlib = combine_normalized_scores(rotlib, 'rotamer_score', ['log_prob_normalized', 'log_occurrence_normalized'], [res_args.prob_weight, res_args.occurrence_weight], False, True)
                log_and_print(f"Identifying most probable rotamers for residue {residue_identity}")
                rotlib = identify_backbone_angles_suitable_for_rotamer(residue_identity, rotlib, rotinfo_dir, f'{resname}_', res_args.rot_sec_struct, res_args.phipsi_occurrence_cutoff, int(res_args.max_rotamers / len(residue_identities)), res_args.rotamer_diff_to_best, res_args.rotamer_chi_binsize, res_args.rotamer_phipsi_binsize, res_args.prob_cutoff)
                log_and_print(f"Found {len(rotlib.index)} phi/psi/chi combinations.")
                rotlibs.append(rotlib)

            rotlib = pd.concat(rotlibs).sort_values("rotamer_score", ascending=False).reset_index(drop=True)
            rotlib = normalize_col(rotlib, 'log_prob', scale=False)
            rotlib = normalize_col(rotlib, 'log_occurrence', scale=False)
            rotlib = combine_normalized_scores(rotlib, 'rotamer_score', ['log_prob_normalized', 'log_occurrence_normalized'], [res_args.prob_weight, res_args.occurrence_weight], False, True)
            rotlib = rotlib.sort_values('rotamer_score', ascending=False).reset_index(drop=True)
            rotlibcsv = os.path.join(rotinfo_dir, f'rotamers_{resname}_combined.csv')
            log_and_print(f"Writing phi/psi combinations to {rotlibcsv}.")
            rotlib.to_csv(rotlibcsv)

            # setting up jobstarters
            if args.jobstarter == "SbatchArray":
                jobstarter = SbatchArrayJobstarter(max_cores=args.cpus)
            elif args.jobstarter == "Local":
                jobstarter = LocalJobStarter(max_cores=args.cpus)
            else: raise KeyError("Jobstarter must be either 'SbatchArray' or 'Local'!")

            log_and_print("Identifying positions for rotamer insertion...")
            rotamer_positions = identify_positions_for_rotamer_insertion(fraglib_path, rotlib, res_args.rot_sec_struct, res_args.phi_psi_bin, os.path.join(fragment_dir, "rotamer_positions"), os.path.join(utils_dir, "identify_positions_for_rotamer_insertion.py"), resname, res_args.chi_std_multiplier, jobstarter=jobstarter)
            log_and_print(f"Found {len(rotamer_positions.index)} fitting positions.")
            log_and_print("Extracting fragments from rotamer positions...")
            combined = extract_fragments(rotamer_positions, fraglib_path, frag_pos_to_replace, res_args.fragsize, os.path.join(fragment_dir, f"{resname}_database_fragments"), os.path.join(utils_dir, "fragment_extraction.py"), jobstarter)
            frag_num = int(len(combined.index) / res_args.fragsize)
            log_and_print(f'Found {frag_num} fragments.')

            #filter fragments
            if frag_num == 0:
                log_and_print("Could not find fragments.")
                raise RuntimeError("Could not find fragments.")
            if sec_dict:
                combined = filter_frags_df_by_secondary_structure_content(combined, sec_dict)
                log_and_print(f"{int(len(combined) / res_args.fragsize)} fragments passed secondary structure filtering with filter {res_args.frag_sec_struct_fraction}.")
            if combined.empty:
                log_and_print('Could not find any fragments that fit criteria! Try adjusting filter values!')
                raise RuntimeError('Could not find any fragments that fit criteria! Try adjusting filter values!')

            log_and_print(f"Averaging and sorting fragments by fragment score with weights (backbone: {res_args.backbone_score_weight}, rotamer: {res_args.rotamer_score_weight}).")
            combined = sort_frags_df_by_score(combined, res_args.backbone_score_weight, res_args.rotamer_score_weight, res_args.fragsize)

            frag_dict = {}
            for pos, df in combined.groupby('rotamer_pos', sort=True):
                frag_dict[pos] = df
                log_and_print(f"Created {int(len(frag_dict[pos].index) / res_args.fragsize)} unique fragments for position {pos}.")
            combined = combined.groupby('frag_num', sort=False).mean(numeric_only=True)

            # visualize information about fragments
            violinplot_multiple_cols(dataframe=combined, cols=['fragment_score', 'backbone_score', 'rotamer_score'], titles=['fragment score', 'backbone score', 'rotamer score'], y_labels=['AU', 'AU', 'AU'], dims=[(-0.05, 1.05), (-0.05, 1.05), (-0.05, 1.05)], out_path=os.path.join(fraginfo_dir, f"{resname}_pre_clash_filter.png"), show_fig=False)
            del combined

        #################################### CREATE FRAGS, ATTACH ROTAMERS, FILTER ####################################

        fragments = Structure.Structure('fragments')
        frags_table = []
        frags_info = []
        frag_num = 0

        for pos, pos_df in frag_dict.items():

            log_and_print(f'Creating fragment structures, attaching rotamer, superpositioning with theozyme residue, calculating rmsd to all accepted fragments with cutoff {res_args.rmsd_cutoff} A for position {pos}.')

            pos_df["flipped"] = False
            # check if residues should be flipped to increase number of fragments
            flip = False
            if not res_args.not_flip_symmetric and any(x in tip_symmetric_residues() for x in residue_identities):
                flip = True

            check_dict = {"selected_frags": [], "selected_frag_dfs": [], "sc_clashes": [], "channel_clashes": 0, "bb_clashes": 0, "rmsd_fails": 0}
            for _, df in pos_df.groupby('frag_num', sort=False):
                # check if maximum number of fragments has been reached
                if len(check_dict["selected_frags"]) >= res_args.max_frags_per_residue / len(frag_dict):
                    break
                frag = create_fragment_from_df(df, pos, aa_alphabet)
                frag = align_to_sidechain(frag, frag[pos], theozyme_residue, False)
                check_dict = check_fragment(
                    frag=frag,
                    check_dict=check_dict,
                    frag_df=df,
                    ligand=ligand,
                    channel=channel if res_args.preserve_channel_coordinates else None,
                    rotamer_position=pos,
                    rmsd_cutoff=res_args.rmsd_cutoff,
                    bb_lig_clash_vdw_multiplier=res_args.bb_lig_clash_vdw_multiplier,
                    rot_lig_clash_vdw_multiplier=res_args.rot_lig_clash_vdw_multiplier,
                    channel_frag_clash_vdw_multiplier=res_args.channel_frag_clash_vdw_multiplier)

                if flip is True:
                    flipped_frag = copy.deepcopy(frag)
                    flipped_df = df.copy()

                    flipped_frag = align_to_sidechain(flipped_frag, frag[pos], theozyme_residue, True)
                    flipped_df["flipped"] = True
                    check_dict = check_fragment(
                        frag=flipped_frag,
                        check_dict=check_dict,
                        frag_df=flipped_df,
                        ligand=ligand,
                        channel=channel if res_args.preserve_channel_coordinates else None,
                        rotamer_position=pos,
                        rmsd_cutoff=res_args.rmsd_cutoff,
                        bb_lig_clash_vdw_multiplier=res_args.bb_lig_clash_vdw_multiplier,
                        rot_lig_clash_vdw_multiplier=res_args.rot_lig_clash_vdw_multiplier,
                        channel_frag_clash_vdw_multiplier=res_args.channel_frag_clash_vdw_multiplier)


            log_and_print(f"Discarded {check_dict['channel_clashes']} fragments that show clashes between backbone and channel placeholder with VdW multiplier {res_args.channel_frag_clash_vdw_multiplier}")
            log_and_print(f"Discarded {check_dict['bb_clashes']} fragments that show clashes between backbone and ligand with VdW multiplier {res_args.bb_lig_clash_vdw_multiplier}")
            log_and_print(f"Discarded {len(check_dict['sc_clashes'])} fragments that show clashes between sidechain and ligand with VdW multiplier {res_args.rot_lig_clash_vdw_multiplier}")
            if len(check_dict["sc_clashes"]) > 0:
                log_and_print(f"Atoms involved in sidechain-ligand clashes: {Counter(check_dict['sc_clashes'])}")
                log_and_print(f"You might want to try to adjust the <rot_lig_clash_vdw_multiplier> parameter (currently: {res_args.rot_lig_clash_vdw_multiplier}) for this residue: {chain}{resnum}")
            log_and_print(f"Discarded {check_dict['rmsd_fails']} fragments that did not pass RMSD cutoff of {res_args.rmsd_cutoff} to all other picked fragments")
            passed_frags = len(check_dict["selected_frags"])
            if passed_frags < 1:
                log_and_print(f"Could not find any passing fragments for {chain}{resnum} position {pos}!")
            log_and_print(f"Found {passed_frags} fragments for position {pos} of a maximum of {res_args.max_frags_per_residue / len(frag_dict)}.")

            for frag, df in zip(check_dict["selected_frags"], check_dict["selected_frag_dfs"]):
                rot = df.iloc[pos-1].squeeze()
                if covalent_bonds and theozyme_residue.get_resname() != rot['AA']:
                    covalent_bonds = [exchange_covalent(covalent_bond) for covalent_bond in covalent_bonds]
                if covalent_bonds and rot['flipped'] is True:
                    covalent_bonds = [flip_covalent(covalent_bond, rot["AA"]) for covalent_bond in covalent_bonds]

                updated_bonds = update_covalent_bonds(covalent_bonds, rot["AA"], pos, lig_dict)

                row = pd.Series({'model_num': frag_num, 'rotamer_pos': pos, 'rotamer_id': rot['AA'], 'AAs': df['AA'].to_list(), 'frag_length': len(df.index), 'backbone_score': df['backbone_score'].mean(), 'fragment_score': df['fragment_score'].mean(), 'rotamer_probability': rot['probability'], 'phi_psi_occurrence': rot['phi_psi_occurrence'], 'backbone_probability': df['backbone_probability'].mean(), 'covalent_bonds': updated_bonds, 'rotamer_score': df['rotamer_score'].mean()})
                model = Model.Model(frag_num)
                model.add(frag)
                if ligand:
                    model.add(ligand)
                    row['ligand_chain'] = ligand.id
                if res_args.preserve_channel_coordinates:
                    model.add(channel)
                    row['channel_chain'] = channel.id
                fragments.add(model)
                df['frag_num'] = frag_num
                frags_table.append(df)
                frags_info.append(row)
                frag_num += 1
            del check_dict

        log_and_print(f'Found {len(frags_info)} fragments that passed all filters.')

        #write fragment info to disk
        frags_table = pd.concat(frags_table)
        frags_table_path = os.path.join(fraginfo_dir, f'fragments_{resname}.csv')
        log_and_print(f'Writing fragment details to {frags_table_path}.')
        frags_table.to_csv(frags_table_path)

        #write multimodel fragment pdb to disk
        filename_pdb = os.path.join(fragment_dir, f'{resname}.pdb')
        log_and_print(f'Writing multimodel fragment pdb to {filename_pdb}.')
        save_structure_to_pdbfile(fragments, filename_pdb, multimodel=True)
        #utils.write_multimodel_structure_to_pdb(fragments, filename_pdb)

        #write output json to disk
        frags_info = pd.DataFrame(frags_info)
        frags_info['poses'] = os.path.abspath(filename_pdb)
        frags_info['poses_description'] = f'{resname}'
        filename_json = os.path.join(fragment_dir, f'{resname}.json')
        log_and_print(f'Writing output json to {filename_json}.')
        frags_info.to_json(filename_json)

        if res_args.pick_frags_from_db:
            combined = frags_table.groupby('frag_num', sort=False).mean(numeric_only=True)
            violinplot_multiple_cols(combined, cols=['fragment_score', 'backbone_score', 'rotamer_score'], titles=['fragment score', 'backbone score', 'rotamer score'], y_labels=['AU', 'AU', 'AU'], dims=[(-0.05, 1.05), (-0.05, 1.05), (-0.05, 1.05)], out_path=os.path.join(fraginfo_dir, f"{resname}_post_filter.png"), show_fig=False)

        assembly.append(frags_info)
        log_and_print(f"Done in {round(time.time() - start, 1)} seconds!")

    database_dir = os.path.join(riff_diff_dir, "database")
    utils_dir = os.path.join(riff_diff_dir, "utils")
    working_dir = os.path.join(args.working_dir, f"{args.output_prefix}_motif_library_assembly" if args.output_prefix else "motif_library_assembly")

    # check if output already exists
    out_json = os.path.join(args.working_dir, f'{args.output_prefix}_selected_paths.json' if args.output_prefix else "selected_paths.json")
    if os.path.exists(out_json):
        logging.error(f'Output file already exists at {out_json}!')
        raise RuntimeError(f'Output file already exists at {out_json}!')

    if args.jobstarter == "SbatchArray":
        jobstarter = SbatchArrayJobstarter(max_cores=args.cpus)
    elif args.jobstarter == "Local":
        jobstarter = LocalJobStarter(max_cores=args.cpus)
    else:
        logging.error(f"Jobstarter must be either 'SbatchArray' or 'Local', not {args.jobstarter}!")
        raise KeyError(f"Jobstarter must be either 'SbatchArray' or 'Local', not {args.jobstarter}!")

    in_df = pd.concat(assembly)

    ################## CLASH DETECTION ##########################

    log_and_print("Fragment selection completed, continuing with clash checks...")

    clash_dir = os.path.join(working_dir, 'clash_check')
    os.makedirs(clash_dir, exist_ok=True)

    grouped_df = in_df.groupby('poses', sort=False)
    counter = 0
    chains = []
    df_list = []
    structdict = {}

    for pose, pose_df in grouped_df:
        log_and_print(f'Working on {pose}...')
        pose_df['input_poses'] = pose_df['poses']
        pose_df['chain_id'] = list(string.ascii_uppercase)[counter]
        struct = load_structure_from_pdbfile(pose, all_models=True)
        model_dfs = []
        for index, series in pose_df.iterrows():
            chain = struct[series['model_num']]['A']
            chain.id = list(string.ascii_uppercase)[counter]
            model_dfs.append(series)
        pose_df = pd.DataFrame(model_dfs)
        structdict[struct.id] = struct
        filename = os.path.join(clash_dir, f'{struct.id}_rechained.pdb')
        struct.id = filename
        save_structure_to_pdbfile(pose=struct, save_path=filename, multimodel=True)
        pose_df['poses'] = os.path.abspath(filename)
        chains.append(list(string.ascii_uppercase)[counter])
        counter += 1
        pose_df["covalent_bonds"] = pose_df.apply(lambda row: replace_covalent_bonds_chain(row["chain_id"], row['covalent_bonds']), axis=1)
        df_list.append(pose_df)

    ligands = [lig for lig in struct[0]['Z'].get_residues()]
    for lig in ligands:
        lig.id = ("H", lig.id[1], lig.id[2])

    combined = pd.concat(df_list)
    grouped_df = combined.groupby('poses', sort=False)

    # generate every possible combination of input models
    num_models = [len(df.index) for _, df in grouped_df]
    num_combs = 1
    for i in num_models:
        num_combs *= i
    log_and_print(f'Generating {num_combs} possible fragment combinations...')

    init = time.time()

    #combinations = itertools.product(*[[row for _, row in pose_df.iterrows()] for _, pose_df in grouped_df])
    log_and_print('Performing pairwise clash detection...')
    ensemble_dfs = run_clash_detection(data=combined, directory=clash_dir, bb_multiplier=args.frag_frag_bb_clash_vdw_multiplier, sc_multiplier=args.frag_frag_sc_clash_vdw_multiplier, script_path=os.path.join(utils_dir, "clash_detection.py"), jobstarter=jobstarter)

    #calculate scores
    score_df = ensemble_dfs.groupby('ensemble_num', sort=False).mean(numeric_only=True)
    score_df = normalize_col(score_df, 'fragment_score', scale=True, output_col_name='ensemble_score')
    log_and_print(f'Found {len(score_df.index)} non-clashing ensembles.')

    plotpath = os.path.join(working_dir, "clash_filter.png")
    log_and_print(f"Plotting data at {plotpath}.")
    score_df_size = len(score_df.index)
    if score_df_size > 1000000:
        log_and_print(f"Downsampling dataframe for plotting because dataframe size is too big ({score_df_size} rows). Plotting only 1000000 random rows.")
        score_df_small = score_df.sample(n=1000000)
        violinplot_multiple_cols(score_df_small, cols=['ensemble_score', 'backbone_probability', 'rotamer_probability', 'phi_psi_occurrence'], titles=['ensemble_score', 'mean backbone\nprobability', 'mean rotamer\nprobability', 'mean phi psi\noccurrence'], y_labels=['score', 'probability', 'probability', 'probability'], out_path=plotpath, show_fig=False)
    else:
        violinplot_multiple_cols(score_df, cols=['ensemble_score', 'backbone_probability', 'rotamer_probability', 'phi_psi_occurrence'], titles=['ensemble_score', 'mean backbone\nprobability', 'mean rotamer\nprobability', 'mean phi psi\noccurrence'], y_labels=['score', 'probability', 'probability', 'probability'], out_path=plotpath, show_fig=False)

    # pre-filtering to reduce df size
    score_df_top = score_df.nlargest(args.max_top_out, 'ensemble_score')

    if args.max_random_out > 0:
        # drop all previously picked paths
        remaining_index = score_df.index.difference(score_df_top.index)
        score_df = score_df.loc[remaining_index]
        if not score_df.empty:
            sample_n = min(args.max_random_out, len(score_df))
            score_df = score_df.sample(n=sample_n, replace=False)
        score_df = pd.concat([score_df, score_df_top]) # combine with top ensembles
    else:
        score_df = score_df_top

    plotpath = os.path.join(working_dir, "pre_filter.png")
    log_and_print(f"Plotting selected ensemble results at {plotpath}.")
    violinplot_multiple_cols(score_df, cols=['ensemble_score', 'backbone_probability', 'rotamer_probability', 'phi_psi_occurrence'], titles=['ensemble_score', 'mean backbone\nprobability', 'mean rotamer\nprobability', 'mean phi psi\noccurrence'], y_labels=['score', 'probability', 'probability', 'probability'], out_path=plotpath, show_fig=False)

    post_clash = ensemble_dfs.merge(score_df['ensemble_score'], left_on='ensemble_num', right_index=True).sort_values('ensemble_num').reset_index(drop=True)

    log_and_print(f'Completed clash check in {round(time.time() - init, 0)} s.')
    if len(post_clash.index) == 0:
        log_and_print('No ensembles found! Try adjusting VdW multipliers or pick different fragments!')
        raise RuntimeError('No ensembles found! Try adjusting VdW multipliers or pick different fragments!')

    # sort ensembles by score
    log_and_print("Sorting ensembles by score...")
    post_clash = sort_dataframe_groups_by_column(df=post_clash, group_col="ensemble_num", sort_col="ensemble_score", ascending=False)
    post_clash["ensemble_num"] = post_clash.groupby("ensemble_num", sort=False).ngroup() + 1
    log_and_print("Sorting completed.")


    log_and_print("Creating paths out of ensembles...")
    post_clash['path_score'] = post_clash['ensemble_score']
    post_clash['path_num_matches'] = 0

    # filter for top ensembles to speed things up, since paths within an ensemble have the same score
    paths = ["".join(perm) for perm in itertools.permutations(chains)]
    #post_clash = sort_dataframe_groups_by_column(df=post_clash, group_col="ensemble_num", sort_col="path_score", ascending=False)
    dfs = [post_clash.assign(path_name=post_clash['ensemble_num'].astype(str)+"_" + p) for p in paths]
    path_df = pd.concat(dfs, ignore_index=True)
    log_and_print("Done creating paths.")

    pdb_dir = os.path.join(working_dir, "motif_library")
    os.makedirs(pdb_dir, exist_ok=True)

    if args.max_paths_per_ensemble:
        df_list = []
        for _, ensembles in path_df.groupby('ensemble_num', sort=False):
            df = sort_dataframe_groups_by_column(ensembles, group_col="path_name", sort_col="path_score", ascending=False, filter_top_n=args.max_paths_per_ensemble, randomize_ties=True)
            df_list.append(df)
        path_df = pd.concat(df_list)

    # select top n paths
    log_and_print(f"Selecting top {args.max_top_out} paths...")
    top_path_df = sort_dataframe_groups_by_column(df=path_df, group_col="path_name", sort_col="path_score", ascending=False, randomize_ties=True, filter_top_n=args.max_top_out)

    log_and_print(f"Found {int(len(top_path_df.index)/len(chains))} paths.")

    # select random paths
    if args.max_random_out > 0:
        # remove all selected paths
        log_and_print(f"Selecting random {args.max_top_out} paths...")

        path_df = (
            path_df
            .merge(top_path_df[['ensemble_num', 'path_name']], on=['ensemble_num', 'path_name'], how='left', indicator=True)
            .query('_merge == "left_only"')
            .drop(columns='_merge')
        )

        path_df["randomizer_score"] = 0
        random_path_df = sort_dataframe_groups_by_column(df=path_df, group_col="path_name", sort_col="randomizer_score", ascending=False, randomize_ties=True, filter_top_n=args.max_random_out)
        log_and_print(f"Found {int(len(random_path_df.index)/len(chains))} random paths.")

        selected_paths = pd.concat([random_path_df, top_path_df])
    else: selected_paths = top_path_df

    # create path dataframe
    log_and_print("Creating path dataframe...")
    aggregate = {'poses': concat_columns,
                 'chain_id': concat_columns,
                 'model_num': concat_columns,
                 'rotamer_pos':concat_columns, 
                 'frag_length': concat_columns, 
                 'path_score': 'mean', 
                 'backbone_probability': [("backbone_probability", concat_columns), ("backbone_probability_mean", "mean")],
                 'rotamer_probability': [("rotamer_probability", concat_columns), ("rotamer_probability_mean", "mean")],
                 'phi_psi_occurrence': [("phi_psi_occurrence", concat_columns), ("phi_psi_occurrence_mean", "mean")],
                 'covalent_bonds': concat_columns}

    selected_paths = selected_paths.groupby('path_name', sort=False).agg(aggregate).reset_index(names=["path_name"])
    selected_paths.columns = ['path_name', 'poses', 'chain_id', 'model_num', 'rotamer_pos', 'frag_length',
                               'path_score', 'backbone_probability', 'backbone_probability_mean', 'rotamer_probability',
                               'rotamer_probability_mean','phi_psi_occurrence', 'phi_psi_occurrence_mean', 'covalent_bonds']

    # create residue selections
    selected_paths["fixed_residues"] = selected_paths.apply(lambda row: split_str_to_dict(row['chain_id'], row['rotamer_pos'], sep=","), axis=1)
    selected_paths["fixed_residues"] = selected_paths.apply(lambda row: from_dict(row["fixed_residues"]), axis=1)
    selected_paths["motif_residues"] = selected_paths.apply(lambda row: create_motif_residues(row['chain_id'], row['frag_length'], sep=","), axis=1)
    selected_paths["motif_residues"] = selected_paths.apply(lambda row: from_dict(row["motif_residues"]), axis=1)
    selected_paths["ligand_motif"] = [from_dict({"Z":[i+1 for i, _ in enumerate(ligands)]}) for id in selected_paths.index]

    selected_paths["path_order"] = selected_paths['path_name'].str.split('_').str[-1]
    selected_paths["motif_contigs"] = selected_paths.apply(lambda row: create_motif_contig(row['chain_id'], row['frag_length'], row['path_order'], sep=","), axis=1)
    if channel_path:
        selected_paths["channel_contig"] = selected_paths.apply(lambda row: create_motif_contig("Q", str(channel_size), "Q", sep=","), axis=1)

    # combine multiple ligands into one for rfdiffusion
    ligand = copy.deepcopy(ligands)
    for lig in ligand:
        lig.resname = "LIG"

    lib_path = os.path.join(working_dir, "motif_library")
    log_and_print(f"Writing motif library .pdbs to {lib_path}")
    os.makedirs(lib_path, exist_ok=True)
    selected_paths["poses"] = create_pdbs(selected_paths, lib_path, ligand, channel_path, args.preserve_channel_coordinates)
    selected_paths["input_poses"] = selected_paths["poses"]
    selected_paths["poses_description"] = selected_paths.apply(lambda row: description_from_path(row['poses']), axis=1)


    log_and_print(f"Writing data to {out_json}")

    ligand_dir = os.path.join(working_dir, 'ligand')
    os.makedirs(ligand_dir, exist_ok=True)
    params_paths = []
    ligand_paths = []
    for index, ligand in enumerate(ligands):
        save_structure_to_pdbfile(ligand, lig_path:=os.path.abspath(os.path.join(ligand_dir, f"LG{index+1}.pdb")))
        lig_name = ligand.get_resname()
        ligand_paths.append(lig_path)
        if len(list(ligand.get_atoms())) > 3:
            # store ligand as .mol file for rosetta .molfile-to-params.py
            log_and_print("Running 'molfile_to_params.py' to generate params file for Rosetta.")
            lig_molfile = openbabel_fileconverter(input_file=lig_path, output_file=lig_path.replace(".pdb", ".mol2"), input_format="pdb", output_format=".mol2")
            cmd = f"{os.path.join(PROTFLOW_ENV, 'python')} {os.path.join(utils_dir, 'molfile_to_params.py')} -n {lig_name} -p {ligand_dir}/LG{index+1} {lig_molfile} --keep-names --clobber --chain=Z"
            LocalJobStarter().start(cmds=[cmd], jobname="moltoparams", output_path=ligand_dir)
            params_paths.append(lig_path.replace(".pdb", ".params"))
        else:
            log_and_print(f"Ligand at {lig_path} contains less than 4 atoms. No Rosetta Params file can be written for it.")

    if params_paths:
        selected_paths["params_path"] = ",".join(params_paths)
    if ligand_paths:
        selected_paths["ligand_path"] = ",".join(ligand_paths)

    # write output json
    selected_paths.to_json(out_json)

    violinplot_multiple_cols(selected_paths, cols=['backbone_probability_mean', 'phi_psi_occurrence_mean', 'rotamer_probability_mean'], titles=['mean backbone\nprobability', 'mean phi/psi\nprobability', 'mean rotamer\nprobability'], y_labels=['probability', 'probability', 'probability'], out_path=os.path.join(working_dir, "selected_paths_info.png"), show_fig=False)

    log_and_print('Done!')


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # mandatory input
    argparser.add_argument("--riff_diff_dir", default=".", type=str, help="Path to the riff_diff directory.")
    argparser.add_argument("--input_json", default=None, type=str, help="Alternative to CLI input. Mandatory for specifying covalent bonds.")
    argparser.add_argument("--theozyme_pdb", type=str, help="Path to pdbfile containing theozyme.")
    argparser.add_argument("--theozyme_resnums", nargs="+", help="List of residue numbers with chain information (e.g. 'A25 A38 B188') in theozyme pdb to find fragments for.")
    argparser.add_argument("--working_dir", type=str, help="Output directory")
    argparser.add_argument("--output_prefix", type=str, default=None, help="Prefix for all output files")
    argparser.add_argument("--ligands", nargs="+", type=str, help="List of ligands in theozyme pdb with chain information in the format 'X188 Z1'.")

    # important parameters
    argparser.add_argument("--fragment_pdb", type=str, default=None, help="Path to backbone fragment pdb. If not set, an idealized 7-residue helix fragment is used.")
    argparser.add_argument("--pick_frags_from_db", action="store_true", help="Select backbone fragments from database instead of providing a specific backbone manually. WARNING: This is much more time consuming!")
    argparser.add_argument("--custom_channel_path", type=str, default=None, help="Use a custom channel placeholder. Must be the path to a .pdb file.")
    argparser.add_argument("--channel_chain", type=str, default=None, help="Chain of the custom channel placeholder (if using a custom channel specified with <custom_channel_path>)")
    argparser.add_argument("--preserve_channel_coordinates", action="store_true", help="Copies channel from channel reference pdb without superimposing on moitf-substrate centroid axis. Useful when channel is present in catalytic array.")

    argparser.add_argument("--rotamer_positions", default="auto", nargs='+', help="Position in fragment the rotamer should be inserted, can either be int or a list containing first and last position (e.g. 2,6 if rotamer should be inserted at every position from 2 to 6). Recommended not to include N- and C-terminus! If auto, rotamer is inserted at every position (except N- and C-terminus).")
    argparser.add_argument("--rmsd_cutoff", type=float, default=0.5, help="Set minimum RMSD of output fragments. Increase to get more diverse fragments, but high values might lead to very long runtime or few fragments!")
    argparser.add_argument("--prob_cutoff", type=float, default=0.05, help="Do not return any phi/psi combinations with chi angle probabilities below this value")
    argparser.add_argument("--add_equivalent_func_groups", action="store_true", help="use ASP/GLU, GLN/ASN and VAL/ILE interchangeably.")

    # stuff you might want to adjust
    argparser.add_argument("--max_frags_per_residue", type=int, default=150, help="Maximum number of fragments that should be returned per active site residue.")
    #argparser.add_argument("--covalent_bonds", type=str, nargs="+", default=None, help="Add covalent bond(s) between residues and ligands in the form 'Res1-Res1Atom:Lig1-Lig1Atom,Res2-Res2Atom:Lig2-Lig2Atom'. Atom names should follow PDB numbering schemes. Example: 'A23-NE2:Z1-C1 A26-OE1:Z1-C11' for two covalent bonds between the NE2 atom of a Histidine at position A23 to C1 atom of ligand Z1 and the OE1 atom of a glutamic acid at A26 to C11 on the same ligand.")
    argparser.add_argument("--rot_lig_clash_vdw_multiplier", type=float, default=0.8, help="Multiplier for Van-der-Waals radii for clash detection between rotamer and ligand. Functional groups are not checked! Clash is detected if a distance between atoms < (VdW_radius_atom1 + VdW_radius_atom2)*multiplier.")
    argparser.add_argument("--bb_lig_clash_vdw_multiplier", type=float, default=1.0, help="Multiplier for Van-der-Waals radii for clash detection between fragment backbone and ligand. Clash is detected if a distance between atoms < (VdW_radius_atom1 + VdW_radius_atom2)*multiplier.")
    argparser.add_argument("--channel_frag_clash_vdw_multiplier", type=float, default=1.0, help="Multiplier for Van-der-Waals radii for clash detection between fragment backbone and channel placeholder. Clash is detected if a distance between atoms < (VdW_radius_atom1 + VdW_radius_atom2)*multiplier.")

    # options if running in fragment picking mode (<pick_frags_from_db> is set)
    argparser.add_argument("--fragsize", type=int, default=7, help="Size of output fragments. Only used if <pick_frags_from_db> is set.")
    argparser.add_argument("--rot_sec_struct", type=str, default=None, help="Limit fragments to secondary structure at rotamer position. Provide string of one-letter code of dssp secondary structure elements (B, E, G, H, I, T, S, -), e.g. 'HE' if rotamer should be in helices or beta strands. Only used if <pick_frags_from_db> is set.")
    argparser.add_argument("--frag_sec_struct_fraction", type=str, default=None, help="Limit to fragments containing at least fraction of residues with the provided secondary structure. If fragment should have at least 50 percent helical residues OR 60 percent beta-sheet, pass 'H:0.5,E:0.6'. Only used if <pick_frags_from_db> is set.")
    argparser.add_argument("--phipsi_occurrence_cutoff", type=float, default=0.5, help="Limit how common the phi/psi combination of a certain rotamer has to be. Value is in percent.")
    argparser.add_argument("--jobstarter", type=str, default="SbatchArray", help="Defines if jobs run locally or distributed on a cluster using a protflow jobstarter. Must be one of ['SbatchArray', 'Local'].")
    argparser.add_argument("--cpus", type=int, default=60, help="Defines how many cpus should be used for distributed computing.")
    argparser.add_argument("--rotamer_chi_binsize", type=float, default=10, help="Filter for diversifying found rotamers. Lower numbers mean more similar rotamers will be found. Similar rotamers will still be accepted if their backbone angles are different (if rotamer_phipsi_bin is set).")
    argparser.add_argument("--rotamer_phipsi_binsize", type=float, default=20, help="Filter for diversifying found rotamers. Lower numbers mean similar rotamers from more similar backbone angles will be accepted.")

    # stuff you probably don't want to touch
    argparser.add_argument("--phi_psi_bin", type=float, default=9.9, help="Binsize used to identify if fragment fits to phi/psi combination. Should not be above 10!")
    argparser.add_argument("--max_rotamers", type=int, default=80, help="maximum number of phi/psi combination that should be returned. Can be increased if not enough fragments are found downstream (e.g. because secondary structure filter was used, and there are not enough phi/psi combinations in the output that fit to the specified secondary structure.")
    argparser.add_argument("--rotamer_diff_to_best", type=float, default=1, help="Accept rotamers that have a probability not lower than this percentage of the most probable accepted rotamer. 1 means all rotamers will be accepted.")
    argparser.add_argument("--not_flip_symmetric", action="store_true", help="Do not flip tip symmetric residues (ARG, ASP, GLU, LEU, PHE, TYR, VAL).")
    argparser.add_argument("--prob_weight", type=float, default=2, help="Weight for rotamer probability importance when picking rotamers.")
    argparser.add_argument("--occurrence_weight", type=float, default=1, help="Weight for phi/psi-occurrence importance when picking rotamers.")
    argparser.add_argument("--backbone_score_weight", type=float, default=1, help="Weight for importance of fragment backbone score (boltzman score of number of occurrences of similar fragments in the database) when sorting fragments.")
    argparser.add_argument("--rotamer_score_weight", type=float, default=1, help="Weight for importance of rotamer score (combined score of probability and occurrence) when sorting fragments.")
    argparser.add_argument("--chi_std_multiplier", type=float, default=2, help="Multiplier for chi angle standard deviation to check if rotamer in database fits to desired rotamer.")


    # stuff you might want to adjust
    argparser.add_argument("--max_paths_per_ensemble", type=int, default=4, help="Maximum number of paths per ensemble (=same fragments but in different order)")
    argparser.add_argument("--frag_frag_bb_clash_vdw_multiplier", type=float, default=0.9, help="Multiplier for VanderWaals radii for clash detection inbetween backbone fragments. Clash is detected if distance_between_atoms < (VdW_radius_atom1 + VdW_radius_atom2)*multiplier.")
    argparser.add_argument("--frag_frag_sc_clash_vdw_multiplier", type=float, default=0.8, help="Multiplier for VanderWaals radii for clash detection between fragment sidechains and backbones. Clash is detected if distance_between_atoms < (VdW_radius_atom1 + VdW_radius_atom2)*multiplier.")
    argparser.add_argument("--fragment_score_weight", type=float, default=1, help="Maximum number of cpus to run on")
    argparser.add_argument("--max_top_out", type=int, default=100, help="Maximum number of top-ranked output paths")
    argparser.add_argument("--max_random_out", type=int, default=100, help="Maximum number of random-ranked output paths")


    arguments = argparser.parse_args()

    if arguments.input_json:
        config = import_input_json(arguments.input_json)

        if arguments.theozyme_resnums:
            theozyme_resnums = arguments.theozyme_resnums
        elif "theozyme_resnums" in config:
            theozyme_resnums = config["theozyme_resnums"]
        else:
            raise KeyError("<theozyme_resnums> is mandatory input!")

        # check for wrong config input (e.g. typos etc)
        for args_key in config:
            if not hasattr(arguments, args_key) and not args_key in theozyme_resnums:
                raise KeyError(f"{args_key} in input json is not a valid input!")

        # Merge config and CLI arguments
        for args_key, args_value in config.items():
            setattr(arguments, args_key, args_value)  # Add config values to args

    # check if mandatory fields are present
    for args_key in ["theozyme_pdb", "theozyme_resnums", "working_dir", "ligands"]:
        if not getattr(arguments, args_key):
            raise KeyError(f"<{args_key}> is mandatory input!")

    main(arguments)
