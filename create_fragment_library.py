import logging
import os
import sys
import copy
import json
import string
import itertools
import copy
import time
import Bio.PDB
from Bio.PDB import Structure, Model, Chain
import pandas as pd
import numpy as np
import shutil
from collections import Counter

# import dependencies
import Bio
from Bio.PDB import Structure, Model, Chain, Residue, Atom
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools

# protflow
from protflow.utils.biopython_tools import load_structure_from_pdbfile, save_structure_to_pdbfile
from protflow.utils.utils import vdw_radii
from protflow.jobstarters import SbatchArrayJobstarter, LocalJobStarter
from protflow.config import PROTFLOW_ENV
from protflow.utils.plotting import violinplot_multiple_cols, violinplot_multiple_cols_dfs
from protflow.utils.openbabel_tools import openbabel_fileconverter
from protflow.poses import description_from_path
from protflow.residues import ResidueSelection, from_dict


def split_pdb_numbering(pdbnum):
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
    symres = ["ARG", "ASP", "GLU", "LEU", "PHE", "TYR", "VAL"]
    return symres

def return_residue_rotamer_library(library_folder:str, residue_identity:str):
    '''
    finds the correct library for a given amino acid and drops not needed chi angles
    '''
    prefix = residue_identity.lower()
    rotlib = pd.read_csv(os.path.join(library_folder, f'{prefix}.bbdep.rotamers.lib'))
    if residue_identity in AAs_up_to_chi3():
        rotlib.drop(['chi4', 'chi4sig'], axis=1, inplace=True)
    elif residue_identity in AAs_up_to_chi2():
        rotlib.drop(['chi3', 'chi3sig', 'chi4', 'chi4sig'], axis=1, inplace=True)
    elif residue_identity in AAs_up_to_chi1():
        rotlib.drop(['chi2', 'chi2sig', 'chi3', 'chi3sig', 'chi4', 'chi4sig'], axis=1, inplace=True)

    return rotlib

def AAs_up_to_chi1():
    AAs = ['CYS', 'SER', 'THR', 'VAL']
    return AAs

def AAs_up_to_chi2():
    AAs = ['ASP', 'ASN', 'HIS', 'ILE', 'LEU', 'PHE', 'PRO', 'TRP', 'TYR']
    return AAs

def AAs_up_to_chi3():
    AAs = ['GLN', 'GLU', 'MET']
    return AAs

def AAs_up_to_chi4():
    AAs = ['ARG', 'LYS']
    return AAs

def num_chis_for_residue_id(res_id):
    if res_id in AAs_up_to_chi4():
        return 4
    if res_id in AAs_up_to_chi3():
        return 3
    elif res_id in AAs_up_to_chi2():
        return 2
    elif res_id in AAs_up_to_chi1():
        return 1
    else:
        return 0


def rama_plot(df, x_col, y_col, color_col, size_col, save_path=None):
    df_list = []
    for phi_psi, df in df.groupby([x_col, y_col]):
        top = df.sort_values(color_col, ascending=False).head(1)
        df_list.append(top)
    df = pd.concat(df_list)
    df = df[df[size_col] > 0]
    df[size_col] = df[size_col] * 100
    fig, ax = plt.subplots()
    norm_color = plt.Normalize(0, df[color_col].max())
    cmap = plt.cm.Blues
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
    #utils.create_output_dir_change_filename(output_dir, output_prefix + f'{residue_identity}_rama_pre_filtering')

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

    if rotamer_chi_binsize and rotamer_phipsi_binsize:
        rotlib = filter_rotlib_for_rotamer_diversity(rotlib, rotamer_chi_binsize, rotamer_phipsi_binsize)

    if max_output:
        rotlib = rotlib.head(max_output)

    if rotamer_diff_to_best:
        rotlib = rotlib[rotlib['probability'] >= rotlib['probability'].max() * (1 - rotamer_diff_to_best)]

    if rotlib.empty:
        raise RuntimeError('Could not find any rotamers that fit. Try setting different filter values!')

    rotlib.reset_index(drop=True, inplace=True)

    filename = os.path.join(output_dir, output_prefix + f'{residue_identity}_rama_post_filtering')
    rama_plot(rotlib, 'phi', 'psi', 'probability', 'phi_psi_occurrence', filename)

    return rotlib

def angle_difference(angle1, angle2):

    return min([abs(angle1 - angle2), abs(angle1 - angle2 + 360), abs(angle1 - angle2 - 360)])

def filter_rotlib_for_rotamer_diversity(rotlib:pd.DataFrame, rotamer_chi_binsize:float, rotamer_phipsi_binsize:float=None):
    accepted_rotamers = []
    if not rotamer_phipsi_binsize: rotamer_phipsi_binsize = 361
    chi_columns = [column for column in rotlib.columns if column.startswith('chi') and not column.endswith('sig')]

    for index, row in rotlib.iterrows():
        accept_list = []
        if len(accepted_rotamers) == 0:
            accepted_rotamers.append(row)
            continue
        for accepted_rot in accepted_rotamers:
            column_accept_list = []
            phipsi_difference = sum([angle_difference(row['phi'], accepted_rot['phi']), angle_difference(row['psi'], accepted_rot['psi'])])
            for column in chi_columns:
                #only accept rotamers that are different from already accepted ones
                if angle_difference(row[column], accepted_rot[column]) >= rotamer_chi_binsize / 2:
                    column_accept_list.append(True)
                else:
                    # if no rotamer_phipsi_binsize was defined or phi/psi angles are similar to accepted one, kick it out
                    if not rotamer_phipsi_binsize or phipsi_difference < rotamer_phipsi_binsize:
                        column_accept_list.append(False)
                    #still accept rotamers that are similar if their backbone angles are different enough
                    else:
                        column_accept_list.append(True)
            if True in column_accept_list:
                accept_list.append(True)
            else:
                accept_list.append(False)
        if set(accept_list) == {True}:
            accepted_rotamers.append(row)
    rotlib = pd.DataFrame(accepted_rotamers)
    return rotlib


def filter_rotamers_by_sec_struct(rotlib:pd.DataFrame, secondary_structure:str):
    filtered_list = []
    sec_structs = [*secondary_structure]
    #phi and psi angle range was determined from fragment library
    if "-" in sec_structs:
        phi_range = [x for x in range(-170, -39, 10)] + [x for x in range(60, 81, 10)]
        psi_range = [x for x in range(-180, -159, 10)] + [x for x in range(-40, 181, 10)]
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(filtered)
    if "B" in sec_structs:
        phi_range = [x for x in range(-170, -49, 10)]
        psi_range = [x for x in range(-180, -169, 10)] + [x for x in range(80, 181, 10)]
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(filtered)
    if "E" in sec_structs:
        phi_range = [x for x in range(-170, -59, 10)]
        psi_range = [x for x in range(-180, -169, 10)] + [x for x in range(90, 181, 10)]
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(filtered)
    if "G" in sec_structs:
        phi_range = [x for x in range(-130, -39, 10)] + [x for x in range(50, 71, 10)]
        psi_range = [x for x in range(-50, 41, 10)]
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(copy.deepcopy(filtered))
    if "H" in sec_structs:
        phi_range = [x for x in range(-100, -39, 10)]
        psi_range = [x for x in range(-60, 1, 10)]
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(filtered)
    if "I" in sec_structs:
        phi_range = [x for x in range(-140, -49, 10)]
        psi_range = [x for x in range(-80, 1, 10)]
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(filtered)
    if "S" in sec_structs:
        phi_range = [x for x in range(-170, -49, 10)] + [x for x in range(50, 111, 10)]
        psi_range = [x for x in range(-180, -149, 10)] + [x for x in range(-60, 181, 10)]
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(filtered)
    if "T" in sec_structs:
        phi_range = [x for x in range(-130, -40, 10)] + [x for x in range(40, 111, 10)]
        psi_range = [x for x in range(-60, 61, 10)] + [x for x in range(120, 151, 10)]
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

def is_unique(df_column):
    '''
    determines if all values in column are the same. quicker than nunique according to some guy on stackoverflow
    '''
    a = df_column.to_numpy()
    return (a[0] == a).all()

def check_for_chainbreaks(df, columname, fragsize):
    '''
    returns true if dataframe column is consistently numbered
    '''
    if df[columname].diff().sum() + 1 == fragsize:
        return True
    else:
        return False

def filter_frags_df_by_secondary_structure_content(frags_df, frag_sec_struct_fraction):

    frags_df_list = []
    for frag_num, df in frags_df.groupby('frag_num', sort=False):
        for sec_struct in frag_sec_struct_fraction:
            if df['ss'].str.contains(sec_struct).sum() / len(df.index) >= frag_sec_struct_fraction[sec_struct]:
                frags_df_list.append(df)
                break
    if len(frags_df_list) > 0:
        frags_df = pd.concat(frags_df_list)
        return frags_df
    else:
        return pd.DataFrame()

def filter_frags_df_by_score(frags_df, score_cutoff, scoretype, mode):

    frags_df_list = []
    for frag_num, df in frags_df.groupby('frag_num', sort=False):
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
    else:
        return pd.DataFrame()

def add_frag_to_structure(frag, structure):
    frag_num = len([model for model in structure.get_models()])
    model = Model.Model(frag_num)
    model.add(frag)
    structure.add(model)



    
def check_fragment(frag, check_dict, frag_df, ligand, channel, rotamer_position, rmsd_cutoff, bb_lig_clash_vdw_multiplier, rot_lig_clash_vdw_multiplier, channel_frag_clash_vdw_multiplier):

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
    else:
        return None
    
                

def atoms_of_functional_groups():
    return ["NH1", "NH2", "OD1", "OD2", "ND2", "NE", "SG", "OE1", "OE2", "NE2", "ND1", "NZ", "SD", "OG", "OG1", "NE1", "OH"]


def sort_frags_df_by_score(frags_df, backbone_score_weight, rotamer_score_weight, frag_length):

    # calculate ratio of identical backbone fragments
    frags_df['backbone_count'] = frags_df.groupby('frag_identifier')['frag_num'].transform('nunique')
    frags_df['backbone_probability'] = frags_df['backbone_count'] / frags_df['frag_num'].nunique()

    df_list = []
    frag_num = 0
    for identifier, unique_df in frags_df.groupby('frag_identifier', sort=False):
        rotamer_pos = unique_df.iloc[0, unique_df.columns.get_loc('rotamer_pos')]
        rotamer_id = unique_df.iloc[0, unique_df.columns.get_loc('rotamer_id')]
        phis = []
        psis = []
        omegas = []
        indices = [[index + pos for index in range(0, len(unique_df.index), frag_length)] for pos in range(0, frag_length)]
        for col in ['phi', 'psi', 'omega']:
            unique_df.loc[unique_df[col] <= -175, col] += 360
        for index, index_list in enumerate(indices):
            df = unique_df.iloc[index_list]                
            phis.append(df['phi'].median())
            psis.append(df['psi'].median())
            omegas.append(df['omega'].median())
        
        unique_df.iloc[0:frag_length, unique_df.columns.get_loc('AA')] = 'GLY'
        unique_df.iloc[rotamer_pos - 1, unique_df.columns.get_loc('AA')] = rotamer_id
        unique_df.iloc[0:frag_length, unique_df.columns.get_loc('phi')] = phis
        unique_df.iloc[0:frag_length, unique_df.columns.get_loc('psi')] = psis
        unique_df.iloc[0:frag_length, unique_df.columns.get_loc('omega')] = omegas
        unique_df.iloc[0:frag_length, unique_df.columns.get_loc('frag_num')] = frag_num
        
        frag_num += 1
        df_list.append(unique_df.iloc[0:frag_length])

    
    frags_df = pd.concat(df_list)
    frags_df.drop(['pdb', 'ss', 'frag_identifier'], axis=1, inplace=True)
    
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

def create_fragment_from_df(df:pd.DataFrame, rotamer_position, AA_alphabet):
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


    fragment = attach_rotamer_to_fragments(chain, rotamer_position, rotamer_id, chi_angles, prob, AA_alphabet)

    for res in chain:
        try:
            delattr(res, "internal_coord")
        except:
            continue
    
    delattr(fragment, "internal_coord")

    return fragment

def attach_rotamer_to_fragments(frag, rot_pos, rotamer_id, chi_angles, probability, AA_alphabet):
    to_mutate = frag[rot_pos]
    resid = to_mutate.id
    backbone_angles = extract_backbone_angles(frag, rot_pos)
    backbone_bondlengths = extract_backbone_bondlengths(frag, rot_pos)
    res = generate_rotamer(AA_alphabet, rotamer_id, resid, backbone_angles["phi"], backbone_angles["psi"], backbone_angles["omega"], backbone_angles["carb_angle"], backbone_angles["tau"], backbone_bondlengths["N_CA"], backbone_bondlengths["CA_C"], backbone_bondlengths["C_O"], chi_angles[0], chi_angles[1], chi_angles[2], chi_angles[3], probability)
    delattr(res, 'internal_coord')
    rotamer_on_fragments = attach_rotamer_to_backbone(frag, to_mutate, res)

    return rotamer_on_fragments

def attach_rotamer_to_backbone(fragment, fragment_residue, rotamer):
    fragment.detach_child(fragment_residue.id)
    to_mutate_atoms = []
    res_atoms = []
    for atom in ["N", "CA", "C"]:
        to_mutate_atoms.append(fragment_residue[atom])
        res_atoms.append(rotamer[atom])
    sup = Bio.PDB.Superimposer()
    sup.set_atoms(to_mutate_atoms, res_atoms)
    sup.rotran
    sup.apply(rotamer)
    rotamer.detach_child("O")
    rotamer.add(fragment_residue["O"])
    fragment.insert(rotamer.id[1]-1, rotamer)

    return fragment

def check_if_angle_in_bin(df, phi, psi, phi_psi_bin):

    df['phi_difference'] = df.apply(lambda row: angle_difference(row['phi'], phi), axis=1)
    df['psi_difference'] = df.apply(lambda row: angle_difference(row['psi'], psi), axis=1)

    df = df[(df['psi_difference'] < phi_psi_bin / 2) & (df['phi_difference'] < phi_psi_bin / 2)]
    df = df.drop(['phi_difference', 'psi_difference'], axis=1)
    return df


def identify_positions_for_rotamer_insertion(fraglib_path, rotlib, rot_sec_struct, phi_psi_bin, directory, script_path, prefix, chi_std_multiplier, jobstarter) -> pd.DataFrame:

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

    #sbatch_options = ["-c1", f'-e {directory}/{prefix}_identify_rotamer_positions.err -o {directory}/{prefix}_identify_rotamer_positions.out']

    cmds = [f"{os.path.join(PROTFLOW_ENV, 'python')} {script_path} --input_json {in_file} --fraglib {fraglib_path} --output_pickle {out_file} --phi_psi_bin {phi_psi_bin} --chi_std_multiplier {chi_std_multiplier}" for in_file, out_file in zip(in_filenames, out_filenames)]
    if rot_sec_struct:
        cmds = [cmd + f" --rot_sec_struct {rot_sec_struct}" for cmd in cmds]
    
    jobstarter.start(cmds=cmds, jobname="position_identification", output_path=directory)
    #sbatch_array_jobstarter(cmds=cmds, sbatch_options=sbatch_options, jobname="id_rot_pos", max_array_size=320, wait=True, remove_cmdfile=False, cmdfile_dir=directory)

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
    ''''''
    median = df[col].median()
    std = df[col].std()
    if not output_col_name:
        output_col_name = f"{col}_normalized"
    if df[col].nunique() == 1:
        df[output_col_name] = 0
        return df
    df[output_col_name] = (df[col] - median) / std
    if scale == True:
        df = scale_col(df=df, col=output_col_name, inplace=True)
    return df

def scale_col(df:pd.DataFrame, col:str, inplace=False) -> pd.DataFrame:
    #scale column to values between 0 and 1
    factor = df[col].max() - df[col].min()
    df[f"{col}_scaled"] = df[col] / factor
    df[f"{col}_scaled"] = df[f"{col}_scaled"] + (1 - df[f"{col}_scaled"].max())
    if inplace == True:
        df[col] = df[f"{col}_scaled"]
        df.drop(f"{col}_scaled", axis=1, inplace=True)
    return df


def create_frag_identifier(df, pos):
    dihedrals = ['phi', 'psi', 'omega']
    index = df['rotamer_index'].values[0]
    # Round the values and replace -180 with 180 using vectorized operations
    for col in dihedrals:
        rounded_col = f'{col}_rounded'
        df[rounded_col] = df[col].round(-1)
        df[rounded_col] = df[rounded_col].replace(-180, 180)  # Replace -180 with 180
    
    # Create the identifier using vectorized string operations
    identifier = (df['phi_rounded'].astype(str) + '_' + 
                  df['psi_rounded'].astype(str) + '_' + 
                  df['omega_rounded'].astype(str)).str.cat(sep='_') + f"_{index}_{pos}"
    return identifier

def has_consecutive_numbers(df:pd.DataFrame, fragsize):
    if df['position'].diff().sum() + 1 == fragsize: return True
    else: return False

def trim_indices(arr, upper_limit, lower_limit, n):
    # remove n indices as long as there are indices above upper_limit or below lower limit
    while arr[-1] > upper_limit:
        arr = arr[:-n] 
    while arr[0] < lower_limit:
        arr = arr[n:]
    return arr

def sample_random_ensembles(df: pd.DataFrame, group_col: str, n: int, seed: int | None = None) -> pd.DataFrame:
    """
    Return subset of df containing all rows for n randomly chosen groups from `group_col`.
    Sampling is without replacement. If n > #groups, it caps at #groups.
    """
    groups = df[group_col].drop_duplicates()
    if len(groups) == 0:
        raise RuntimeError(f"No groups found in column '{group_col}'.")
    n_eff = min(int(n), len(groups))
    picked = groups.sample(n=n_eff, random_state=seed)
    return df[df[group_col].isin(picked)].reset_index(drop=True)

def extract_fragments(rotamer_positions_df: pd.DataFrame, fraglib: pd.DataFrame, frag_pos_to_replace: list, fragsize: int, working_dir: str):
    '''
    frag_pos_to_replace: the position in the fragment the future rotamer should be inserted. central position recommended.
    residue_identity: only accept fragments with the correct residue identity at that position (recommended)
    rotamer_secondary_structure: accepts a string describing secondary structure (B: isolated beta bridge residue, E: strand, G: 3-10 helix, H: alpha helix, I: pi helix, T: turn, S: bend, -: none (not in the sense of no filter --> use None instead!)). e.g. provide EH if central atom in fragment should be a helix or strand.
    '''

    #choose fragments from fragment library that contain the positions selected above
    fragnum = 0
    frag_dict = {}
    rotamer_positions_df["temp_index_for_merge"] = rotamer_positions_df.index
    os.makedirs(working_dir, exist_ok=True)
    for pos in frag_pos_to_replace:
        # read in previous results if they exist
        if os.path.isfile(outfile := os.path.join(working_dir, f"fragments_{pos}.json")):
            log_and_print(f"Found previous results at {outfile}.")
            frags_df = pd.read_json(outfile)
            frag_dict[pos] = frags_df
            continue

        rotamer_positions_df["temp_pos_for_merge"] = pos
        
        # define start and end index for each position
        index_starts = rotamer_positions_df.index - pos + 1
        index_ends = index_starts + fragsize

        # create range between start and end
        all_values = []
        for start, end in zip(index_starts, index_ends):
            if start >= 0 and end <= fraglib.index.max():
                all_values.extend(range(start, end))  # Append the range to the list
        indices = np.array(all_values)

        # check if indices are below 0 or above fraglib size
        #indices = trim_indices(indices, fraglib.index.max(), 0, fragsize)

        log_and_print(indices.max())
        log_and_print(indices.min())

        # extract all indices
        df = fraglib.loc[indices]
        df['temp_index_for_merge'] = df.index
        df.reset_index(drop=True, inplace=True)

        # group by fragsize
        group_key = df.index // fragsize
        # check if each group has a unique pdb (to prevent taking frags with res from different pdbs) and consecutive numbers (to prevent using frags with chainbreaks)
        valid_groups = df.groupby(group_key).filter(lambda x: x['pdb'].nunique() == 1 and has_consecutive_numbers(x, fragsize))
        if valid_groups.empty:
            log_and_print(f"No fragments passed for position {pos}!")
            frag_dict[pos] = pd.DataFrame()
            continue

        valid_groups.reset_index(drop=True, inplace=True)

        # Create a new identifier column based on the valid groups
        group_key = valid_groups.index // fragsize
        valid_group_ids = valid_groups.groupby(group_key).ngroup() + 1  # Create unique group identifiers

        # Assign the identifier to the filtered DataFrame
        valid_groups['frag_num'] = valid_group_ids + fragnum

        # number residues within fragments from 1 to fragsize
        valid_groups['residue_numbers'] = valid_groups.groupby(group_key).cumcount() + 1

        # merge with rotamer_positions df on index column and correct position (to make sure not merging with rotamer at another position!)
        valid_groups_merge = valid_groups.copy()
        preserve_cols = ['temp_index_for_merge', 'temp_pos_for_merge', 'AA', 'probability', 'phi_psi_occurrence', 'rotamer_score', 'rotamer_index']
        valid_groups_merge = valid_groups_merge[['temp_index_for_merge', 'residue_numbers', 'frag_num']].merge(rotamer_positions_df[preserve_cols], left_on=['temp_index_for_merge', 'residue_numbers'], right_on=['temp_index_for_merge', 'temp_pos_for_merge'])

        # modify df for downstream processing
        valid_groups_merge.rename({'AA': 'rotamer_id'}, inplace=True, axis=1)
        valid_groups_merge.drop(['temp_index_for_merge', 'residue_numbers'], axis=1, inplace=True)
        valid_groups_merge['phi_psi_occurrence'] = valid_groups_merge['phi_psi_occurrence'] * 100

        # merge back with all residues from fragments
        frags_df = valid_groups.merge(valid_groups_merge, on="frag_num")

        # set non-rotamer positions to 0
        frags_df.loc[frags_df['residue_numbers'] != pos, 'probability'] = None
        frags_df.loc[frags_df['residue_numbers'] != pos, 'phi_psi_occurrence'] = None

        # define rotamer position
        frags_df['rotamer_pos'] = pos

        # assign identifier based on phi/psi/omega angles, rotamer id and rotamer position
        frags_l = []
        for _, df in frags_df.groupby("frag_num", sort=False):
            df["frag_identifier"] = create_frag_identifier(df,pos)
            frags_l.append(df)
        frags_df = pd.concat(frags_l)

        # add fragnum so that fragment numbering is continous for next position
        fragnum = fragnum + frags_df['frag_num'].max()

        # drop identifier column
        frags_df.drop(['temp_index_for_merge', 'temp_pos_for_merge'], axis=1, inplace=True)
        
        # write frags_df to json
        frags_df.to_json(outfile)

        frag_dict[pos] = frags_df

    return frag_dict

def is_unique(s):
    '''
    determines if all values in column are the same. quicker than nunique according to some guy on stackoverflow
    '''
    a = s.to_numpy()
    return (a[0] == a).all()


def extract_backbone_angles(chain, resnum:int):
    '''
    takes a biopython chain and extracts phi/psi/omega angles of specified residue
    '''
    #convert to internal coordinates, read phi/psi angles
    #chain = copy.deepcopy(chain)
    chain.internal_coord
    chain.atom_to_internal_coordinates()
    phi = chain[resnum].internal_coord.get_angle("phi")
    psi = chain[resnum].internal_coord.get_angle("psi")
    omega = chain[resnum].internal_coord.get_angle("omg")
    carb_angle = round(chain[resnum].internal_coord.get_angle("N:CA:C:O"), 1)
    tau = round(chain[resnum].internal_coord.get_angle("tau"), 1)
    if not phi == None:
        phi = round(phi, 1)
    if not psi == None:
        psi = round(psi, 1)
    if not omega == None:
        omega = round(omega, 1)
    return {"phi": phi, "psi": psi, "omega": omega, "carb_angle": carb_angle, "tau": tau}

def extract_backbone_bondlengths(chain, resnum:int):
    '''
    takes a biopython chain and extracts phi/psi/omega angles of specified residue
    '''
    #convert to internal coordinates, read phi/psi angles
    #chain = copy.deepcopy(chain)
    chain.internal_coord
    chain.atom_to_internal_coordinates()
    n_ca = round(chain[resnum].internal_coord.get_length("N:CA"), 3)
    ca_c = round(chain[resnum].internal_coord.get_length("CA:C"), 3)
    c_o = round(chain[resnum].internal_coord.get_length("C:O"), 3)
    return {"N_CA": n_ca, "CA_C": ca_c, "C_O": c_o}



def generate_rotamer(AAalphabet_structure, residue_identity:str, res_id, phi:float=None, psi:float=None, omega:float=None, carb_angle:float=None, tau:float=None, N_CA_length:float=None, CA_C_length:float=None, C_O_length:float=None, chi1:float=None, chi2:float=None, chi3:float=None, chi4:float=None, rot_probability:float=None):
    '''
    builds a rotamer from residue identity, phi/psi/omega/chi angles
    '''
    alphabet = copy.deepcopy(AAalphabet_structure)
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
    #drop all rows that do not contain the rotamer, return a series --> might cause issues downstream if more than one rotamer on a fragment
    rotamer_pos = df.dropna(subset = ['probability']).squeeze()
    return(rotamer_pos)

def align_to_sidechain(entity, entity_residue_to_align, sidechain, flip_symmetric:bool=True):
    '''
    aligns an input structure (bb_fragment_structure, resnum_to_align) to a sidechain residue (sc_structure, resnum_to_alignto)
    '''
    sc_residue_identity = sidechain.get_resname()

    #superimpose structures based on specified atoms
    bbf_atoms = atoms_for_func_group_alignment(entity_residue_to_align)
    sc_atoms = atoms_for_func_group_alignment(sidechain)
    if flip_symmetric == True and sc_residue_identity in tip_symmetric_residues():
        order = [1, 0, 2]
        sc_atoms = [sc_atoms[i] for i in order]
    sup = Bio.PDB.Superimposer()
    sup.set_atoms(sc_atoms, bbf_atoms)
    sup.rotran
    sup.apply(entity)

    return entity

def identify_his_central_atom(histidine, ligand):
    HIS_NE2 = histidine["NE2"]
    HIS_ND1 = histidine["ND1"]
    lig_atoms =  [atom for atom in ligand.get_atoms()]
    NE2_distance = min([HIS_NE2 - atom for atom in lig_atoms])
    ND1_distance = min([HIS_ND1 - atom for atom in lig_atoms])
    if NE2_distance < ND1_distance:
        his_central_atom = "NE2"
    else:
        his_central_atom = "ND1"
    return his_central_atom

def rotate_histidine_fragment(entity, degrees, theozyme_residue, his_central_atom, ligand):
    if his_central_atom == "auto":
        if not ligand:
            raise RuntimeError(f"Ligand is required if using <flip_histidines='auto'>!")
        his_central_atom = identify_his_central_atom(theozyme_residue, ligand)
    if his_central_atom == "NE2":
        half = theozyme_residue["ND1"].coord + 0.5 * (theozyme_residue["CG"].coord - theozyme_residue["ND1"].coord)
    elif his_central_atom == "ND1":
        half = theozyme_residue["NE2"].coord + 0.5 * (theozyme_residue["CD2"].coord - theozyme_residue["NE2"].coord)
    entity = rotate_entity_around_axis(entity, theozyme_residue[his_central_atom].coord, half, degrees)
    
    return entity

def rotate_phenylalanine_fragment(entity, degrees, theozyme_residue):

    center = theozyme_residue["CZ"].coord - 0.5 * (theozyme_residue["CZ"].coord - theozyme_residue["CG"].coord)
    vector_A = theozyme_residue["CG"].coord - theozyme_residue["CZ"].coord
    vector_B = theozyme_residue["CE1"].coord - theozyme_residue["CZ"].coord
    N = np.cross(vector_A, vector_B)
    N = N / np.linalg.norm(N)
    point = center + N

    entity = rotate_entity_around_axis(entity, center, point, degrees)
    
    return entity

def rotate_entity_around_axis(entity, coords_1, coords_2, angle):
    rotation_axis = coords_1 - coords_2
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    angle_radians = np.radians(angle)
    for atom in entity.get_atoms():
        array = atom.coord - coords_1
        rotated_array = rotate_array_around_vector(array, rotation_axis, angle_radians)
        atom.coord = rotated_array + coords_1
    return entity

def rotation_matrix(V, X):
    K = np.array([[0, -V[2], V[1]],
                  [V[2], 0, -V[0]],
                  [-V[1], V[0], 0]])
    I = np.identity(3)
    R = I + np.sin(X)*K + (1-np.cos(X))*np.dot(K,K)
    return R

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
    func_groups = {
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
    return func_groups


def clean_input_backbone(entity: Structure):
    chains = [chain for chain in entity.get_chains()]
    models = [model for model in entity.get_models()]
    if len(chains) > 1 or len(models):
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
    if not phi == None:
        phi = round(phi, -1)
    if not psi == None:
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

    pdbnames = [fragment_name for res in backbone.get_residues()]
    resnames = [res.resname for res in backbone.get_residues()]
    pos_list = [res.id[1] for res in backbone.get_residues()]
    problist = [float("nan") for res in backbone.get_residues()]
    phi_angles = [extract_backbone_angles(backbone, res.id[1])['phi'] for res in backbone.get_residues()]
    psi_angles = [extract_backbone_angles(backbone, res.id[1])['psi'] for res in backbone.get_residues()]
    omega_angles = [extract_backbone_angles(backbone, res.id[1])['omega'] for res in backbone.get_residues()]
    CA_x_coords_list = [(round(res["CA"].get_coord()[0], 3)) for res in backbone.get_residues()]
    CA_y_coords_list = [(round(res["CA"].get_coord()[1], 3)) for res in backbone.get_residues()]
    CA_z_coords_list = [(round(res["CA"].get_coord()[2], 3)) for res in backbone.get_residues()]
    C_x_coords_list = [(round(res["C"].get_coord()[0], 3)) for res in backbone.get_residues()]
    C_y_coords_list = [(round(res["C"].get_coord()[1], 3)) for res in backbone.get_residues()]
    C_z_coords_list = [(round(res["C"].get_coord()[2], 3)) for res in backbone.get_residues()]
    N_x_coords_list = [(round(res["N"].get_coord()[0], 3)) for res in backbone.get_residues()]
    N_y_coords_list = [(round(res["N"].get_coord()[1], 3)) for res in backbone.get_residues()]
    N_z_coords_list = [(round(res["N"].get_coord()[2], 3)) for res in backbone.get_residues()]
    O_x_coords_list = [(round(res["O"].get_coord()[0], 3)) for res in backbone.get_residues()]
    O_y_coords_list = [(round(res["O"].get_coord()[1], 3)) for res in backbone.get_residues()]
    O_z_coords_list = [(round(res["O"].get_coord()[2], 3)) for res in backbone.get_residues()]

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
    atom = covalent_bond[0]
    exchange_dict = {"OE1": "OD1", "OE2": "OD2", "CD1": "CG1", "CD2": "CG2", "NE2": "ND2", "OD1": "OE1", "OD2": "OE2", "CG1": "CD1", "CG2": "CD2", "ND2": "NE2"}
    try:
        new_atom = exchange_dict[atom]
    except:
        new_atom = atom
    covalent_bond[0] = new_atom
    return covalent_bond

def flip_covalent(covalent_bond, residue):
    atom = covalent_bond[0]
    exchange_dict = {
        "GLU": {"OE1": "OE2", "OE2": "OE1"},
        "ASP": {"OD1": "OD2", "OD2": "OD1"},
        "VAL": {"CD1": "CD2", "CD2": "CD1"},
        "LEU": {"CG1": "CG2", "CG2": "CG1"},
        "ARG": {"NH1": "NH2", "NH2": "NH1"}     
        }
    try:
        new_atom = exchange_dict[residue][atom]
    except:
        new_atom = atom
    covalent_bond[0] = new_atom
    return covalent_bond

def log_and_print(string: str): 
    logging.info(string)
    print(string)
    return string

def combine_normalized_scores(df: pd.DataFrame, name:str, scoreterms:list, weights:list, normalize:bool=False, scale:bool=False):
    if not len(scoreterms) == len(weights):
        raise RuntimeError(f"Number of scoreterms ({len(scoreterms)}) and weights ({len(weights)}) must be equal!")
    df[name] = sum([df[col]*weight for col, weight in zip(scoreterms, weights)]) / sum(weights)
    if df[name].nunique() == 1:
        df[name] = 0
        return df
    
    if normalize == True:
        df = normalize_col(df, name, False)
        df.drop(name, axis=1, inplace=True)
        df.rename(columns={f'{name}_normalized': name}, inplace=True)
    if scale == True:
        df[name] = df[name] / df[name].max()
        df = scale_col(df, name, True)
    return df

def define_rotamer_positions(rotamer_positions, fragsize):
    if rotamer_positions == "auto":
        rotamer_positions = int(fragsize / 2)
        if rotamer_positions * 2 == fragsize:
            rotamer_positions = [rotamer_positions, rotamer_positions + 1]
        else:
            rotamer_positions = [rotamer_positions + 1]
        return rotamer_positions
    elif isinstance(rotamer_positions, list):
        if any(int(pos) > fragsize for pos in rotamer_positions):
            raise KeyError(f"Rotamer positions must be smaller than fragment size!")
        return [int(pos) for pos in rotamer_positions]
    elif isinstance(rotamer_positions, int):
        if rotamer_positions > fragsize:
            raise KeyError(f"Rotamer positions must be smaller than fragment size!")
        return [rotamer_positions]
    else:
        raise KeyError(f"rotamer_positions must be 'auto', a list of int or a single int value, not {type(rotamer_positions)}!")

def update_covalent_bonds(covalent_bonds:str, rotamer_id:str, rotamer_position:int, ligand_dict:dict):
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
    if argument in residue_config:
        return residue_config[argument]
    else:
        return getattr(args, argument)
    
def combine_general_and_residue_specific_config(args, residue_args):

    # create new args from general and residue-specific dict
    res_args = copy.deepcopy(args)
    
    # check for wrong config input (e.g. typos etc)
    for key in residue_args:
        if not hasattr(res_args, key) and not key == "covalent_bonds" or key in ["ligands", "theozyme_resnums"]:
            raise KeyError(f"{key} in input json is not a valid input!")

    # Merge config and CLI arguments
    for key, value in residue_args.items():
        setattr(res_args, key, value)  # Add config values to args

    return res_args

def create_frag_sec_struct_fraction_dict(frag_sec_struct_fraction: str, fragsize: int, rot_sec_struct: str):
    if frag_sec_struct_fraction:
        sec_structs = frag_sec_struct_fraction.split(',')
        sec_dict = {}
        for i in sec_structs:
            sec, frac = i.split(':')
            frac = float(frac)
            if frac > 1 or frac < 0:
                logging.error(f'Fraction for secondary structure {sec} must be a value between 0 and 1, but it is {frac}!')
                raise ValueError(f'Fraction for secondary structure {sec} must be a value between 0 and 1, but it is {frac}!')
            if (fragsize - frac * fragsize) < 1 and sec != rot_sec_struct and rot_sec_struct != None:
                logging.error(f"If limiting all fragment residues to secondary structure {sec}, it is not possible that the rotamer has secondary structure {rot_sec_struct}!")
                raise KeyError(f"If limiting all fragment residues to secondary structure {sec}, it is not possible that the rotamer has secondary structure {rot_sec_struct}!")
            elif (fragsize - frac * fragsize) < 1 and rot_sec_struct == None and len(sec_structs) == 1:
                log_and_print(f"Setting <rot_sec_struct> to {sec} because all residues in fragment have to have secondary structure {sec}!")
                rot_sec_struct = sec
            sec_dict[sec] = float(frac)
    else:
        sec_dict = None
    
    return sec_dict, rot_sec_struct

def replace_covalent_bonds_chain(chain:str, covalent_bonds:str=None) -> ResidueSelection:
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
    log_and_print(f'Calculating pairwise compatibility maps...')

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
    ensemble_df.sort_values("ensemble_num", inplace=True)
    ensemble_df.reset_index(drop=True, inplace=True)

    #log_and_print("Saving results...")
    #ensemble_df.to_pickle(os.path.join(directory, "ensembles.pkl"))

    return ensemble_df

def sort_dataframe_groups_by_column(df:pd.DataFrame, group_col:str, sort_col:str, method="mean", ascending:bool=True, filter_top_n:int=None, randomize_ties:bool=False) -> pd.DataFrame:
    # group by group column and calculate mean values
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
    non_null_elements = group.dropna().astype(str)
    return ','.join(non_null_elements) if not non_null_elements.empty else None

def split_str_to_dict(key_str, value_str, sep):
    return dict(zip(key_str.split(sep), [list(i) for i in value_str.split(sep)]))

def create_motif_residues(chain_str, fragsize_str, sep:str=","):
    motif_residues = [[i for i in range(1, int(fragsize)+1)] for fragsize in fragsize_str.split(sep)]
    return dict(zip(chain_str.split(sep), motif_residues))

def replace_covalent_bonds_chain(chain:str, covalent_bonds:str=None) -> ResidueSelection:
    if not isinstance(covalent_bonds, str):
        return None
    covalent_bonds = covalent_bonds.split(",")
    new_cov_bonds = []
    for cov_bond in covalent_bonds:
        rot, lig = cov_bond.split(":")
        rechained = rot.split("_")[0][:-1] + chain + "_" + rot.split("_")[1]
        new_cov_bonds.append(":".join([rechained, lig]))
    return ",".join(new_cov_bonds)

def create_motif_contig(chain_str, fragsize_str, path_order, sep):
    chains = chain_str.split(sep)
    fragsizes = fragsize_str.split(sep)
    contig = [f"{chain}1-{length}" for chain, length in zip(chains, fragsizes)]
    contig = f"{sep}".join(sorted(contig, key=lambda x: path_order.index(x[0])))
    return contig

def create_pdbs(df:pd.DataFrame, output_dir, ligand, channel_path, preserve_channel_coordinates):
    filenames = []
    for i, row in df.iterrows():
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
            model = add_polyala_to_pose(model, polyala_path=channel_path, polyala_chain="Q", ligand_chain='Z')
        save_structure_to_pdbfile(struct, filename)
        filenames.append(filename)
    return filenames


def add_polyala_to_pose(pose: Structure, polyala_path:str, polyala_chain:str="Q", ligand_chain:str="Z", ignore_atoms:"list[str]"=["H"]) -> Structure:
    '''
    
    '''
    # load polyala:
    polyala = load_structure_from_pdbfile(polyala_path)

    # determine polyala chain name (only first chain will be considered)
    pa_chain = [x.id for x in polyala.get_chains()][0]

    pa_atoms = [atom for atom in polyala.get_atoms() if atom.name not in ignore_atoms]


    frag_protein_atoms, frag_ligand_atoms = get_protein_and_ligand_atoms(pose, ligand_chain=ligand_chain, ignore_atoms=ignore_atoms)

    # calculate vector between fragment and ligand centroids
    frag_protein_centroid = np.mean(frag_protein_atoms, axis=0)
    frag_ligand_centroid = np.mean(frag_ligand_atoms, axis=0)
    vector_fragment = frag_ligand_centroid - frag_protein_centroid

    # calculate vector between CA of first and last residue of polyala
    polyala_ca = [atom.get_coord() for atom in pa_atoms if atom.id == "CA"]
    ca1, ca2 = polyala_ca[0], polyala_ca[-1]
    vector_polyala = ca2 - ca1

    # calculate rotation between vectors
    R = Bio.PDB.rotmat(Bio.PDB.Vector(vector_polyala), Bio.PDB.Vector(vector_fragment))

    # rotate polyala and translate into motif
    polyala_rotated = apply_rotation_to_pose(polyala, ca1, R)
    polyala_translated = apply_translation_to_pose(polyala_rotated, frag_ligand_centroid - ca1)

    # change chain id of polyala and add into pose:
    if polyala_chain in [chain.id for chain in pose.get_chains()]: raise KeyError(f"Chain {polyala_chain} already found in pose. Try other chain name!")
    if pa_chain != polyala_chain: polyala_translated[pa_chain].id = polyala_chain
    pose.add(polyala_translated[polyala_chain])
    return pose

def apply_rotation_to_pose(pose: Structure, origin: "list[float]", R: "list[list[float]]") -> Structure:
    ''''''
    for chain in pose:
        for residue in chain:
            for atom in residue:
                atom.coord = np.dot(R, atom.coord - origin) + origin
    return pose

def apply_translation_to_pose(pose: Structure, vector: "list[float]") -> Structure:
    ''''''
    for chain in pose:
        for residue in chain:
            for atom in residue:
                atom.coord += vector
    return pose

def get_protein_and_ligand_atoms(pose: Structure, ligand_chain, bb_atoms=["CA", "C", "N", "O"], ignore_atoms=["H"]) -> "tuple[list]":
    '''AAA'''
    if type(ligand_chain) == type("str"):
        # get all CA coords of protein:
        protein_atoms = np.array([atom.get_coord() for atom in get_protein_atoms(pose, ligand_chain) if atom.id in bb_atoms])

        # get Ligand Heavyatoms:
        ligand_atoms = np.array([atom.get_coord() for atom in pose[ligand_chain].get_atoms() if atom.id not in ignore_atoms])

    elif type(ligand_chain) == Bio.PDB.Chain.Chain:
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
    if ligand_chain: chains.remove(ligand_chain)

    # select specified atoms
    pose_atoms = [atom for chain in chains for atom in pose[chain].get_atoms()]
    if atms: pose_atoms = [atom for atom in pose_atoms if atom.id in atms]

    return pose_atoms

def create_ligand_dict(ligand_ids, theozyme):
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
            lig.detach_parent()
            # set occupancy to 1 (to prevent downstream issues)
            for atom in lig.get_atoms():
                atom.occupancy = 1
            lig.id = (lig.id[0], index+1, lig.id[2])
            ligand.add(lig)
            lig_dict[chain] = {resnum: lig}
    return lig_dict, ligand

def import_channel(path, chain, database):
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

    if hasattr(res_args, "covalent_bonds"):
        if not isinstance(res_args.covalent_bonds, list):
            setattr(res_args, "covalent_bonds", [res_args.covalent_bonds])
        covalent_bonds = extract_covalent_bond_info(res_args.covalent_bonds, theozyme_residue, lig_dict, resnum, chain)
    else:
        covalent_bonds = None
    return covalent_bonds


def build_frag_dict(rotlib, backbone_df):
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
    AA_alphabet = load_structure_from_pdbfile(os.path.join(database_dir, 'AA_alphabet.pdb'), all_models=True)

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
        except:
            raise KeyError(f"Could not find residue {resnum} on chain {chain} in theozyme {args.theozyme_pdb}!")
        
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
            log_and_print(f"Identifying rotamers...")
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
            log_and_print(f"Importing fragment library from {fraglib_path}")
            
            fraglib = import_fragment_library(fraglib_path)

            rotlibs = []

            for residue_identity in residue_identities:
                #find rotamer library for given amino acid
                log_and_print(f"Importing backbone dependent rotamer library for residue {residue_identity} from {database_dir}")
                rotlib = return_residue_rotamer_library(rotlib_dir, residue_identity)
                rotlib = normalize_col(rotlib, 'log_prob', scale=False)
                rotlib = normalize_col(rotlib, 'log_occurrence', scale=False)
                rotlib = combine_normalized_scores(rotlib, 'rotamer_score', ['log_prob_normalized', 'log_occurrence_normalized'], [res_args.prob_weight, res_args.occurrence_weight], False, True)
                log_and_print(f"Identifying most probable rotamers for residue {residue_identity}")
                rotlib = identify_backbone_angles_suitable_for_rotamer(residue_identity, rotlib, rotinfo_dir, f'{resname}_', res_args.rot_sec_struct, res_args.phipsi_occurrence_cutoff, int(res_args.max_phi_psis / len(residue_identities)), res_args.rotamer_diff_to_best, res_args.rotamer_chi_binsize, res_args.rotamer_phipsi_binsize, res_args.prob_cutoff)
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
            if args.jobstarter == "SbatchArray": jobstarter = SbatchArrayJobstarter(max_cores=args.cpus)
            elif args.jobstarter == "Local": jobstarter = LocalJobStarter(max_cores=args.cpus)
            else: raise KeyError("Jobstarter must be either 'SbatchArray' or 'Local'!")

            log_and_print(f"Identifying positions for rotamer insertion...")
            rotamer_positions = identify_positions_for_rotamer_insertion(fraglib_path, rotlib, res_args.rot_sec_struct, res_args.phi_psi_bin, os.path.join(fragment_dir, "rotamer_positions"), os.path.join(utils_dir, "identify_positions_for_rotamer_insertion.py"), resname, res_args.chi_std_multiplier, jobstarter=jobstarter)
            log_and_print(f"Found {len(rotamer_positions.index)} fitting positions.")
            log_and_print(f"Extracting fragments from rotamer positions...")
            frag_dict = extract_fragments(rotamer_positions, fraglib, frag_pos_to_replace, res_args.fragsize, os.path.join(fragment_dir, f"{resname}_database_fragments"))
            frag_num = int(sum([len(frag_dict[pos].index) for pos in frag_dict]) / res_args.fragsize)
            log_and_print(f'Found {frag_num} fragments.')

            #filter fragments
            for pos in frag_dict:
                frag_nums = int(len(frag_dict[pos].index) / res_args.fragsize)
                log_and_print(f'Found {frag_nums} fragments for position {pos}.')
                if frag_nums == 0:
                    frag_dict.pop(pos)
                    log_and_print(f"Could not find fragments for position {pos}.")
                    continue
                if sec_dict:
                    if os.path.isfile(filtered_frags := os.path.join(fragment_dir, f"{resname}_database_fragments", f"{pos}_sec_struct_filtered_frags.json")):
                        log_and_print(f"Found previous filter results at {filtered_frags}.")
                        frag_dict[pos] = pd.read_json(filtered_frags)
                    else:
                        frag_dict[pos] = filter_frags_df_by_secondary_structure_content(frag_dict[pos], sec_dict)
                        frag_dict[pos].to_json(filtered_frags)
                    log_and_print(f"{int(len(frag_dict[pos]) / res_args.fragsize)} fragments passed secondary structure filtering with filter {res_args.frag_sec_struct_fraction} for position {pos}.")
                if frag_dict[pos].empty:
                    frag_dict.pop(pos)
                    log_and_print(f"Could not find fragments for position {pos}.")

            if len(frag_dict) == 0:
                raise RuntimeError('Could not find any fragments that fit criteria! Try adjusting filter values!')
        

            combined = pd.concat([frag_dict[pos] for pos in frag_dict])
            log_and_print(f"Averaging and sorting fragments by fragment score with weights (backbone: {res_args.backbone_score_weight}, rotamer: {res_args.rotamer_score_weight}).")
            if os.path.isfile(averaged_frags := os.path.join(fragment_dir, f"{resname}_database_fragments", "averaged_sorted_frags.json")):
                log_and_print(f"Found previous averaging results at {averaged_frags}.")
                combined = pd.read_json(averaged_frags)
            else:
                combined = sort_frags_df_by_score(combined, res_args.backbone_score_weight, res_args.rotamer_score_weight, res_args.fragsize)
                combined.to_json(averaged_frags)

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

        for pos in frag_dict:

            log_and_print(f'Creating fragment structures, attaching rotamer, superpositioning with theozyme residue, calculating rmsd to all accepted fragments with cutoff {res_args.rmsd_cutoff} A for position {pos}.')

            frag_dict[pos]["flipped"] = False
            # check if residues should be flipped to increase number of fragments
            flip = False
            if not res_args.not_flip_symmetric and any(x in tip_symmetric_residues() for x in residue_identities):
                flip = True

            check_dict = {"selected_frags": [], "selected_frag_dfs": [], "sc_clashes": [], "channel_clashes": 0, "bb_clashes": 0, "rmsd_fails": 0}
            for _, df in frag_dict[pos].groupby('frag_num', sort=False):
                # check if maximum number of fragments has been reached
                if len(check_dict["selected_frags"]) >= res_args.max_frags_per_residue / len(frag_dict):
                    break
                frag = create_fragment_from_df(df, pos, AA_alphabet)
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
                
                if flip == True:
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
                if covalent_bonds and rot['flipped'] == True:
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
            del(check_dict)

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

    if args.jobstarter == "SbatchArray": jobstarter = SbatchArrayJobstarter(max_cores=args.cpus)
    elif args.jobstarter == "Local": jobstarter = LocalJobStarter(max_cores=args.cpus)
    else: 
        logging.error(f"Jobstarter must be either 'SbatchArray' or 'Local', not {args.jobstarter}!")
        raise KeyError(f"Jobstarter must be either 'SbatchArray' or 'Local', not {args.jobstarter}!")

    in_df = pd.concat(assembly)

    ################## CLASH DETECTION ##########################
    # safety for later
    args.max_out = 0
    if args.max_top_out:
        args.max_out += args.max_top_out
    if args.max_random_out:
        args.max_out += args.max_random_out

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
    num_models = [len(df.index) for group, df in grouped_df]
    num_combs = 1
    for i in num_models: num_combs *= i
    log_and_print(f'Generating {num_combs} possible fragment combinations...')

    init = time.time()

    #combinations = itertools.product(*[[row for _, row in pose_df.iterrows()] for _, pose_df in grouped_df])
    log_and_print(f'Performing pairwise clash detection...')
    ensemble_dfs = run_clash_detection(data=combined, directory=clash_dir, bb_multiplier=args.frag_frag_bb_clash_vdw_multiplier, sc_multiplier=args.frag_frag_sc_clash_vdw_multiplier, script_path=os.path.join(utils_dir, "clash_detection.py"), jobstarter=jobstarter)

    #calculate scores
    score_df = ensemble_dfs.groupby('ensemble_num', sort=False).mean(numeric_only=True)
    score_df = normalize_col(score_df, 'fragment_score', scale=True, output_col_name='ensemble_score')
    log_and_print(f'Found {len(score_df.index)} non-clashing ensembles.')

    plotpath = os.path.join(working_dir, "clash_filter.png")
    log_and_print(f"Plotting data at {plotpath}.")
    violinplot_multiple_cols(score_df, cols=['backbone_probability', 'rotamer_probability', 'phi_psi_occurrence'], titles=['mean backbone\nprobability', 'mean rotamer\nprobability', 'mean phi psi\noccurrence'], y_labels=['probability', 'probability', 'probability'], out_path=plotpath, show_fig=False)

    # pre-filtering to reduce df size
    score_df.sort_values('ensemble_score', ascending=True, inplace=True)
    score_df_top = score_df.head(args.max_top_out)
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

    post_clash = ensemble_dfs.merge(score_df['ensemble_score'], left_on='ensemble_num', right_index=True).sort_values('ensemble_num').reset_index(drop=True)

    log_and_print(f'Completed clash check in {round(time.time() - init, 0)} s.')
    if len(post_clash.index) == 0:
        log_and_print(f'No ensembles found! Try adjusting VdW multipliers or pick different fragments!')
        raise RuntimeError(f'No ensembles found! Try adjusting VdW multipliers or pick different fragments!')

    # sort ensembles by score
    log_and_print("Sorting ensembles by score...")
    post_clash = sort_dataframe_groups_by_column(df=post_clash, group_col="ensemble_num", sort_col="ensemble_score", ascending=False)
    post_clash["ensemble_num"] = post_clash.groupby("ensemble_num", sort=False).ngroup() + 1 
    log_and_print("Sorting completed.")


    log_and_print("Creating paths out of ensembles...")
    post_clash['path_score'] = post_clash['ensemble_score']
    post_clash['path_num_matches'] = 0

    # Generate all path orderings once
    paths = ["".join(perm) for perm in itertools.permutations(chains)]

    if args.sample_ensembles:
        # NEW: randomly sample ensembles first (before exploding to paths)
        log_and_print(f"Randomly selecting {args.sample_ensembles} ensembles (keeping ALL paths for each).")
        post_clash = sample_random_ensembles(post_clash, "ensemble_num", args.sample_ensembles, seed=args.sample_seed)

        # Important: do NOT apply any top-N filtering here; we want all paths for the sampled ensembles.
        dfs = [post_clash.assign(path_name=post_clash['ensemble_num'].astype(str) + "_" + p) for p in paths]
        path_df = pd.concat(dfs, ignore_index=True)
    else:
        # Original behavior: keep only the top ensembles (by group mean score) to bound the path explosion.
        post_clash = sort_dataframe_groups_by_column(
            df=post_clash, group_col="ensemble_num", sort_col="path_score",
            ascending=False, filter_top_n=args.max_out
        )
        dfs = [post_clash.assign(path_name=post_clash['ensemble_num'].astype(str) + "_" + p) for p in paths]
        path_df = pd.concat(dfs, ignore_index=True)

    log_and_print("Done creating paths.")

    pdb_dir = os.path.join(working_dir, "motif_library")
    os.makedirs(pdb_dir, exist_ok=True)

    if args.sample_ensembles:
        if args.max_paths_per_ensemble:
            log_and_print("NOTE: Ignoring --max_paths_per_ensemble because --sample_ensembles is set (keeping ALL paths for each sampled ensemble).")
        if args.max_out is not None:
            log_and_print("NOTE: Ignoring --max_out because --sample_ensembles is set (keeping ALL paths for each sampled ensemble).")
        top_path_df = path_df
    else:
        if args.max_paths_per_ensemble:
            df_list = []
            for _, ensembles in path_df.groupby('ensemble_num', sort=False):
                df = sort_dataframe_groups_by_column(
                    ensembles, group_col="path_name", sort_col="path_score",
                    ascending=False, filter_top_n=args.max_paths_per_ensemble, randomize_ties=True
                )
                df_list.append(df)
            path_df = pd.concat(df_list)

        log_and_print(f"Selecting top {args.max_out} paths...")
        top_path_df = sort_dataframe_groups_by_column(
            df=path_df, group_col="path_name", sort_col="path_score",
            ascending=False, filter_top_n=args.max_out
    )
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
    log_and_print(f"Creating path dataframe...")
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
        ligand_pdbfile = save_structure_to_pdbfile(ligand, lig_path:=os.path.abspath(os.path.join(ligand_dir, f"LG{index+1}.pdb")))
        lig_name = ligand.get_resname()
        ligand_paths.append(lig_path)
        if len(list(ligand.get_atoms())) > 3:
            # store ligand as .mol file for rosetta .molfile-to-params.py
            log_and_print(f"Running 'molfile_to_params.py' to generate params file for Rosetta.")
            lig_molfile = openbabel_fileconverter(input_file=lig_path, output_file=lig_path.replace(".pdb", ".mol2"), input_format="pdb", output_format=".mol2")
            cmd = f"{os.path.join(PROTFLOW_ENV, 'python')} {os.path.join(utils_dir, 'molfile_to_params.py')} -n {lig_name} -p {ligand_dir}/LG{index+1} {lig_molfile} --keep-names --clobber --chain=Z"
            LocalJobStarter().start(cmds=[cmd], jobname="moltoparams", output_path=ligand_dir)
            params_paths.append(lig_path.replace(".pdb", ".params"))
        else:
            log_and_print(f"Ligand at {ligand_pdbfile} contains less than 4 atoms. No Rosetta Params file can be written for it.")

    if params_paths:
        selected_paths["params_path"] = ",".join(params_paths)
    if ligand_paths:
        selected_paths["ligand_path"] = ",".join(ligand_paths)

    # write output json
    selected_paths.to_json(out_json)

    violinplot_multiple_cols(selected_paths, cols=['backbone_probability_mean', 'phi_psi_occurrence_mean', 'rotamer_probability_mean'], titles=['mean backbone\nprobability', 'mean phi/psi\nprobability', 'mean rotamer\nprobability'], y_labels=['probability', 'probability', 'probability'], out_path=os.path.join(working_dir, "selected_paths_info.png"), show_fig=False)

    log_and_print(f'Done!')


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
    argparser.add_argument("--ligands", nargs="+", type=str, help="List of ligands in the format 'X188 Z1'.")
    
    # important parameters
    argparser.add_argument("--fragment_pdb", type=str, default=None, help="Path to backbone fragment pdb. If not set, an idealized 7-residue helix fragment is used.")
    argparser.add_argument("--pick_frags_from_db", action="store_true", help="Select backbone fragments from database instead of providing a specific backbone manually. WARNING: This is much more time consuming!")
    argparser.add_argument("--custom_channel_path", type=str, default=None, help="Use a custom channel placeholder. Must be the path to a .pdb file.")
    argparser.add_argument("--channel_chain", type=str, default=None, help="Chain of the custom channel placeholder (if using a custom channel specified with <custom_channel_path>)")
    argparser.add_argument("--preserve_channel_coordinates", action="store_true", help="Copies channel from channel reference pdb without superimposing on moitf-substrate centroid axis. Useful when channel is present in catalytic array.")

    argparser.add_argument("--rotamer_positions", default="auto", nargs='+', help="Position in fragment the rotamer should be inserted, can either be int or a list containing first and last position (e.g. 2,6 if rotamer should be inserted at every position from 2 to 6). Recommended not to include N- and C-terminus! If auto, rotamer is inserted at every position when using backbone finder and in the central location when using fragment finder.")
    argparser.add_argument("--rmsd_cutoff", type=float, default=0.3, help="Set minimum RMSD of output fragments. Increase to get more diverse fragments, but high values might lead to very long runtime or few fragments!")
    argparser.add_argument("--prob_cutoff", type=float, default=0.05, help="Do not return any phi/psi combinations with chi angle probabilities below this value")
    argparser.add_argument("--add_equivalent_func_groups", action="store_true", help="use ASP/GLU, GLN/ASN and VAL/ILE interchangeably.")

    # stuff you might want to adjust
    argparser.add_argument("--max_frags_per_residue", type=int, default=100, help="Maximum number of fragments that should be returned per active site residue.")
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
    argparser.add_argument("--rotamer_chi_binsize", type=float, default=None, help="Filter for diversifying found rotamers. Lower numbers mean more similar rotamers will be found. Similar rotamers will still be accepted if their backbone angles are different. Recommended value: 15")
    argparser.add_argument("--rotamer_phipsi_binsize", type=float, default=None, help="Filter for diversifying found rotamers. Lower numbers mean similar rotamers from more similar backbone angles will be accepted. Recommended value: 50")

    # stuff you probably don't want to touch
    argparser.add_argument("--phi_psi_bin", type=float, default=9.9, help="Binsize used to identify if fragment fits to phi/psi combination. Should not be above 10!")
    argparser.add_argument("--max_phi_psis", type=int, default=15, help="maximum number of phi/psi combination that should be returned. Can be increased if not enough fragments are found downstream (e.g. because secondary structure filter was used, and there are not enough phi/psi combinations in the output that fit to the specified secondary structure.")
    argparser.add_argument("--rotamer_diff_to_best", type=float, default=0.7, help="Accept rotamers that have a probability not lower than this percentage of the most probable accepted rotamer. 1 means all rotamers will be accepted.")
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
    argparser.add_argument("--max_top_out", type=int, default=500, help="Maximum number of top-ranked output paths")
    argparser.add_argument("--max_random_out", type=int, default=500, help="Maximum number of random-ranked output paths")

    # ensemble sampling args
    argparser.add_argument("--sample_ensembles", type=int, default=None, help="If set, randomly select this many ensembles (by ensemble_num) and output ALL paths from those ensembles.")
    argparser.add_argument("--sample_seed", type=int, default=None, help="Optional RNG seed for --sample_ensembles (reproducible sampling).")

    args = argparser.parse_args()

    if args.input_json:
        config = import_input_json(args.input_json)

        if args.theozyme_resnums:
            theozyme_resnums = args.theozyme_resnums
        elif "theozyme_resnums" in config:
            theozyme_resnums = config["theozyme_resnums"]
        else:
            raise KeyError(f"<theozyme_resnums> is mandatory input!")
    
        # check for wrong config input (e.g. typos etc)
        for key in config:
            if not hasattr(args, key) and not key in theozyme_resnums:
                raise KeyError(f"{key} in input json is not a valid input!")

        # Merge config and CLI arguments
        for key, value in config.items():
            setattr(args, key, value)  # Add config values to args
        
    # check if mandatory fields are present
    for key in ["theozyme_pdb", "theozyme_resnums", "working_dir", "ligands"]:
        if not getattr(args, key):
            raise KeyError(f"<{key}> is mandatory input!")

    main(args)
