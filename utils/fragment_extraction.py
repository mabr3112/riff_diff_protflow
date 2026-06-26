import os
import numpy as np
import pandas as pd


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
        rotamer = int(group['rotamer_index'].iloc[0])
        return (dihedral_seq, rotamer, pos)

    # Apply group-wise identifier creation
    identifiers = df.groupby('frag_num', sort=False).apply(make_identifier)
    identifiers.name = 'frag_identifier'
    identifiers = identifiers.reset_index()

    # Merge the identifier back to the full DataFrame
    df = df.merge(identifiers, on='frag_num', how='left')

    return df



def main(args):
    # Load pose files (JSON with model/chain info)
    '''
    frag_pos_to_replace: the position in the fragment the future rotamer should be inserted. central position recommended.
    residue_identity: only accept fragments with the correct residue identity at that position (recommended)
    rotamer_secondary_structure: accepts a string describing secondary structure (B: isolated beta bridge residue, E: strand, G: 3-10 helix, H: alpha helix, I: pi helix, T: turn, S: bend, -: none (not in the sense of no filter --> use None instead!)). e.g. provide EH if central atom in fragment should be a helix or strand.
    '''


    rotamer_positions_df = pd.read_pickle(args.rotpos_path)
    fraglib = pd.read_pickle(args.fraglib_path)

    #choose fragments from fragment library that contain the positions selected above
    fragnum = 0
    frags_dfs = []

    for pos in args.rotamer_positions.split(","):
        pos = int(pos)
        rotamer_positions_df["temp_pos_for_merge"] = pos

        # define start and end index for each position
        index_starts = rotamer_positions_df.index - pos + 1
        index_ends = index_starts + args.fragsize

        # create range between start and end
        all_values = []
        for start, end in zip(index_starts, index_ends):
            if start >= 0 and end <= fraglib.index.max():
                all_values.extend(range(start, end))  # Append the range to the list
        indices = np.array(all_values)

        # extract all indices
        df = fraglib.loc[indices]
        df['temp_index_for_merge'] = df.index
        df.reset_index(drop=True, inplace=True)

        # group by fragsize
        group_key = df.index // args.fragsize
        # check if each group has a unique pdb (to prevent taking frags with res from different pdbs) and consecutive numbers (to prevent using frags with chainbreaks)
        valid_groups = df.groupby(group_key).filter(lambda x: x['pdb'].nunique() == 1 and has_consecutive_numbers(x, "position", args.fragsize))
        if valid_groups.empty:
            continue

        valid_groups.reset_index(drop=True, inplace=True)

        # Create a new identifier column based on the valid groups
        group_key = valid_groups.index // args.fragsize
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

        # set non-rotamer positions to None
        frags_df.loc[frags_df['residue_numbers'] != pos, 'probability'] = None
        frags_df.loc[frags_df['residue_numbers'] != pos, 'phi_psi_occurrence'] = None
        for chi_angle in [col for col in frags_df.columns if col.startswith("chi")]:
            frags_df.loc[frags_df['residue_numbers'] != pos, chi_angle] = None


        # define rotamer position
        frags_df['rotamer_pos'] = pos

        # assign identifier based on phi/psi/omega angles, rotamer id and rotamer position
        frags_df = assign_frag_identifier(frags_df, pos)

        # add fragnum so that fragment numbering is continous for next position
        fragnum = fragnum + frags_df['frag_num'].max()

        # drop identifier column
        frags_df.drop(['temp_index_for_merge', 'temp_pos_for_merge'], axis=1, inplace=True)

        frags_dfs.append(frags_df)

    # write frags_df to pickle
    if frags_dfs:
        frags_dfs = pd.concat(frags_dfs)
        frags_dfs.to_pickle(args.outfile)
    else:
        pd.DataFrame().to_pickle(args.outfile)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--rotpos_path", type=str, required=True, help="Path to rotamer positions pickle")
    parser.add_argument("--fraglib_path", type=str, required=True, help="Path to fragment library")
    parser.add_argument("--rotamer_positions", type=str, required=True, help="Rotamer positions")
    parser.add_argument("--fragsize", type=int, required=True, help="Fragment size")
    parser.add_argument("--outfile", type=str, help="Path to output file")


    args = parser.parse_args()
    main(args)
