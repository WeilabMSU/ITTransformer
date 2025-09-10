"""
    Summary: persistent interaction homology for k (k=2 in the current version) simplicial complex

    1. timeit, should be removed in the final version.
    2. 

    Last update: 05-16-2025

"""



import numpy as np
import itertools
from functools import wraps
import copy
import argparse
import sys
import time
from collections import defaultdict
from bitarray import bitarray


def timeit(func):
    """ Timer """
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"{'='*5} Function - {func.__name__} - took {total_time:.3e} seconds {'='*5}")
        return result
    return timeit_wrapper


class InteractionHomologyOfSimplicialComplex_independent(object):
    
    def __init__(self):
        self.barcode_min_width = 1e-10
        self.barcode_max_death = 1e15
    
    @staticmethod
    def func_re_idx_simplicial_complex(simplicial_complex: dict, map_dict: dict) -> dict:
        return {d: [tuple(map(map_dict.get, sim)) for sim in sim_list] for d, sim_list in simplicial_complex.items()}

    @staticmethod
    def func_delete_irrelevant_simplicial_complex_with_distance(
        simplicial_complex: dict, simplicial_complex_form_distance: dict, considered_indices_set: set
    ) -> dict:
        new_simplicial_complex = defaultdict(list)
        new_simplicial_complex_form_distance = defaultdict(list)

        # Loop over dimensions and simplices
        for dim, simplex_list in simplicial_complex.items():
            for i, simplex in enumerate(simplex_list):
                # if bool(considered_indices_set & set(simplex)):
                if any(idx in considered_indices_set for idx in simplex):
                    new_simplicial_complex[dim].append(simplex)
                    new_simplicial_complex_form_distance[dim].append(simplicial_complex_form_distance[dim][i])

        return dict(new_simplicial_complex), dict(new_simplicial_complex_form_distance)

    @staticmethod
    def func_interaction_simplex_to_pivots_and_cycles_sparse(prev_dim_simplices, dim_simplices):
        """Use the dict: set to save the columns infor, faster in extreamly large case"""

        # Create a set of (dim_n-1)-simplices for fast lookup
        simplex_index_dict = {sim: i for i, sim in enumerate(prev_dim_simplices)}

        col_num = len(dim_simplices)
        boundary_matrix = {}  # Sparse representation as a dictionary

        # Process each (dim_n)-simplex
        for idx_n, (x, y) in enumerate(dim_simplices):
            column_entries = set()

            # Part 1: Compute ∂x ⊗ y
            x_len = len(x)
            if x_len > 1:
                for omitted_n in range(x_len):
                    x_y = (x[:omitted_n] + x[omitted_n+1:], y)
                    row_index = simplex_index_dict.get(x_y)
                    if row_index is not None:
                        column_entries.add(row_index)

            # Part 2: Compute x ⊗ ∂y
            y_len = len(y)
            if y_len > 1:
                for omitted_n in range(y_len):
                    x_y = (x, y[:omitted_n] + y[omitted_n+1:])
                    row_index = simplex_index_dict.get(x_y)
                    if row_index is not None:
                        column_entries.add(row_index)

            if column_entries:
                boundary_matrix[idx_n] = column_entries

        # Matrix column reduction
        pivot_pairs = {}
        empty_columns_idx = set()
        for col_idx in range(col_num):
            if col_idx not in boundary_matrix or not boundary_matrix[col_idx]:
                empty_columns_idx.add(col_idx)
                continue

            # Get the pivot index for the column using max (no need to sort)
            pivot_index = max(boundary_matrix[col_idx])
            pivot_pairs[pivot_index] = col_idx

            # Perform elimination on subsequent columns
            for next_col in range(col_idx + 1, col_num):
                if next_col in boundary_matrix and pivot_index in boundary_matrix[next_col]:
                    # XOR equivalent by symmetric difference of sets
                    boundary_matrix[next_col] = boundary_matrix[next_col].symmetric_difference(boundary_matrix[col_idx])

        return pivot_pairs, empty_columns_idx

    @staticmethod
    def func_interaction_simplex_to_pivots_and_cycles_bitarray(prev_dim_simplices, dim_simplices):
        """ Use the bitarray to save the columns, slower than sparse in large case"""

        # Create a set of (dim_n-1)-simplices for fast lookup
        simplex_index_dict = {sim: i for i, sim in enumerate(prev_dim_simplices)}

        col_num = len(dim_simplices)
        row_num = len(prev_dim_simplices)

        # Initialize the boundary matrix for previous dimension with empty bitarrays for each row
        boundary_matrix = [bitarray(row_num) for _ in range(col_num)]
        for row in boundary_matrix:
            row.setall(0)  # Set all bits to 0

        # Process each (dim_n)-simplex
        for idx_n, (x, y) in enumerate(dim_simplices):

            # Part 1: Compute ∂x ⊗ y
            x_len = len(x)
            if x_len > 1:
                for omitted_n in range(x_len):
                    x_y = (x[:omitted_n] + x[omitted_n+1:], y)
                    if x_y in simplex_index_dict:
                        boundary_matrix[idx_n][simplex_index_dict[x_y]] = 1

            # Part 2: Compute x ⊗ ∂y
            y_len = len(y)
            if y_len > 1:
                for omitted_n in range(y_len):
                    x_y = (x, y[:omitted_n] + y[omitted_n+1:])
                    if x_y in simplex_index_dict:
                        boundary_matrix[idx_n][simplex_index_dict[x_y]] = 1

        # matrix column reduction
        pivot_pairs = {}  # {row_idx: col_idx}
        empty_columns_idx = set()
        for col_idx, pivot_col in enumerate(boundary_matrix):

            if pivot_col.any():  # Check if pivot_row is non-zero

                # Identify the position of the last '1' bit in the pivot col
                pivot_index = pivot_col.find(1, right=True)
                pivot_pairs[pivot_index] = col_idx

                # Perform elimination on cols with a '1' in the pivot position
                for i in range(col_idx+1, col_num):
                    if boundary_matrix[i][pivot_index]:
                        boundary_matrix[i] ^= pivot_col  # XOR to eliminate
            else:  # Save column is all-zero
                empty_columns_idx.add(col_idx)

        return pivot_pairs, empty_columns_idx

    @staticmethod
    def func_compute_barcode_dim_0(max_interaction_complex, filtration_parameters, barcode_max_death):
        # Step 1: Preprocess inter_sim_dim1 to create a dictionary for quick lookup
        inter_sim_dim1_dict = defaultdict(list)
        param1_dict = {}

        for (interaction, param) in zip(max_interaction_complex[1], filtration_parameters[1]):
            for vertex in interaction:  # vertex are component in 1-dim inter simplex
                if len(vertex) == 1:
                    if vertex not in param1_dict or param < param1_dict[vertex]:
                        param1_dict[vertex] = param  # store minimum parameter for each vertex

        # Step 2: Construct barcode_dim_0 using dictionary lookups for efficient updates
        barcode_dim_0 = []
        for (inter_sim_dim0, param_0) in zip(max_interaction_complex[0], filtration_parameters[0]):
            # Default death time is barcode_max_death
            death_time = param1_dict.get(inter_sim_dim0[0], barcode_max_death)
            barcode_dim_0.append([param_0, death_time])

        return barcode_dim_0

    @staticmethod
    def func_sort_complex_by_form_distance(input_complex, input_complex_form_distance):

        sorted_complex = {}
        sorted_complex_form_distance = {}
        for key, simplex_list in input_complex.items():
            if len(simplex_list) > 0:
                combined = list(zip(input_complex[key], input_complex_form_distance[key]))
                combined_sorted = sorted(combined, key=lambda x: x[1])
                sorted_complex[key], sorted_complex_form_distance[key] = map(list, zip(*combined_sorted))

        return sorted_complex, sorted_complex_form_distance

    @staticmethod
    def func_reduce_interaction_complex_with_distance_dim2(interaction_complex):
        """"Reduce the 2-interaction simpleices, for speeding up"""
        assert 2 in interaction_complex, 'No 2-interaction simplices found.'
        return None

    @staticmethod
    def func_complexes_to_interaction_complex_with_distance(
        complex_0, complex_0_distance, complex_1, complex_1_distance, max_dim=1
    ):
        
        def build_vertex_index(complex_):
            vertex_to_simplices = defaultdict(list)
            for dim, simplices in complex_.items():
                for idx, simplex in enumerate(simplices):
                    for vertex in simplex:
                        vertex_to_simplices[vertex].append((dim, idx, simplex))
            return vertex_to_simplices

        interaction_complex = defaultdict(list)
        interaction_simplex_form_distance = defaultdict(list)

        # Preprocess complexes to sets and build vertex indices
        index_0 = build_vertex_index(complex_0)
        index_1 = build_vertex_index(complex_1)

        # Iterate over the vertices shared in both indices
        common_vertices = set(index_0.keys()) & set(index_1.keys())
        for vertex in common_vertices:
            simplices_0 = index_0[vertex]
            simplices_1 = index_1[vertex]

            for dim_key_0, i0, simplex_0 in simplices_0:
                for dim_key_1, i1, simplex_1 in simplices_1:
                    interaction_dim = dim_key_0 + dim_key_1
                    if interaction_dim > max_dim:
                        continue

                    interaction_complex[interaction_dim].append((simplex_0, simplex_1))
                    interaction_simplex_form_distance[interaction_dim].append(
                        max(complex_0_distance[dim_key_0][i0], complex_1_distance[dim_key_1][i1])
                    )

        return dict(interaction_complex), dict(interaction_simplex_form_distance)

    def persistent_interaction_homology_from_max_interaction_complex(
        self, max_interaction_complex: dict, filtration_parameters: dict, max_dim: int=None, sort_bar: bool=False,
    ) -> dict:

        # Initial barcodes
        barcodes = {dim: [] for dim in range(max_dim+1)}

        # For barcodes dim 0
        if 1 in max_interaction_complex:
            barcodes[0] = self.func_compute_barcode_dim_0(
                max_interaction_complex, filtration_parameters, self.barcode_max_death)
        else:
            barcodes[0] = [[p, self.barcode_max_death] for p in filtration_parameters[0]]

        if (max_dim < 1) or (max(max_interaction_complex.keys()) <= 1):
            return barcodes

        # For barcodes dim > 0
        # _, empty_columns_idx = self.func_interaction_simplex_to_pivots_and_cycles_sparse(
        #     max_interaction_complex[0], max_interaction_complex[1])
        _, empty_columns_idx = self.func_interaction_simplex_to_pivots_and_cycles_bitarray(
            max_interaction_complex[0], max_interaction_complex[1])
        
        max_boundary_dim = max(max_interaction_complex.keys())

        for dim in range(1, min([max_dim+1, max_boundary_dim])):
            # pivot_pairs, next_empty_columns_idx = self.func_interaction_simplex_to_pivots_and_cycles_sparse(
            # max_interaction_complex[dim], max_interaction_complex[dim+1])
            pivot_pairs, next_empty_columns_idx = self.func_interaction_simplex_to_pivots_and_cycles_bitarray(
            max_interaction_complex[dim], max_interaction_complex[dim+1])

            for empty_idx in empty_columns_idx:
                birth_distance = filtration_parameters[dim][empty_idx]
                if empty_idx in pivot_pairs:
                    death_distance = filtration_parameters[dim+1][pivot_pairs[empty_idx]]
                    if np.abs(death_distance - birth_distance) > self.barcode_min_width:
                        barcodes[dim].append([birth_distance, death_distance])
                else:
                    death_distance = self.barcode_max_death
                    barcodes[dim].append([birth_distance, death_distance])
            empty_columns_idx = next_empty_columns_idx

        # Sort the barcode
        if sort_bar:
            for dim, bars in barcodes.items():
                barcodes[dim] = np.array(sorted(bars, key=lambda x: (x[0], x[1])))
        return barcodes

    def persistent_interaction_homology_from_simplicial_complexes(
        self, complex_0, complex_0_distance, complex_1, complex_1_distance, max_dim=1, sort_bar=False) -> dict:

        max_interaction_complex_dim = max_dim+1
        interaction_complex, interaction_simplex_form_distance = self.func_complexes_to_interaction_complex_with_distance(
            complex_0, complex_0_distance, complex_1, complex_1_distance, max_interaction_complex_dim)
        
        self.interaction_complex, self.interaction_simplex_form_distance = self.func_sort_complex_by_form_distance(
            interaction_complex, interaction_simplex_form_distance)

        barcodes = self.persistent_interaction_homology_from_max_interaction_complex(
            self.interaction_complex, self.interaction_simplex_form_distance, max_dim, sort_bar)
        return barcodes


def example_1():
    # from multiple simplicial complex
    complex_0 = {0: [(0,), (1,), (2,)], 1: [(0, 1), (0, 2), (1, 2)]}
    complex_1 = {0: [(0,), (1,), (2,)], 1: [(0, 1), (0, 2), (1, 2)]}

    complex_0_distance = {0: [0, 0, 0], 1: [1, 1, 1.414]}
    complex_1_distance = {0: [0, 0, 0], 1: [1, 1, 1.414]}

    aa = InteractionHomologyOfSimplicialComplex_independent()
    barcodes = aa.persistent_interaction_homology_from_simplicial_complexes(
        complex_0, complex_0_distance, complex_1, complex_1_distance, max_dim=2)
    print(barcodes)

    return None


def example_2():
    input_data_1 = np.array([[0, 0], [0, 1], [1, 0]])
    input_data_2 = np.array([[0, 0], [0, 1], [1, 0]])
    input_data_idx_1 = [0, 1, 2]
    input_data_idx_2 = [0, 1, 2]

    input_data_1 = np.array(
        [
            [-1.708449,   0.801017,   0.426393],
            [ 1.708598,  -0.801695,   0.424990],
            [-0.138557,  -1.474758,   0.850645],
            [-0.844723,   0.043594,   1.643698],
            [-1.872974,  -0.815276,   0.473685],
            [-0.883847,  -1.449612,  -0.835888],
            [ 0.802067,  -1.501626,  -0.781854],
            [ 0.844626,  -0.046751,   1.643624],
            [ 0.138281,   1.473139,   0.853298],
            [ 1.872908,   0.814457,   0.475282],
            [ 1.790694,  -0.033505,  -1.022062],
            [-1.790583,   0.035388,  -1.022091],
            [ 0.000044,   0.001533,  -1.689435],
            [ 0.883914,   1.451140,  -0.833233],
            [-0.802002,   1.503066,  -0.779290],
        ]
    )
    input_data_idx_1 = list(range(15))

    input_data_2 = np.array(
        [
            [-1.708449,   0.801017,   0.426393],
            [ 1.708598,  -0.801695,   0.424990],
            [-0.138557,  -1.474758,   0.850645],
            [-0.844723,   0.043594,   1.643698],
            [-1.872974,  -0.815276,   0.473685],
            [-0.044117,  -2.478637,   1.463455],
            [-1.291388,  -2.396633,  -1.415122],
            [ 1.265833,  -2.466648,  -1.279228],
            [ 2.526761,  -1.418991,   0.767245],
            [ 1.287622,  -0.120512,   2.734956],
            [-1.287759,   0.115411,   2.735142],
            [-2.849004,  -1.326059,   0.891660],
            [ 0.043879,   2.475893,   1.467950],
            [ 2.848803,   1.324609,   0.894335],
            [ 2.722457,  -0.161592,  -1.731534],
            [ 0.000079,   0.002564,  -2.871842],
            [-2.722359,   0.164769,  -1.731314],
            [-2.526670,   1.417777,   0.769758],
            [-1.265761,   2.468976,  -1.274937],
            [ 1.291491,   2.399212,  -1.410723],
        ]
    )
    input_data_idx_2 = [0, 1, 3, 4, 5] + [15 + i for i in range(15)]

    interaction_max_dim = 1
    max_dim_1 = 2
    max_dim_2 = 0
    from interaction_homology_simplicialcomplex import InteractionHomologyOfSimplicialComplex
    from scipy.spatial import distance
    
    pih = InteractionHomologyOfSimplicialComplex()

    persistent_barcode = pih.persistent_interaction_homology_from_multi_cloudpoints_for_vr_complex(
        cloudpoints_0=input_data_1, cloudpoints_1=input_data_2,
        vertex_idx_0=input_data_idx_1, vertex_idx_1=input_data_idx_2,
        max_dim_0=max_dim_1,  max_dim_1=max_dim_2,
        max_dim=interaction_max_dim,
    )

    complex_0, complex_0_distance = pih.complex_0, pih.complex_form_distance_0
    complex_1, complex_1_distance = pih.complex_1, pih.complex_form_distance_1

    tt = InteractionHomologyOfSimplicialComplex_independent()
    bar = tt.persistent_interaction_homology_from_simplicial_complexes(
        complex_0, complex_0_distance, complex_1, complex_1_distance, interaction_max_dim
    )    


    # print(persistent_barcode)
    for k, v in persistent_barcode.items():
        print(f'dim: {k}', np.round(v, 3))
        print('++++++++++')
        print(f'dim: {k}', np.round(bar[k], 3))

    return None


def example_3():
    # from multiple simplicial complex
    complex_0 = {0: [(0,), (1,), (2,)], 1: [(0, 1), (0, 2)]}
    complex_1 = {0: [(0,), (1,)]}

    complex_0_distance = {0: [0, 0, 0], 1: [1, 1.1], 2: [1.5]}
    complex_1_distance = {0: [0, 0]}

    aa = InteractionHomologyOfSimplicialComplex_independent()
    barcodes = aa.persistent_interaction_homology_from_simplicial_complexes(
        complex_0, complex_0_distance, complex_1, complex_1_distance, max_dim=1)
    print(barcodes)

    return None


def main():
    # example_1()
    example_3()
    return None


if __name__ == "__main__":
    main()