import sys
import os
import numpy as np
import gudhi
from pymatgen.core.structure import Structure
import argparse
import glob
from tqdm import tqdm
import concurrent.futures
import json

from collections import defaultdict
from pymatgen.core.periodic_table import Element

# Add the project root to the Python path to allow for package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main.itt.utils import ElementClusterInfo

class PIHGenerator:
    """
    Generates Persistent Homology features for a given structure file (CIF or XYZ).
    The features consist of four parts:
    0. Persistent homology features for the entire structure
    1. Persistent homology features for each element cluster
    2. Persistent interaction homology features between clusters
    3. Graph-based cluster embeddings (for use with graph module in modeling_itt.py)
    """
    def __init__(self, abc_norm=64, 
                 # Distance parameters for each part
                 structure_max_dist=25.0,  # Part 0: entire structure
                 cluster_max_dist=25.0,    # Part 1: clusters
                 interaction_max_dist=5.0, # Part 2: interactions
                 
                 # Dimension parameters for each part
                 structure_max_dim=2,      # Part 0: entire structure
                 cluster_max_dim=2,        # Part 1: clusters
                 interaction_max_dim=1,    # Part 2: interactions
                 
                 # Max dimensions for interaction alpha complexes
                 interaction_complex0_max_dim=0,  # First alpha complex in interaction
                 interaction_complex1_max_dim=2,  # Second alpha complex in interaction
                 
                 # Binning parameters for entire structure PH
                 structure_bin_start=0.0,
                 structure_bin_end=25.0,
                 structure_bin_step=0.1,
                 
                 # Binning parameters for cluster PH
                 cluster_bin_start=0.0,
                 cluster_bin_end=25.0,
                 cluster_bin_step=0.1,
                 
                 # Binning parameters for interaction PH
                 interaction_bin_start=0.0,
                 interaction_bin_end=10.0,
                 interaction_bin_step=0.04,
                 
                 vec_dtype=np.uint16,
                 graph_radius=8.0,
                 graph_max_num_nbr=12,
                 ):
        """
        Initializes the PIHGenerator.

        Args:
            abc_norm (float): The target lattice parameter length for normalization.
            
            # Distance parameters
            structure_max_dist (float): Maximum distance for entire structure Alpha complex
            cluster_max_dist (float): Maximum distance for cluster Alpha complex
            interaction_max_dist (float): Maximum distance for interaction Alpha complex
            
            # Dimension parameters
            structure_max_dim (int): Maximum dimension for entire structure PH
            cluster_max_dim (int): Maximum dimension for cluster PH
            interaction_max_dim (int): Maximum dimension for interaction PH
            
            # Max dimensions for interaction alpha complexes
            interaction_complex0_max_dim (int): Max dimension for first alpha complex in interaction
            interaction_complex1_max_dim (int): Max dimension for second alpha complex in interaction
            
            # Binning parameters for entire structure PH
            structure_bin_start (float): Start of binning for entire structure PH
            structure_bin_end (float): End of binning for entire structure PH
            structure_bin_step (float): Step size for entire structure PH binning
            
            # Binning parameters for cluster PH
            cluster_bin_start (float): Start of binning for cluster PH
            cluster_bin_end (float): End of binning for cluster PH
            cluster_bin_step (float): Step size for cluster PH binning
            
            # Binning parameters for interaction PH
            interaction_bin_start (float): Start of binning for interaction PH
            interaction_bin_end (float): End of binning for interaction PH
            interaction_bin_step (float): Step size for interaction PH binning
            
            # Graph parameters for part3 (raw input preparation only)
            graph_radius (float): Cutoff radius for neighbor search in graph construction
            graph_max_num_nbr (int): Maximum number of neighbors per atom
            
            vec_dtype (numpy.dtype): The data type for feature vectors.
        """
        self.abc_norm = abc_norm
        
        # Store distance parameters
        self.structure_max_dist = structure_max_dist
        self.cluster_max_dist = cluster_max_dist
        self.interaction_max_dist = interaction_max_dist
        
        # Store dimension parameters
        self.structure_max_dim = structure_max_dim
        self.cluster_max_dim = cluster_max_dim
        self.interaction_max_dim = interaction_max_dim
        
        # Store interaction complex dimensions
        self.interaction_complex0_max_dim = interaction_complex0_max_dim
        self.interaction_complex1_max_dim = interaction_complex1_max_dim
        
        # Store graph parameters (only for raw input preparation)
        self.graph_radius = graph_radius
        self.graph_max_num_nbr = graph_max_num_nbr
        
        # Initialize binning grids for each type of feature
        self.structure_grid = np.arange(structure_bin_start, structure_bin_end + structure_bin_step, structure_bin_step)
        self.structure_num_bins = len(self.structure_grid) - 1
        
        self.cluster_grid = np.arange(cluster_bin_start, cluster_bin_end + cluster_bin_step, cluster_bin_step)
        self.cluster_num_bins = len(self.cluster_grid) - 1
        
        self.interaction_grid = np.arange(interaction_bin_start, interaction_bin_end + interaction_bin_step, interaction_bin_step)
        self.interaction_num_bins = len(self.interaction_grid) - 1
        
        # Pre-calculate vector sizes for each part
        # Part 0 (structure): (max_dim + 1) * num_bins
        self.structure_vector_size = (self.structure_max_dim + 1) * self.structure_num_bins
        
        # Part 1 (clusters): (max_dim + 1) * num_bins
        self.cluster_vector_size = (self.cluster_max_dim + 1) * self.cluster_num_bins
        
        # Part 2 (interactions): (max_dim + 1) * num_bins
        self.interaction_vector_size = (self.interaction_max_dim + 1) * self.interaction_num_bins
        
        # Pre-create zero vectors for each part
        self.structure_zero_vector = np.zeros(self.structure_vector_size, dtype=vec_dtype)
        self.cluster_zero_vector = np.zeros(self.cluster_vector_size, dtype=vec_dtype)
        self.interaction_zero_vector = np.zeros(self.interaction_vector_size, dtype=vec_dtype)

        # Initialize graph-related components for raw input preparation
        self.element_info = ElementClusterInfo()
        self.dtype = vec_dtype

    def _normalize_structure(self, structure):
        """
        Expands the structure to meet the abc_norm requirement.
        """
        a, b, c = structure.lattice.abc
        scaling_factors = [
            max([1, round(self.abc_norm / a)]),
            max([1, round(self.abc_norm / b)]),
            max([1, round(self.abc_norm / c)]),
        ]
        supercell = structure.copy()
        supercell.make_supercell(scaling_factors)
        return supercell

    def _get_atom_clusters(self, structure):
        """
        Splits atoms into clusters based on element type.
        Returns a dictionary with 0-indexed cluster IDs.
        """
        clusters = {i: [] for i in range(self.element_info.total_element_cluster)}  # 0-indexed clusters
        for site in structure:
            symbol = site.specie.symbol
            cluster_id = self.element_info.get_cluster_from_symbol(symbol)
            if cluster_id is not None:
                clusters[cluster_id].append(site.coords)
        
        for cluster_id in clusters:
            clusters[cluster_id] = np.array(clusters[cluster_id])
            
        return clusters

    def _calculate_ph_and_vectorize(self, points, max_dim=2, grid=None, num_bins=None, max_dist=None):
        """
        Calculates persistent homology and vectorizes the barcodes.
        This implementation counts the number of persistence bars fully contained 
        in each grid interval.

        Args:
            points: np.ndarray of point coordinates
            max_dim: Maximum dimension for PH calculation
            grid: Binning grid to use
            num_bins: Number of bins in the grid
            max_dist: Maximum distance for Alpha complex construction
        """
        if grid is None:
            grid = self.structure_grid
        if num_bins is None:
            num_bins = self.structure_num_bins
        if max_dist is None:
            max_dist = self.structure_max_dist
            
        num_dims = max_dim + 1
        total_vector_size = num_dims * num_bins
        if points.shape[0] < num_dims + 1 or num_bins <= 0:
            return np.zeros(total_vector_size, dtype=self.dtype)
            
        alpha_complex = gudhi.AlphaComplex(points=points)
        simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=max_dist**2)
        
        simplex_tree.persistence()
        
        all_dim_vectors = []
        for dim in range(num_dims):
            barcodes = simplex_tree.persistence_intervals_in_dimension(dim)
            dim_vector = np.zeros(num_bins, dtype=self.dtype)

            if len(barcodes) > 0:
                # Filtration values are squared radii for Alpha Complex
                births = np.sqrt(barcodes[:, 0])
                deaths = np.sqrt(barcodes[:, 1])
                
                # Replace infinity with a large number for vectorization
                deaths[deaths == np.inf] = max_dist

                for i in range(num_bins):
                    bin_start = grid[i]
                    bin_end = grid[i+1]
                    # Count bars [b, d) such that bin_start >= b and d >= bin_end
                    count = np.sum((births <= bin_start) & (deaths >= bin_end))
                    dim_vector[i] = count
            
            all_dim_vectors.append(dim_vector)
            
        return np.concatenate(all_dim_vectors)

    def _get_entire_structure_features(self, structure):
        """
        Part 0: Generate persistent homology features for the entire structure.
        """
        all_points = structure.cart_coords

        return self._calculate_ph_and_vectorize(
            all_points, 
            max_dim=self.structure_max_dim,
            grid=self.structure_grid,
            num_bins=self.structure_num_bins,
            max_dist=self.structure_max_dist
        )

    def _get_cluster_features(self, clusters):
        """
        Part 1: Generate persistent homology features for each element cluster.
        Returns zero vectors for empty clusters.
        """
        ph_vectors_clusters = []
        cluster_ids = sorted(clusters.keys())  # 0-indexed cluster IDs
        
        for cluster_id in cluster_ids:
            points = clusters[cluster_id]
            if len(points) == 0:  # Empty cluster
                ph_vectors_clusters.append(self.cluster_zero_vector)
            else:
                ph_vector_cluster = self._calculate_ph_and_vectorize(
                    points, 
                    max_dim=self.cluster_max_dim,
                    grid=self.cluster_grid,
                    num_bins=self.cluster_num_bins,
                    max_dist=self.cluster_max_dist
                )
                ph_vectors_clusters.append(ph_vector_cluster)
                
        return ph_vectors_clusters

    def _get_interaction_features(self, clusters):
        """
        Part 2: Generate persistent interaction homology features between clusters.
        Considers all permutations of cluster interactions since the interaction
        between cluster A and B is different from B and A.
        Returns zero vectors for interactions involving empty point sets.
        """
        interaction_vectors = []
        cluster_ids = sorted(clusters.keys())  # 0-indexed cluster IDs
        
        for cluster_id1 in cluster_ids:
            points1 = clusters[cluster_id1]
            
            for cluster_id2 in cluster_ids:
                if cluster_id1 == cluster_id2:  # Skip self-interactions
                    continue
                    
                points2 = clusters[cluster_id2]
                
                # If either point set is empty, use zero vector
                if len(points1) == 0 or len(points2) == 0:
                    interaction_vectors.append(self.interaction_zero_vector)
                    continue
                    
                # Get interaction features between these two clusters
                interaction_vector = self._calculate_pih_andvectorize(
                    points1, 
                    points2, 
                    max_dim=self.interaction_max_dim,
                    grid=self.interaction_grid,
                    num_bins=self.interaction_num_bins,
                    max_dist=self.interaction_max_dist,
                    complex0_max_dim=self.interaction_complex0_max_dim,
                    complex1_max_dim=self.interaction_complex1_max_dim
                )
                interaction_vectors.append(interaction_vector)
        return interaction_vectors

    def _calculate_pih_andvectorize(self, xyz0, xyz1, max_dim=1, grid=None, num_bins=None, 
                                     max_dist=10, complex0_max_dim=0, complex1_max_dim=1):
        """
        Compute persistent interaction homology barcode vector for two sets of coordinates.
        Uses the same binning approach as _calculate_ph_and_vectorize.
        
        Args:
            xyz0: np.ndarray, shape (n0, 3), anchor/overlap points
            xyz1: np.ndarray, shape (n1, 3), other cluster
            max_dim: int, max dimension for interaction homology
            grid: np.ndarray, binning grid
            num_bins: int, number of bins
            max_dist: float, maximum distance for Alpha complex
            complex0_max_dim: int, max dimension for first alpha complex
            complex1_max_dim: int, max dimension for second alpha complex
        Returns:
            np.ndarray: binned barcode vector (concatenated for all dims)
        """
        import gudhi
        from main.itt.persistent_interaction_homology import InteractionHomologyOfSimplicialComplex_independent as IHSC
        from collections import defaultdict

        num_dims = max_dim + 1

        # 1. 0-dim alpha complex for xyz0
        n0 = xyz0.shape[0]
        alpha_complex_0 = defaultdict(list)
        alpha_complex_params_0 = defaultdict(list)
        for i in range(n0):
            alpha_complex_0[0].append((i,))
            alpha_complex_params_0[0].append(0.0)

        # 2. Alpha complex for combined xyz0+xyz1 (xyz0 first)
        xyz_combined = np.vstack([xyz0, xyz1])
        alpha_complex_1 = defaultdict(list)
        alpha_complex_params_1 = defaultdict(list)
        
        # Create a set of vertex indices from xyz0 for quick lookup
        xyz0_vertices = set(range(n0))
        
        ac = gudhi.AlphaComplex(points=xyz_combined)
        st = ac.create_simplex_tree(max_alpha_square=max_dist**2)
        for simplex, filt in st.get_filtration():
            dim = len(simplex) - 1
            if dim > complex1_max_dim:
                continue
            if filt > max_dist**2:
                continue
                
            # Count how many vertices in this simplex are from xyz0
            xyz0_vertex_count = sum(1 for v in simplex if v in xyz0_vertices)
            
            # Only add simplex if it doesn't contain multiple vertices from xyz0
            if xyz0_vertex_count <= 1:
                alpha_complex_1[dim].append(tuple(simplex))
                alpha_complex_params_1[dim].append(np.sqrt(filt))

        # 3. delete the irrelevant simplicial complex
        overlap_set = set(range(n0))
        alpha_complex_0, alpha_complex_params_0 = IHSC.func_delete_irrelevant_simplicial_complex_with_distance(
            alpha_complex_0, alpha_complex_params_0, overlap_set)
        alpha_complex_1, alpha_complex_params_1 = IHSC.func_delete_irrelevant_simplicial_complex_with_distance(
            alpha_complex_1, alpha_complex_params_1, overlap_set)

        # 3. Compute persistent interaction homology
        ihsc = IHSC()
        barcodes = ihsc.persistent_interaction_homology_from_simplicial_complexes(
            alpha_complex_0, alpha_complex_params_0, alpha_complex_1, alpha_complex_params_1, max_dim=max_dim)

        # 4. Bin the barcodes for each dimension using the same approach as _calculate_ph_and_vectorize
        all_dim_vectors = []
        for dim in range(num_dims):
            bars = np.array(barcodes.get(dim, []))
            dim_vector = np.zeros(num_bins, dtype=self.dtype)

            if len(bars) > 0:
                births = bars[:, 0]
                deaths = bars[:, 1]
                # Replace infinity with a large number for vectorization
                deaths[deaths == np.inf] = max_dist

                for i in range(num_bins):
                    bin_start = grid[i]
                    bin_end = grid[i+1]
                    # Count bars [b, d) such that bin_start >= b and d >= bin_end
                    count = np.sum((births <= bin_start) & (deaths >= bin_end))
                    dim_vector[i] = count

            all_dim_vectors.append(dim_vector)

        return np.concatenate(all_dim_vectors)

    def _get_graph_features(self, structure):
        """
        Part 3: Generate raw graph input features for the structure.
        This prepares the raw input vectors that will be processed by the neural network
        in modeling_itt.py. No neural network processing is done here.
        
        Returns:
            tuple: (unique_atom_nos, unique_nbr_fea_dist, unique_nbr_fea_idx, unique_cluster_indices) where:
                - unique_atom_nos: Raw atom features, shape [num_unique_atoms]
                - unique_nbr_fea_dist: Neighbor distances, shape [num_unique_atoms, max_num_nbr]
                - unique_nbr_fea_idx: Neighbor indices, shape [num_unique_atoms, max_num_nbr]
                - unique_cluster_indices: Cluster assignment for each atom, shape [num_unique_atoms]
        """
        # Get unique items
        unique_atom_indices, unique_atom_weights = self._get_unique_atom_features(structure)

        # Get atom features, (raw 92-dimensional features from atom_init.json)
        atom_nos = np.array([atom.specie.number for atom in structure], dtype=self.dtype)
        
        # Get neighbor features
        all_nbrs = structure.get_all_neighbors(self.graph_radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        
        nbr_fea_idx, nbr_fea_dist = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.graph_max_num_nbr:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) + [0] * (self.graph_max_num_nbr - len(nbr)))
                nbr_fea_dist.append(list(map(lambda x: x[1], nbr)) + [self.graph_radius + 1.] * (self.graph_max_num_nbr - len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:self.graph_max_num_nbr])))
                nbr_fea_dist.append(list(map(lambda x: x[1], nbr[:self.graph_max_num_nbr])))
        
        # Unique and scalued distance. Max distance is the 2**16 / 1000, for saving 3 dicemal
        unique_nbr_fea_dist = np.array(np.round(nbr_fea_dist, 3)*1000, dtype=self.dtype)[unique_atom_indices]

        nbr_fea_idx = np.array(nbr_fea_idx, dtype=self.dtype)

        # Remap neighbor indices to new unique atom indices
        orig_to_unique = {orig_idx: new_idx for new_idx, orig_idx in enumerate(unique_atom_indices)}
        remapped_nbr_fea_idx = []
        for orig_idx in unique_atom_indices:
            nbrs = nbr_fea_idx[orig_idx]
            remapped = [orig_to_unique[nidx] if nidx in orig_to_unique else 0 for nidx in nbrs]
            remapped_nbr_fea_idx.append(remapped)
        unique_nbr_fea_idx = np.array(remapped_nbr_fea_idx)
        
        # Get cluster indices for each atom
        cluster_indices = np.array([self.element_info.get_cluster_from_atomic_number(str(num)) for num in atom_nos], dtype=self.dtype)

        # Get unique output
        unique_atom_nos = atom_nos[unique_atom_indices]  # get unique atom numbers, shape [num_unique_atoms]
        unique_cluster_indices = cluster_indices[unique_atom_indices]

        return unique_atom_nos, unique_nbr_fea_dist, unique_nbr_fea_idx, unique_cluster_indices, np.array(unique_atom_weights, dtype=self.dtype)

    def _get_unique_atom_features(self, structure):
        """
        Part 3: Generate unique atom features for the structure.
        """
        atomic_numbers = [site.specie.Z for site in structure.sites]
        n_atoms = len(structure.sites)

        # Get atomic radii
        atomic_radii = {}
        for site in structure.sites:
            element = Element.from_Z(site.specie.Z)
            # Use atomic radius, fallback to covalent radius if atomic radius not available
            try:
                radius = element.atomic_radius
            except:
                radius = element.covalent_radius
            atomic_radii[site.specie.Z] = radius
        
        # Precompute neigbors up to 3 bonds away using stomic radii
        neighbors_dict = {i: {1: set(), 2: set(), 3: set()} for i in range(n_atoms)}
        
        # 1st shell neighbors using atomic radii with periodic conditions
        for i in range(n_atoms):
            center_atom = structure[i]
            center_radius = atomic_radii[center_atom.specie.Z]
            
            # Get all neighbors within cutoff distance (sum of radii + small buffer)
            # cutoff = center_radius + max(atomic_radii.values()) + 0.5  # 0.5Å buffer
            cutoff = 8
            neighbors = structure.get_neighbors(center_atom, cutoff)
            
            # Filter neighbors based on bond distance (sum of atomic radii)
            for neighbor in neighbors:
                neighbor_idx = neighbor.index
                if neighbor_idx != i:
                    neighbor_radius = atomic_radii[structure[neighbor_idx].specie.Z]
                    bond_distance = center_radius + neighbor_radius
                    
                    # If distance is within bond length (with some tolerance)
                    if neighbor.distance(center_atom) <= bond_distance * 1.3:  # 30% tolerance
                        neighbors_dict[i][1].add(neighbor_idx)
        
        # 2nd shell neighbors
        for i in range(n_atoms):
            n1 = neighbors_dict[i][1]
            n2 = set()
            for j in n1:
                n2.update(neighbors_dict[j][1])
            n2.discard(i)
            n2 -= n1
            neighbors_dict[i][2] = n2
        
        # 3rd shell neighbors
        for i in range(n_atoms):
            n2 = neighbors_dict[i][2]
            n3 = set()
            for j in n2:
                n3.update(neighbors_dict[j][1])
            n3.discard(i)
            n3 -= neighbors_dict[i][1]
            n3 -= n2
            neighbors_dict[i][3] = n3

        # Build the vector for each atom
        vectors = []
        for i in range(n_atoms):
            n_i = atomic_numbers[i]
            N1 = [atomic_numbers[j] for j in neighbors_dict[i][1]]
            N2 = [atomic_numbers[j] for j in neighbors_dict[i][2]]
            N3 = [atomic_numbers[j] for j in neighbors_dict[i][3]]
            vec = [
                n_i,
                len(N1),
                # int(sum(N1)),
                int(sum([x**2 for x in N1])),
                len(N2),
                # int(sum(N2)),
                int(sum([x**2 for x in N2])),
                len(N3),   # using length
                # int(sum(N3)),
                int(sum([x**2 for x in N3])),
            ]
            vectors.append(tuple(vec))
        
        # Group atoms by their vectors
        vector_to_indices = defaultdict(list)
        for idx, vec in enumerate(vectors):
            vector_to_indices[vec].append(idx)
        
        unique_atom_indices = [indices[0] for indices in vector_to_indices.values()]
        unique_atom_weights = [len(indices) for indices in vector_to_indices.values()]
        return unique_atom_indices, unique_atom_weights

    def _read_xyz_file(self, xyz_path):
        """
        Read an XYZ file and return a Structure object.
        
        Args:
            xyz_path (str): Path to the XYZ file.
            
        Returns:
            Structure: A pymatgen Structure object.
            
        Raises:
            ValueError: If the XYZ file format is invalid.
        """
        try:
            with open(xyz_path, 'r') as f:
                lines = f.readlines()
                
            # First line should be the number of atoms
            try:
                num_atoms = int(lines[0].strip())
            except ValueError:
                raise ValueError("First line of XYZ file must be the number of atoms")
                
            # Second line is usually a comment
            comment = lines[1].strip()
            
            # Parse atom coordinates and species
            species = []
            coords = []
            
            for line in lines[2:2+num_atoms]:
                parts = line.strip().split()
                if len(parts) < 4:
                    raise ValueError(f"Invalid line in XYZ file: {line}")
                    
                species.append(parts[0])
                coords.append([float(x) for x in parts[1:4]])
                
            # Create a cubic lattice with large enough cell size
            lattice_size = max(max(abs(x) for x in coord) for coord in coords) * 2.1
            lattice = [[lattice_size, 0, 0], [0, lattice_size, 0], [0, 0, lattice_size]]
            
            # Create Structure object
            structure = Structure(lattice=lattice, species=species, coords=coords, coords_are_cartesian=True)
            return structure
            
        except Exception as e:
            raise ValueError(f"Error reading XYZ file {xyz_path}: {str(e)}")

    def generate_features(self, file_path):
        """
        Main workflow to generate features for a structure file (CIF or XYZ).
        Generates four types of features:
        0. Persistent homology features for the entire structure
        1. Persistent homology features for each element cluster
        2. Persistent interaction homology features between clusters
        3. Graph-based cluster embeddings, raw input vectors for the graph module

        Args:
            file_path (str): Path to the input file (CIF or XYZ).

        Returns:
            tuple: Four numpy arrays containing:
                - entire_structure_features: Features for the entire structure
                - cluster_features: Features for each element cluster
                - interaction_features: Features for interactions between clusters
                - graph_features: Features for graph-based cluster embeddings
        """
        try:
            # Determine file type and read accordingly
            if file_path.lower().endswith('.cif'):
                structure = Structure.from_file(file_path)
            elif file_path.lower().endswith('.xyz'):
                structure = self._read_xyz_file(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}. Only .cif and .xyz files are supported.")
                
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None, None, None, None

        # Normalize structure to meet size requirements
        supercell = self._normalize_structure(structure)
        
        # Get atom clusters
        clusters = self._get_atom_clusters(supercell)
        
        # Generate all four types of features
        entire_structure_features = self._get_entire_structure_features(supercell)
        cluster_features = self._get_cluster_features(clusters)
        interaction_features = self._get_interaction_features(clusters)
        graph_features = self._get_graph_features(structure)
        
        return entire_structure_features, cluster_features, interaction_features, graph_features

    def process_file(self, file_path, file_type='cif'):
        """
        Process a single file and return formatted features.
        
        Args:
            file_path (str): Path to the input file
            file_type (str): Type of file ('cif' or 'xyz')
            
        Returns:
            dict: Formatted features with part0, part1, part2, part3 structure or None if processing failed
        """
        try:
            # Get the file ID (basename without extension)
            file_id = os.path.splitext(os.path.basename(file_path))[0]
            
            # Generate features
            entire_structure_features, cluster_features, interaction_features, graph_features = self.generate_features(file_path)
            if entire_structure_features is None:  # Check if any feature is None
                return None
            
            # Unpack graph features
            unique_atom_nos, unique_nbr_fea_dist, unique_nbr_fea_idx, unique_cluster_indices, unique_atom_weights = graph_features
                
            # Format features
            formatted_features = {
                'part0': entire_structure_features.reshape(1, -1),  # Only reshape part0 to [1, embedding_dim]
                'part1': np.array(cluster_features),  # Already [7, embedding_dim]
                'part2': np.array(interaction_features),  # Already [42, embedding_dim]
                'part3_atom_nos': unique_atom_nos,  # [num_unique_atoms]
                'part3_nbr_fea_dist': unique_nbr_fea_dist,  # [num_unique_atoms, max_num_nbr]
                'part3_nbr_fea_idx': unique_nbr_fea_idx,  # [num_unique_atoms, max_num_nbr]
                'part3_cluster_indices': unique_cluster_indices,  # [num_unique_atoms]
                'part3_atom_weights': unique_atom_weights,
                'cif_id': file_id,
            }
            return formatted_features
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return None

    def process_directory(self, input_dir, output_file, file_type='cif', num_workers=1):
        """
        Process all structure files in a directory and save features.
        
        Args:
            input_dir (str): Directory containing structure files
            output_file (str): Path to save the output features
            file_type (str): Type of files to process ('cif' or 'xyz')
            num_workers (int): Number of worker threads for parallel processing
        """
        # Get all files of the specified type
        pattern = os.path.join(input_dir, f'*.{file_type.lower()}')
        file_paths = sorted(glob.glob(pattern))
        num_files = len(file_paths)
        if not file_paths:
            print(f"No {file_type} files found in {input_dir}")
            return

        print(f"Found {num_files} {file_type} files")

        # Process the first file to get feature shapes
        first_result = self.process_file(file_paths[0], file_type)
        if first_result is None:
            print(f"Failed to process the first file: {file_paths[0]}")
            return

        part0_shape = first_result['part0'].shape[1:]
        part1_shape = first_result['part1'].shape[1:]
        part2_shape = first_result['part2'].shape[1:]

        # Pre-allocate arrays for fixed-shape features
        part0_arr = np.zeros((num_files, *part0_shape), dtype=first_result['part0'].dtype)
        part1_arr = np.zeros((num_files, *part1_shape), dtype=first_result['part1'].dtype)
        part2_arr = np.zeros((num_files, *part2_shape), dtype=first_result['part2'].dtype)
        
        # Initialize lists for variable-shape graph features
        part3_atom_fea_list = []
        part3_nbr_fea_list = []
        part3_nbr_fea_idx_list = []
        part3_cluster_indices_list = []
        cif_ids = [None] * num_files

        # Helper for parallel processing
        def process_and_store(idx_file):
            idx, file_path = idx_file
            result = self.process_file(file_path, file_type)
            return idx, result

        # Parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_and_store, enumerate(file_paths)))

        for idx, result in results:
            if result is not None:
                part0_arr[idx] = result['part0']
                part1_arr[idx] = result['part1']
                part2_arr[idx] = result['part2']
                part3_atom_fea_list.append(result['part3_atom_nos'])
                part3_nbr_fea_list.append(result['part3_nbr_fea_dist'])
                part3_nbr_fea_idx_list.append(result['part3_nbr_fea_idx'])
                part3_cluster_indices_list.append(result['part3_cluster_indices'])
                cif_ids[idx] = result['cif_id']

        combined_features = {
            'part0': part0_arr,
            'part1': part1_arr,
            'part2': part2_arr,
            'part3_atom_nos': part3_atom_fea_list,
            'part3_nbr_fea_dist': part3_nbr_fea_list,
            'part3_nbr_fea_idx': part3_nbr_fea_idx_list,
            'part3_cluster_indices': part3_cluster_indices_list,
            'cif_ids': cif_ids
        }
        np.savez_compressed(output_file, **combined_features)
        print(f"Features saved to {output_file}")

    def combine_feature_files(self, input_dir, output_file):
        """
        Combine separate .npz feature files in a folder into a single .npz file.

        Args:
            input_dir (str): Directory containing individual .npz feature files.
            output_file (str): Path to save the combined .npz file.
        """
        import glob
        npz_files = sorted(glob.glob(os.path.join(input_dir, '*.npz')))
        if not npz_files:
            print(f"No .npz files found in {input_dir}")
            return

        # Initialize lists to store features
        part0_list = []
        part1_list = []
        part2_list = []
        part3_atom_fea_list = []
        part3_nbr_fea_list = []
        part3_nbr_fea_idx_list = []
        part3_cluster_indices_list = []
        cif_ids = []

        # Process each file
        for npz_path in tqdm(npz_files, desc="Combining feature files"):
            data = np.load(npz_path, allow_pickle=True)
            
            # Get features and cif_ids
            part0 = data['part0']
            part1 = data['part1']
            part2 = data['part2']
            
            # Handle graph features
            if 'part3_atom_nos' in data:
                # New format with separate graph components
                part3_atom_fea = data['part3_atom_nos']
                part3_nbr_fea = data['part3_nbr_fea_dist']
                part3_nbr_fea_idx = data['part3_nbr_fea_idx']
                part3_cluster_indices = data['part3_cluster_indices']
            else:
                # Old format - skip for now
                continue
                
            # Handle cif_ids
            if 'cif_id' in data:
                ids = [str(data['cif_id'])]
            else:
                ids = [str(x) for x in data['cif_ids']]
                
            # Append features and ids
            part0_list.append(part0)
            part1_list.append(part1)
            part2_list.append(part2)
            part3_atom_fea_list.append(part3_atom_fea)
            part3_nbr_fea_list.append(part3_nbr_fea)
            part3_nbr_fea_idx_list.append(part3_nbr_fea_idx)
            part3_cluster_indices_list.append(part3_cluster_indices)
            cif_ids.extend(ids)

        # Stack the features
        part0_arr = np.stack(part0_list)  # Will be [w, n0, d0]
        part1_arr = np.stack(part1_list)  # Will be [w, n1, d1]
        part2_arr = np.stack(part2_list)  # Will be [w, n2, d2]

        # Create the combined features dictionary
        combined_features = {
            'part0': part0_arr,
            'part1': part1_arr,
            'part2': part2_arr,
            'part3_atom_fea': part3_atom_fea_list,
            'part3_nbr_fea': part3_nbr_fea_list,
            'part3_nbr_fea_idx': part3_nbr_fea_idx_list,
            'part3_cluster_indices': part3_cluster_indices_list,
            'cif_ids': np.array(cif_ids)
        }
        
        # Save the combined features
        np.savez_compressed(output_file, **combined_features)
        print(f"Combined features saved to {output_file}")
        print(f"Final shapes - part0: {part0_arr.shape}, part1: {part1_arr.shape}, part2: {part2_arr.shape}")
        print(f"Graph features: {len(part3_atom_fea_list)} structures with variable atom counts")

def main():
    """
    Main function to run the PIH feature generation from the command line.
    """
    parser = argparse.ArgumentParser(description="Generate Persistent Interaction Homology (PIH) features from structure files (CIF or XYZ).")
    parser.add_argument("input_path", type=str, help="Path to input file or directory, or folder of .npz files for combining")
    parser.add_argument("--output-file", type=str, default=None, help="Path to save the output features as a .npz file")
    parser.add_argument("--file-type", type=str, default='cif', choices=['cif', 'xyz'], help="Type of input files (default: cif)")
    parser.add_argument("--abc-norm", type=float, default=64.0, help="Target lattice parameter length for normalization.")
    parser.add_argument("--structure-max-dist", type=float, default=25.0, help="Maximum distance for entire structure Alpha complex")
    parser.add_argument("--cluster-max-dist", type=float, default=25.0, help="Maximum distance for cluster Alpha complex")
    parser.add_argument("--interaction-max-dist", type=float, default=10.0, help="Maximum distance for interaction Alpha complex")
    parser.add_argument("--structure-max-dim", type=int, default=2, help="Maximum dimension for entire structure PH")
    parser.add_argument("--cluster-max-dim", type=int, default=2, help="Maximum dimension for cluster PH")
    parser.add_argument("--interaction-max-dim", type=int, default=1, help="Maximum dimension for interaction PH")
    parser.add_argument("--interaction-complex0-max-dim", type=int, default=0, help="Max dimension for first alpha complex in interaction")
    parser.add_argument("--interaction-complex1-max-dim", type=int, default=2, help="Max dimension for second alpha complex in interaction")
    parser.add_argument("--structure-bin-start", type=float, default=0.0, help="Start of binning for entire structure PH")
    parser.add_argument("--structure-bin-end", type=float, default=25.0, help="End of binning for entire structure PH")
    parser.add_argument("--structure-bin-step", type=float, default=0.1, help="Step size for entire structure PH binning")
    parser.add_argument("--cluster-bin-start", type=float, default=0.0, help="Start of binning for cluster PH")
    parser.add_argument("--cluster-bin-end", type=float, default=25.0, help="End of binning for cluster PH")
    parser.add_argument("--cluster-bin-step", type=float, default=0.1, help="Step size for cluster PH binning")
    parser.add_argument("--interaction-bin-start", type=float, default=0.0, help="Start of binning for interaction PH")
    parser.add_argument("--interaction-bin-end", type=float, default=5.0, help="End of binning for interaction PH")
    parser.add_argument("--interaction-bin-step", type=float, default=0.02, help="Step size for interaction PH binning")
    parser.add_argument("--graph-radius", type=float, default=8.0, help="Cutoff radius for neighbor search in graph construction")
    parser.add_argument("--graph-max-num-nbr", type=int, default=12, help="Maximum number of neighbors per atom")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of worker threads for parallel processing")
    parser.add_argument("--combine", action='store_true', help="Combine separate .npz feature files in a folder into one .npz file")

    args = parser.parse_args()

    is_dir = os.path.isdir(args.input_path)

    # Set default output file if not provided
    if args.output_file is None:
        if is_dir:
            args.output_file = os.path.join(args.input_path, 'features.npz')
        else:
            base_name = os.path.splitext(os.path.basename(args.input_path))[0]
            args.output_file = f"{base_name}_features.npz"

    generator = PIHGenerator(
        abc_norm=args.abc_norm,
        structure_max_dist=args.structure_max_dist,
        cluster_max_dist=args.cluster_max_dist,
        interaction_max_dist=args.interaction_max_dist,
        structure_max_dim=args.structure_max_dim,
        cluster_max_dim=args.cluster_max_dim,
        interaction_max_dim=args.interaction_max_dim,
        interaction_complex0_max_dim=args.interaction_complex0_max_dim,
        interaction_complex1_max_dim=args.interaction_complex1_max_dim,
        structure_bin_start=args.structure_bin_start,
        structure_bin_end=args.structure_bin_end,
        structure_bin_step=args.structure_bin_step,
        cluster_bin_start=args.cluster_bin_start,
        cluster_bin_end=args.cluster_bin_end,
        cluster_bin_step=args.cluster_bin_step,
        interaction_bin_start=args.interaction_bin_start,
        interaction_bin_end=args.interaction_bin_end,
        interaction_bin_step=args.interaction_bin_step,
        graph_radius=args.graph_radius,
        graph_max_num_nbr=args.graph_max_num_nbr,
        vec_dtype=np.uint16
    )

    if args.combine:
        generator.combine_feature_files(args.input_path, args.output_file)
        return

    # Process input
    if is_dir:
        generator.process_directory(args.input_path, args.output_file, args.file_type, args.num_workers)
    else:
        features = generator.process_file(args.input_path, args.file_type)
        if features is not None:
            np.savez_compressed(args.output_file, **features)
            print(f"Features saved to {args.output_file}")

if __name__ == '__main__':
    # example for xyz file feature generation: python /Users/dongchen/Research_folder/Projects/MOF_paper/code_folder_large/ITTransformer/scripts/func_generate_pih.py /Users/dongchen/Research_folder/Projects/MOF_paper/code_folder_large/example_data/ABETIN_clean_normal.xyz --output-file features.npz --file-type xyz
    # example for cif file feature generation: python /Users/dongchen/Research_folder/Projects/MOF_paper/code_folder_large/ITTransformer/scripts/func_generate_pih.py /Users/dongchen/Research_folder/Projects/MOF_paper/code_folder_large/ITTransformer/examples/MITPAJ_FSR.cif --output-file features.npz --file-type cif
    main()