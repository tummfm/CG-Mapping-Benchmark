import functools
from copy import deepcopy

import math
from typing import Callable, Optional, Union, Set
from collections import OrderedDict

import numpy as onp

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp

import jax.nn as jax_nn

from jax_md import space, partition, nn, util as md_util

from typing import Tuple, Any, Callable

from jax_md_mod.model import sparse_graph

from .utils import safe_norm
from .layers import (
    EquivariantProductBasisLayer,
    InteractionLayer,
    LinearNodeEmbeddingLayer,
    LinearReadoutLayer,
    NonLinearReadoutLayer,
)
from jax_md_mod.model.layers import (
    RadialBesselLayer,
)

from chemutils.models.layers import AtomicEnergyLayer, CELLI



class MACE(hk.Module):
    def __init__(
        self,
        *,
        output_irreps: e3nn.Irreps,  # Irreps of the output, default 1x0e
        num_interactions: int,  # Number of interactions (layers), default 2
        hidden_irreps: e3nn.Irreps,  # 256x0e or 128x0e + 128x1o
        readout_mlp_irreps: e3nn.Irreps,  # Hidden irreps of the MLP in last readout, default 16x0e
        avg_num_neighbors: float,
        num_species: int,
        num_features: int = None,  # Number of features per node, default gcd of hidden_irreps multiplicities
        n_radial_basis: int = 8,  # Number of radial basis functions
        envelope_p: int = 6,  # Order of the envelope polynomial
        max_ell: int = 3,  # Max spherical harmonic degree, default 3
        epsilon: Optional[float] = None,
        correlation: int = 3,  # Correlation order at each layer (~ node_features^correlation), default 3
        activation: Callable = jax.nn.silu,  # activation function
        gate: Callable = jax.nn.sigmoid,  # gate function
        soft_normalization: Optional[float] = None,
        symmetric_tensor_product_basis: bool = True,
        off_diagonal: bool = False,
        interaction_irreps: Union[str, e3nn.Irreps] = "o3_restricted",  # or o3_full
        skip_connection_first_layer: bool = False,
        qeq: int = None,
        charge_embed_n_hidden=32,
        charge_embed_n_layers=3,
    ):
        super().__init__()

        output_irreps = e3nn.Irreps(output_irreps)
        hidden_irreps = e3nn.Irreps(hidden_irreps)
        readout_mlp_irreps = e3nn.Irreps(readout_mlp_irreps)

        self.charge_embed_n_hidden = charge_embed_n_hidden
        self.charge_embed_n_layers = charge_embed_n_layers

        if num_features is None:
            self.num_features = functools.reduce(
                math.gcd, (mul for mul, _ in hidden_irreps)
            )
            self.hidden_irreps = e3nn.Irreps(
                [(mul // self.num_features, ir) for mul, ir in hidden_irreps]
            )
        else:
            self.num_features = num_features
            self.hidden_irreps = hidden_irreps

        if interaction_irreps == "o3_restricted":
            self.interaction_irreps = e3nn.Irreps.spherical_harmonics(max_ell)
        elif interaction_irreps == "o3_full":
            self.interaction_irreps = e3nn.Irreps(e3nn.Irrep.iterator(max_ell))
        else:
            self.interaction_irreps = e3nn.Irreps(interaction_irreps)

        self.correlation = correlation
        self.readout_mlp_irreps = readout_mlp_irreps
        self.activation = activation
        self.gate = gate
        self.num_interactions = num_interactions

        self.qeq = qeq
        if qeq is not None:
            assert qeq < self.num_interactions, (
                f"Qeq layer must be inserted between 0 and {self.num_interactions}"
            )

        self.output_irreps = output_irreps
        self.num_species = num_species
        self.symmetric_tensor_product_basis = symmetric_tensor_product_basis
        self.off_diagonal = off_diagonal
        self.max_ell = max_ell
        self.soft_normalization = soft_normalization
        self.skip_connection_first_layer = skip_connection_first_layer

        epsilon = hk.get_parameter(
            "varepsilon", shape=(),
            init=hk.initializers.Constant(jnp.sqrt(avg_num_neighbors))
        )
        self.epsilon = 1 / jnp.sqrt(1 + epsilon ** 2)

        # Embeddings
        self.node_embedding = LinearNodeEmbeddingLayer(
            self.num_species, self.num_features * self.hidden_irreps
        )
        self.radial_embedding = RadialBesselLayer(
            cutoff=1.0, num_radial=n_radial_basis, envelope_p=envelope_p
        )
        self.envelope_p = envelope_p

    def __call__(
        self,
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
        node_species: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        node_mask: Optional[jnp.ndarray] = None,  # [n_nodes] only used for profiling
        is_training: bool = False,
        charge_fn: Optional[Callable] = None
    ) -> e3nn.IrrepsArray:
        assert vectors.ndim == 2 and vectors.shape[1] == 3
        assert node_species.ndim == 1
        assert senders.ndim == 1 and receivers.ndim == 1
        assert vectors.shape[0] == senders.shape[0] == receivers.shape[0]

        if node_mask is None:
            node_mask = jnp.ones(node_species.shape[0], dtype=jnp.bool_)

        # Embeddings
        node_feats = self.node_embedding(node_species).astype(vectors.dtype)  # [n_nodes, feature * irreps]
        rbf = self.radial_embedding(safe_norm(vectors.array, axis=-1))

        # jax.debug.print("MACE node_feats:{}", node_feats)

        # Interactions
        outputs = []
        node_outputs = None
        qeq_features = None
        for i in range(self.num_interactions):
            first = i == 0
            last = i == self.num_interactions - 1

            if i == self.qeq:
                x = node_feats.filter(keep=["0e", "0o"])
                V = node_feats.filter(drop=["0e", "0o"])

                w = jnp.concat([x[senders].array, rbf], axis=-1)
                w, _, qeq_features = CELLI(
                    charge_embed_n_hidden=self.charge_embed_n_hidden,
                    charge_embed_n_layers=self.charge_embed_n_layers,
                    num_species=self.num_species,
                    mlp_n_hidden=x.irreps.num_irreps,
                    mlp_n_layers=self.charge_embed_n_layers,
                    envelope_p=self.envelope_p,
                    mlp_activation=self.activation,
                    charge_eq_fn=charge_fn
                )(
                    vectors,
                    w,
                    None,
                    senders,
                    node_species,
                )
                y = jax.ops.segment_sum(w, senders, node_species.size)

                alpha = hk.get_parameter("alpha", shape=(), init=hk.initializers.Constant(1.0))
                alpha = jax_nn.softplus(alpha)

                x = (x + alpha * y) / (1 + alpha)

                node_feats = e3nn.concatenate([x, V]).simplify()

            hidden_irreps = (
                self.hidden_irreps
                if not last
                else self.hidden_irreps.filter(self.output_irreps)
            )

            node_outputs, node_feats = MACELayer(
                first=first,
                last=last,
                num_features=self.num_features,
                interaction_irreps=self.interaction_irreps,
                hidden_irreps=hidden_irreps,
                max_ell=self.max_ell,
                epsilon=self.epsilon,
                activation=self.activation,
                gate=self.gate,
                num_species=self.num_species,
                correlation=self.correlation,
                output_irreps=self.output_irreps,
                readout_mlp_irreps=self.readout_mlp_irreps,
                symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
                off_diagonal=self.off_diagonal,
                soft_normalization=self.soft_normalization,
                skip_connection_first_layer=self.skip_connection_first_layer,
                name=f"layer_{i}",
            )(
                vectors,
                node_feats,
                node_species,
                rbf,
                senders,
                receivers,
                node_mask,
                is_training
            )
            outputs += [node_outputs]  # list of [n_nodes, output_irreps]

        return node_outputs, qeq_features, e3nn.stack(outputs, axis=1)  # [n_nodes, num_interactions, output_irreps]


class MACELayer(hk.Module):
    def __init__(
        self,
        *,
        first: bool,
        last: bool,
        num_features: int,
        interaction_irreps: e3nn.Irreps,
        hidden_irreps: e3nn.Irreps,
        activation: Callable,
        gate: Callable,
        num_species: int,
        epsilon: Optional[float],
        name: Optional[str],
        # InteractionBlock:
        max_ell: int,
        # EquivariantProductBasisBlock:
        correlation: int,
        symmetric_tensor_product_basis: bool,
        off_diagonal: bool,
        soft_normalization: Optional[float],
        # ReadoutBlock:
        output_irreps: e3nn.Irreps,
        readout_mlp_irreps: e3nn.Irreps,
        skip_connection_first_layer: bool = False,
    ) -> None:
        super().__init__(name=name)

        # self.dropout = e3nn.haiku.Dropout(p=0.5)

        self.first = first
        self.last = last
        self.num_features = num_features
        self.interaction_irreps = interaction_irreps
        self.hidden_irreps = hidden_irreps
        self.max_ell = max_ell
        self.activation = activation
        self.gate = gate
        self.num_species = num_species
        self.epsilon = epsilon
        self.correlation = correlation
        self.output_irreps = output_irreps
        self.readout_mlp_irreps = readout_mlp_irreps
        self.symmetric_tensor_product_basis = symmetric_tensor_product_basis
        self.off_diagonal = off_diagonal
        self.soft_normalization = soft_normalization
        self.skip_connection_first_layer = skip_connection_first_layer

    def __call__(
        self,
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        node_specie: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        radial_embedding: jnp.ndarray,  # [n_edges, radial_embedding_dim]
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
        node_mask: Optional[jnp.ndarray] = None,  # [n_nodes] only used for profiling
        is_training: bool = False,
    ):
        if node_mask is None:
            node_mask = jnp.ones(node_specie.shape[0], dtype=jnp.bool_)

        sc = None
        if not self.first or self.skip_connection_first_layer:
            sc = e3nn.haiku.Linear(
                self.num_features * self.hidden_irreps,
                num_indexed_weights=self.num_species,
                name="skip_tp",
            )(
                node_specie, node_feats
            )  # [n_nodes, feature * hidden_irreps]

            # print(f"Use dropout for hidden irreps")
            # sc = self.dropout(hk.next_rng_key(), sc, is_training=is_training)

        node_feats = InteractionLayer(
            target_irreps=self.num_features * self.interaction_irreps,
            epsilon=self.epsilon,
            max_ell=self.max_ell,
            activation=self.activation,
        )(
            vectors=vectors,
            node_feats=node_feats,
            radial_embedding=radial_embedding,
            receivers=receivers,
            senders=senders,
        )

        node_feats *= self.epsilon

        if self.first:
            # Selector TensorProduct
            node_feats = e3nn.haiku.Linear(
                self.num_features * self.interaction_irreps,
                num_indexed_weights=self.num_species,
                name="skip_tp_first",
            )(node_specie, node_feats)

        node_feats = EquivariantProductBasisLayer(
            target_irreps=self.num_features * self.hidden_irreps,
            correlation=self.correlation,
            num_species=self.num_species,
            symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
            off_diagonal=self.off_diagonal,
        )(node_feats=node_feats, node_specie=node_specie)

        # node_feats = self.dropout(hk.next_rng_key(), node_feats, is_training=is_training)

        if self.soft_normalization is not None:

            def phi(n):
                n = n / self.soft_normalization
                return 1.0 / (1.0 + n * e3nn.sus(n))

            node_feats = e3nn.norm_activation(
                node_feats, [phi] * len(node_feats.irreps)
            )

        if sc is not None:
            node_feats = node_feats + sc  # [n_nodes, feature * hidden_irreps]

        if not self.last:
            node_outputs = LinearReadoutLayer(self.output_irreps)(
                node_feats
            )  # [n_nodes, output_irreps]
        else:  # Non linear readout for last layer
            node_outputs = NonLinearReadoutLayer(
                self.readout_mlp_irreps,
                self.output_irreps,
                activation=self.activation,
                gate=self.gate
            )(
                node_feats
            )  # [n_nodes, output_irreps]

        return node_outputs, node_feats


def mace_neighborlist_pp(displacement: space.DisplacementFn,
                         r_cutoff: float,
                         n_species: int = 100,
                         positions_test: jnp.ndarray = None,
                         neighbor_test: partition.NeighborList = None,
                         max_edge_multiplier: float = 1.25,
                         max_edges=None,
                         avg_num_neighbors: float = None,
                         mode: str = "energy",
                         per_particle: bool = False,
                         positive_species: bool = False,
                         **mace_kwargs
                         ) -> Tuple[nn.InitFn, Callable[[Any, md_util.Array],
                                                               md_util.Array]]:
    """NequIP model for property prediction.

    Args:
        displacement: Jax_md displacement function
        r_cutoff: Radial cut-off distance of DimeNetPP and the neighbor list
        n_species: Number of different atom species the network is supposed
            to process.
        positions_test: Sample positions to estimate max_edges / max_angles.
            Needs to be provided to enable capping.
        neighbor_test: Sample neighborlist to estimate max_edges / max_angles.
            Needs to be provided to enable capping.
        max_edge_multiplier: Multiplier for initial estimate of maximum edges.
        max_edges: Expected maximum of valid edges.
        avg_num_neighbors: Average number of neighbors per particle. Guessed
            if positions_test and neighbor_test are provided.
        mode: Prediction mode of the model. If "property_prediction" (default),
            returns the learned node features. If "energy_prediction", returns
            the total energy of the system.
        per_particle: Return per-particle energies instead of total energy.
        positive_species: True if the smallest occurring species is 1, e.g., in
            case of atomic numbers.
        mace_kwargs: Kwargs to change the default structure of NequIP.
            For definition of the kwargs, see NequIP.


    Returns:
        A tuple of 2 functions: A init_fn that initializes the model parameters
        and an energy function that computes the energy for a particular state
        given model parameters. The energy function requires the same input as
        other energy functions with neighbor lists in jax_md.energy.
    """

    r_cutoff = jnp.array(r_cutoff, dtype=md_util.f32)

    # Checking only necessary if neighbor list is dense
    _avg_num_neighbors = None
    if positions_test is not None and neighbor_test is not None:
        if neighbor_test.format == partition.Dense:
            print('Capping edges and triplets. Beware of overflow, which is'
                  ' currently not being detected.')

            testgraph, _ = sparse_graph.sparse_graph_from_neighborlist(
                displacement, positions_test, neighbor_test, r_cutoff)
            max_edges = jnp.int32(jnp.ceil(testgraph.n_edges * max_edge_multiplier))

            # cap maximum edges and angles to avoid overflow from multiplier
            n_particles, n_neighbors = neighbor_test.idx.shape
            max_edges = min(max_edges, n_particles * n_neighbors)

            print(f"Estimated max. {max_edges} edges.")

            _avg_num_neighbors = testgraph.n_edges / n_particles
        else:
            n_particles = neighbor_test.idx.shape[0]
            _avg_num_neighbors = onp.sum(neighbor_test.idx[0] < n_particles)
            _avg_num_neighbors /= n_particles

    if avg_num_neighbors is None:
        avg_num_neighbors = _avg_num_neighbors
    assert avg_num_neighbors is not None, (
        "Average number of neighbors not set and no test graph was provided."
    )

    @hk.without_apply_rng
    @hk.transform
    def model(position: md_util.Array,
              neighbor: partition.NeighborList,
              species: md_util.Array = None,
              mask: md_util.Array = None,
              **dynamic_kwargs):
        if species is None:
            print(f"[MACE] Use default species")
            species = jnp.zeros(position.shape[0], dtype=jnp.int32)
        elif positive_species:
            species -= 1
        if mask is None:
            print(f"[MACE] Use default mask")
            mask = jnp.ones(position.shape[0], dtype=jnp.bool_)

        # Compute the displacements for all edges
        dyn_displacement = functools.partial(displacement, **dynamic_kwargs)

        if neighbor.format == partition.Dense:
            graph, _ = sparse_graph.sparse_graph_from_neighborlist(
                dyn_displacement, position, neighbor, r_cutoff,
                species, max_edges=max_edges, species_mask=mask
            )
            senders = graph.idx_i
            receivers = graph.idx_j
        else:
            assert neighbor.idx.shape == (2, neighbor.idx.shape[1]), "Neighbor list has wrong shape."
            senders, receivers = neighbor.idx

        # Set invalid edges to the cutoff to avoid numerical issues
        vectors = jax.vmap(dyn_displacement)(position[senders], position[receivers])
        vectors = jnp.where(
            jnp.logical_and(
                senders < position.shape[0],
                receivers < position.shape[0]
            )[:, jnp.newaxis], vectors, r_cutoff / jnp.sqrt(3))
        vectors /= r_cutoff

        # Sort vectors by length and remove up to max_edges edges
        lengths = jnp.linalg.norm(vectors, axis=-1)
        sort_idx = jnp.argsort(lengths)
        vectors = vectors[sort_idx][:max_edges]
        senders = senders[sort_idx][:max_edges]
        receivers = receivers[sort_idx][:max_edges]

        vectors = e3nn.IrrepsArray(e3nn.Irreps("1o"), vectors)

        net = MACE(
            avg_num_neighbors=avg_num_neighbors,
            num_species=n_species,
            **mace_kwargs
        )

        features, _, _ = net(
            vectors, senders, receivers, species, mask)

        if mode == "energy":
            per_atom_energies, = features.array.T
            per_atom_energies = AtomicEnergyLayer(n_species)(
                per_atom_energies, species)
            per_atom_energies *= mask

            if per_particle:
                return per_atom_energies
            else:
                return md_util.high_precision_sum(per_atom_energies)

        elif mode == "property_prediction":
            return features
        else:
            raise NotImplementedError(f"Mode {mode} not implemented.")

    return jax.jit(model.init), jax.jit(model.apply)


def mace_qeq_neighborlist_pp(displacement: space.DisplacementFn,
                             charge_eq_fn: Callable,
                             r_cutoff: float,
                             n_species: int = 100,
                             positions_test: jnp.ndarray = None,
                             neighbor_test: partition.NeighborList = None,
                             max_edge_multiplier: float = 1.25,
                             max_edges=None,
                             avg_num_neighbors: float = None,
                             mode: str = "energy",
                             per_particle: bool = False,
                             positive_species: bool = False,
                             learn_radius: bool = False,
                             **mace_kwargs
                             ) -> Tuple[nn.InitFn, Callable[[Any, md_util.Array],
                                                               md_util.Array]]:
    """NequIP model for property prediction.

    Args:
        displacement: Jax_md displacement function
        charge_eq_fn: Function computing charges via the QEM method based
            on electronegativities, hardnesses, and radii predicted by
            the neural network.
        r_cutoff: Radial cut-off distance of DimeNetPP and the neighbor list
        n_species: Number of different atom species the network is supposed
            to process.
        positions_test: Sample positions to estimate max_edges / max_angles.
            Needs to be provided to enable capping.
        neighbor_test: Sample neighborlist to estimate max_edges / max_angles.
            Needs to be provided to enable capping.
        max_edge_multiplier: Multiplier for initial estimate of maximum edges.
        max_edges: Expected maximum of valid edges.
        nequip_escn: Use NequIPESCN instead of NequIP (more computational
            efficient).
        avg_num_neighbors: Average number of neighbors per particle. Guessed
            if positions_test and neighbor_test are provided.
        mode: Prediction mode of the model. If "property_prediction" (default),
            returns the learned node features. If "energy_prediction", returns
            the total energy of the system.
        per_particle: Return per-particle energies instead of total energy.
        positive_species: True if the smallest occurring species is 1, e.g., in
            case of atomic numbers.
        learn_radius: True if the radius of the atom for electrostatic
            interactions should be learned.
        mace_kwargs: Kwargs to change the default structure of NequIP.
            For definition of the kwargs, see NequIP.


    Returns:
        A tuple of 2 functions: A init_fn that initializes the model parameters
        and an energy function that computes the energy for a particular state
        given model parameters. The energy function requires the same input as
        other energy functions with neighbor lists in jax_md.energy.
    """

    r_cutoff = jnp.array(r_cutoff, dtype=md_util.f32)

    # Checking only necessary if neighbor list is dense
    _avg_num_neighbors = None
    if positions_test is not None and neighbor_test is not None:
        if neighbor_test.format == partition.Dense:
            print('Capping edges and triplets. Beware of overflow, which is'
                  ' currently not being detected.')

            testgraph, _ = sparse_graph.sparse_graph_from_neighborlist(
                displacement, positions_test, neighbor_test, r_cutoff)
            max_edges = jnp.int32(jnp.ceil(testgraph.n_edges * max_edge_multiplier))

            # cap maximum edges and angles to avoid overflow from multiplier
            n_particles, n_neighbors = neighbor_test.idx.shape
            max_edges = min(max_edges, n_particles * n_neighbors)

            print(f"Estimated max. {max_edges} edges.")

            _avg_num_neighbors = testgraph.n_edges / n_particles
        else:
            n_particles = neighbor_test.idx.shape[0]
            _avg_num_neighbors = onp.sum(neighbor_test.idx[0] < n_particles)
            _avg_num_neighbors /= n_particles

    if avg_num_neighbors is None:
        avg_num_neighbors = _avg_num_neighbors
    assert avg_num_neighbors is not None, (
        "Average number of neighbors not set and no test graph was provided."
    )

    @hk.without_apply_rng
    @hk.transform
    def model(position: md_util.Array,
              neighbor: partition.NeighborList,
              species: md_util.Array = None,
              mask: md_util.Array = None,
              **dynamic_kwargs):
        if species is None:
            print(f"[MACE] Use default species")
            species = jnp.zeros(position.shape[0], dtype=jnp.int32)
        elif positive_species:
            species -= 1
        if mask is None:
            print(f"[MACE] Use default mask")
            mask = jnp.ones(position.shape[0], dtype=jnp.bool_)

        # Compute the displacements for all edges
        dyn_displacement = functools.partial(displacement, **dynamic_kwargs)

        if neighbor.format == partition.Dense:
            graph, _ = sparse_graph.sparse_graph_from_neighborlist(
                dyn_displacement, position, neighbor, r_cutoff,
                species, max_edges=max_edges, species_mask=mask
            )
            senders = graph.idx_i
            receivers = graph.idx_j
        else:
            assert neighbor.idx.shape == (2, neighbor.idx.shape[1]), "Neighbor list has wrong shape."
            senders, receivers = neighbor.idx

        invalid_idx = position.shape[0]

        # Set invalid edges to the cutoff to avoid numerical issues
        vectors = jax.vmap(dyn_displacement)(position[senders], position[receivers])
        vectors = jnp.where(
            jnp.logical_and(
                senders < invalid_idx,
                receivers < position.shape[0]
            )[:, jnp.newaxis], vectors, r_cutoff)
        vectors /= r_cutoff

        # Sort vectors by length and remove up to max_edges edges
        lengths = jnp.linalg.norm(vectors, axis=-1)
        sort_idx = jnp.argsort(lengths)
        vectors = vectors[sort_idx][:max_edges]
        senders = senders[sort_idx][:max_edges]
        receivers = receivers[sort_idx][:max_edges]

        vectors = e3nn.IrrepsArray(
            e3nn.Irreps("1o"), vectors
        )

        net = MACE(
            avg_num_neighbors=avg_num_neighbors,
            num_species=n_species,
            **mace_kwargs
        )

        def _charge_eq_fn(gammas, chis, hardness):
            assert "radius" in dynamic_kwargs.keys(), "Radius not in dynamic_kwargs."

            if learn_radius:
                gammas *= dynamic_kwargs["radius"]
            else:
                gammas = dynamic_kwargs["radius"]

            _, charges = charge_eq_fn(
                position, neighbor, gammas, chi=chis, idmp=hardness,
                mask=mask, **dynamic_kwargs
            )

            # Do not optimize hardness and gammas on energy and force (only
            # indirectly through charges)
            pot = charge_eq_fn(
                position, neighbor, jax.lax.stop_gradient(gammas), mask=mask,
                charge=charges, **dynamic_kwargs
            )

            return charges, pot

        features, qeq_features, _ = net(
            vectors, senders, receivers, species, mask, charge_fn=_charge_eq_fn)

        if mode in ["energy", "energy_and_charge"]:
            per_atom_energies, = features.array.T
            per_atom_energies = AtomicEnergyLayer(n_species)(
                per_atom_energies, species)
            per_atom_energies *= mask

            charges, elec_pot = qeq_features
            total_pot = elec_pot + md_util.high_precision_sum(per_atom_energies)

            if per_particle:
                return per_atom_energies
            else:
                return total_pot, charges

        elif mode == "property_prediction":
            return features
        else:
            raise NotImplementedError(f"Mode {mode} not implemented.")

    return jax.jit(model.init), jax.jit(model.apply)
