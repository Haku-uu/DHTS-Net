
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class ExpertMLP(nn.Module):
    """Two-layer MLP expert used by the Mixture-of-Experts module."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: type[nn.Module] = nn.GELU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class SparseExpertDispatcher:
    """Dispatch tokens to the selected experts and merge their outputs."""

    def __init__(self, num_experts: int, gates: torch.Tensor) -> None:
        self._gates = gates
        self._num_experts = num_experts

        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()

        gates_expanded = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_expanded, 1, self._expert_index)

    def dispatch(self, inputs: torch.Tensor) -> Sequence[torch.Tensor]:
        inputs_expanded = inputs[self._batch_index].squeeze(1)
        return torch.split(inputs_expanded, self._part_sizes, dim=0)

    def combine(self, expert_outputs: Sequence[torch.Tensor], multiply_by_gates: bool = True) -> torch.Tensor:
        if len(expert_outputs) == 0:
            raise RuntimeError("No expert outputs were provided to SparseExpertDispatcher.combine().")

        stitched = torch.cat(expert_outputs, dim=0)
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)

        combined = torch.zeros(
            self._gates.size(0),
            expert_outputs[-1].size(1),
            requires_grad=True,
            device=stitched.device,
            dtype=stitched.dtype,
        )
        combined = combined.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self) -> Sequence[torch.Tensor]:
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class MixtureOfExpertsFeedForwardNetwork(nn.Module):
    """Apply Mixture of Experts to the last feature dimension of an arbitrary tensor."""

    def __init__(self, model_dim: int, num_experts: int = 20, hidden_dim: Optional[int] = None, top_k: int = 2) -> None:
        super().__init__()
        hidden_dim = hidden_dim or 4 * model_dim
        self.model_dim = model_dim
        self.moe = MixtureOfExperts(
            input_size=model_dim,
            output_size=model_dim,
            num_experts=num_experts,
            hidden_size=hidden_dim,
            noisy_gating=True,
            top_k=top_k,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.reshape(-1, self.model_dim)
        y_flat = self.moe(x_flat)
        return y_flat.view(*original_shape)

    def auxiliary_loss(self) -> torch.Tensor:
        if self.moe.last_load_balancing_loss is None:
            return torch.tensor(0.0, device=self.moe.w_gate.device)
        return self.moe.last_load_balancing_loss

class MultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head attention."""

    def __init__(self, model_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads."
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.query_projection = nn.Linear(model_dim, model_dim)
        self.key_projection = nn.Linear(model_dim, model_dim)
        self.value_projection = nn.Linear(model_dim, model_dim)
        self.output_projection = nn.Linear(model_dim, model_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size = query.shape[0]

        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)

        output = torch.cat(torch.split(output, batch_size, dim=0), dim=-1)
        return self.output_projection(output)


class StructuralTemporalPath(nn.Module):
    """Structural Temporal Path for global periodic temporal dependencies."""

    def __init__(self, model_dim: int, feed_forward_dim: int = 256, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(model_dim=model_dim, num_heads=num_heads)
        self.moe_ffn = MixtureOfExpertsFeedForwardNetwork(model_dim=model_dim, hidden_dim=feed_forward_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, dim: int = -2) -> torch.Tensor:
        x = x.transpose(dim, -2)
        residual = x
        output = self.attention(x, x, x)
        output = self.dropout1(output)
        output = self.norm1(residual + output)

        residual = output
        output = self.moe_ffn(output)
        output = self.dropout2(output)
        output = self.norm2(residual + output)
        return output.transpose(dim, -2)


class DiffusiveStructuralPath(nn.Module):
    """Diffusive Structural Path for topology-guided spatial dependencies."""

    def __init__(self, model_dim: int, feed_forward_dim: int = 256, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(model_dim=model_dim, num_heads=num_heads)
        self.diffusion_gate = nn.Sequential(nn.Linear(model_dim, model_dim), nn.GELU())
        self.moe_ffn = MixtureOfExpertsFeedForwardNetwork(model_dim=model_dim, hidden_dim=feed_forward_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, dim: int = -2) -> torch.Tensor:
        x = x.transpose(dim, -2)
        residual = x
        return output.transpose(dim, -2)


class CrossPathAttention(nn.Module):
    """Attention used by cross-path interaction modules."""

    def __init__(self, model_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads."
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.query_projection = nn.Linear(model_dim, model_dim)
        self.key_projection = nn.Linear(model_dim, model_dim)
        self.value_projection = nn.Linear(model_dim, model_dim)
        self.output_projection = nn.Linear(model_dim, model_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:

        return self.output_projection(output)


class CrossTemporalInteraction(nn.Module):
    """Cross-Temporal Interaction between structural and heterogeneous temporal paths."""

    def __init__(self, model_dim: int, feed_forward_dim: int = 256, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = CrossPathAttention(model_dim=model_dim, num_heads=num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, structural_hidden: torch.Tensor, heterogeneous_hidden: Optional[torch.Tensor] = None, dim: int = -2) -> torch.Tensor:
        structural_hidden = structural_hidden.transpose(dim, -2)
        residual = structural_hidden


        residual = output
        output = self.feed_forward(output)
        output = self.dropout2(output)
        output = self.norm2(residual + output)
        return output.transpose(dim, -2)

class HeterogeneousTemporalAttention(nn.Module):

    def __init__(
        self,
        model_dim: int,
        traffic_dim: int = 24,
        num_heads: int = 8,
        pattern_matrix: Optional[torch.Tensor] = None,
        history_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads."
        self.model_dim = model_dim
        self.traffic_dim = traffic_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.history_size = history_size

        if pattern_matrix is not None:
            self.pattern_projection = nn.Linear(pattern_matrix.shape[-1], traffic_dim)
            self.register_buffer("pattern_matrix", pattern_matrix.float())
        else:
            self.pattern_projection = None
            self.pattern_matrix = None

        self.query_projection = nn.Linear(model_dim, model_dim)
        self.key_projection = nn.Linear(model_dim, model_dim)
        self.value_projection = nn.Linear(model_dim, model_dim)
        self.output_projection = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def generate_temporal_mask(similarity: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, seq_len, _ = similarity.shape

        temporal_mask = torch.matmul(temporal_weights, temporal_weights.transpose(1, 2))
        return temporal_mask.unsqueeze(1).expand(-1, num_nodes, -1, -1)

    def compute_pattern_similarity(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, seq_len, _ = x.shape
        traffic_features = x[..., : self.traffic_dim]
        traffic_features = traffic_features.reshape(batch_size * num_nodes, seq_len, self.traffic_dim)
        traffic_features = traffic_features.permute(0, 2, 1).unsqueeze(-1)



        if self.pattern_matrix is not None and self.pattern_projection is not None:
            pattern_library = self.pattern_projection(self.pattern_matrix.to(query_patterns.device))
            similarity = torch.einsum("bhd,phd->bph", query_patterns, pattern_library)
            similarity = similarity.mean(dim=-1)
            similarity = similarity.view(batch_size, num_nodes, -1, pattern_library.size(0))
            pad_size = (self.history_size - 1) // 2
            return F.pad(similarity, pad=(0, 0, pad_size, pad_size), mode="constant")

        return torch.zeros(batch_size, num_nodes, seq_len, 1, device=x.device, dtype=x.dtype)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, seq_len, _ = query.shape

        similarity = self.compute_pattern_similarity(query)
        temporal_mask = self.generate_temporal_mask(similarity)

        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)


        attention_weight = F.softmax(attention_score, dim=-1)
        attention_weight = self.dropout(attention_weight)
        output = attention_weight @ value
        output = torch.cat(torch.split(output, batch_size, dim=0), dim=-1)
        output = output.reshape(batch_size, num_nodes, seq_len, self.model_dim)
        return self.output_projection(output)


class HeterogeneousTemporalPath(nn.Module):

    def __init__(
        self,
        model_dim: int,
        feed_forward_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        pattern_matrix: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.attention = HeterogeneousTemporalAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            pattern_matrix=pattern_matrix,
            dropout=dropout,
        )
        self.moe_ffn = MixtureOfExpertsFeedForwardNetwork(model_dim=model_dim, hidden_dim=feed_forward_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, dim: int = -2) -> torch.Tensor:
        x = x.transpose(dim, -2)
        residual = x
        output = self.attention(x, x, x)
        output = self.dropout1(output)
        output = self.norm1(residual + output)

        residual = output
        output = self.moe_ffn(output)
        output = self.dropout2(output)
        output = self.norm2(residual + output)
        return output.transpose(dim, -2)


class SpatialHeterogeneityAttention(nn.Module):
    """Semantic-similarity-guided spatial attention."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int = 8,
        num_nodes: int = 307,
        semantic_matrix: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads."
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.num_nodes = num_nodes

        if semantic_matrix is not None:
            self.register_buffer("semantic_matrix", semantic_matrix.float())
        else:
            self.semantic_matrix = None

        self.query_projection = nn.Linear(model_dim, model_dim)
        self.key_projection = nn.Linear(model_dim, model_dim)
        self.value_projection = nn.Linear(model_dim, model_dim)
        self.output_projection = nn.Linear(model_dim, model_dim)

    def generate_spatial_semantic_mask(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, _, _ = x.shape


    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_nodes, _ = query.shape
        spatial_semantic_mask = self.generate_spatial_semantic_mask(query)



        attention_weight = F.softmax(attention_score, dim=-1)
        output = attention_weight @ value
        output = torch.cat(torch.split(output, batch_size, dim=0), dim=-1)
        output = output.reshape(batch_size, seq_len, num_nodes, self.model_dim)
        return self.output_projection(output)


class SpatialHeterogeneityPath(nn.Module):
    """Spatial Heterogeneity Path for semantic-similarity relations."""

    def __init__(
        self,
        model_dim: int,
        feed_forward_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_nodes: int = 307,
        semantic_matrix: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.attention = SpatialHeterogeneityAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            num_nodes=num_nodes,
            semantic_matrix=semantic_matrix,
        )
        self.moe_ffn = MixtureOfExpertsFeedForwardNetwork(model_dim=model_dim, hidden_dim=feed_forward_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, dim: int = -2) -> torch.Tensor:
        x = x.transpose(dim, -2)
        residual = x
        output = self.attention(x, x, x)
        output = self.dropout1(output)
        output = self.norm1(residual + output)

        residual = output
        output = self.moe_ffn(output)
        output = self.dropout2(output)
        output = self.norm2(residual + output)
        return output.transpose(dim, -2)


class UnifiedSpatiotemporalEmbedding(nn.Module):
    """Unified Spatiotemporal Embedding in Section 3.3 of the paper."""

    def __init__(
        self,
        num_nodes: int,
        in_steps: int,
        input_dim: int,
        input_embedding_dim: int,
        tod_embedding_dim: int,
        dow_embedding_dim: int,
        adaptive_embedding_dim: int,
        steps_per_day: int,
        steps_per_week: int,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.input_dim = input_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.steps_per_day = steps_per_day
        self.steps_per_week = steps_per_week

        if input_embedding_dim > 0:
            self.value_embedding = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.time_of_day_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.day_of_week_embedding = nn.Embedding(steps_per_week, dow_embedding_dim)
        if adaptive_embedding_dim > 0:
            adaptive_embedding = torch.empty(in_steps, num_nodes, adaptive_embedding_dim)
            self.adaptive_spatiotemporal_embedding = nn.Parameter(nn.init.xavier_uniform_(adaptive_embedding))

    @property
    def output_dim(self) -> int:
        return self.input_embedding_dim + self.tod_embedding_dim + self.dow_embedding_dim + self.adaptive_embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:


        if self.tod_embedding_dim > 0:
            time_of_day_embedding = self.time_of_day_embedding((time_of_day_index * self.steps_per_day).long())
            features.append(time_of_day_embedding)

        if self.dow_embedding_dim > 0:
            day_of_week_embedding = self.day_of_week_embedding(day_of_week_index.long())
            features.append(day_of_week_embedding)

        if self.adaptive_embedding_dim > 0:
            adaptive_embedding = self.adaptive_spatiotemporal_embedding.expand(
                size=(batch_size, *self.adaptive_spatiotemporal_embedding.shape)
            )
            features.append(adaptive_embedding)

        return torch.cat(features, dim=-1)


class TemporalDualPathAttention(nn.Module):

    def __init__(
        self,
        model_dim: int,
        feed_forward_dim: int,
        num_heads: int,
        dropout: float,
        pattern_matrix: Optional[torch.Tensor],
        num_structural_temporal_layers: int = 1,
        num_heterogeneous_temporal_layers: int = 1,
        num_cross_temporal_layers: int = 1,
        use_heterogeneous_temporal_path: bool = True,
        use_cross_temporal_interaction: bool = True,
    ) -> None:
        super().__init__()
        self.use_heterogeneous_temporal_path = use_heterogeneous_temporal_path
        self.use_cross_temporal_interaction = use_cross_temporal_interaction

        self.structural_temporal_path = nn.ModuleList(
            [
                StructuralTemporalPath(model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_structural_temporal_layers)
            ]
        )


    def forward(self, unified_hidden: torch.Tensor) -> torch.Tensor:
        structural_temporal_hidden = unified_hidden.clone()
        heterogeneous_temporal_hidden = unified_hidden.clone()

        for path_layer in self.structural_temporal_path:
            structural_temporal_hidden = path_layer(structural_temporal_hidden, dim=1)



        if self.use_cross_temporal_interaction:
            temporal_hidden = structural_temporal_hidden
            for interaction_layer in self.cross_temporal_interaction:
                temporal_hidden = interaction_layer(structural_temporal_hidden, heterogeneous_temporal_hidden, dim=1)
        else:
            temporal_hidden = structural_temporal_hidden

        return temporal_hidden


class ResidualMLP(nn.Module):
    """Residual MLP used by graph-feature fusion."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x) + x


class DiffusionGraphProjection(nn.Module):
    """Projection for forward/backward diffusion graphs."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x + self.fc2(x)


class DiffusiveStructuralGraphConstructor(nn.Module):
    """Build bidirectional diffusion representations from transition matrices."""

    def __init__(self, num_nodes: int, node_dim: int) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.forward_diffusion_projection = DiffusionGraphProjection(num_nodes, node_dim)
        self.backward_diffusion_projection = DiffusionGraphProjection(num_nodes, node_dim)

    def forward(self, transition_matrix: Sequence[torch.Tensor], batch_size: int, seq_len: int) -> torch.Tensor:
        device = next(self.parameters()).device


        forward_graph = forward_graph.expand(batch_size, seq_len, -1, -1)
        backward_graph = backward_graph.expand(batch_size, seq_len, -1, -1)
        return torch.cat([forward_graph, backward_graph], dim=-1)


class DiffusiveStructuralFeatureFusion(nn.Module):
    """Fuse diffusion-graph features with adaptive spatiotemporal embeddings."""

    def __init__(self, input_dim: int, output_dim: int, num_layers: int, dropout: float = 0.2) -> None:
        super().__init__()


    def forward(self, diffusion_graph: torch.Tensor, adaptive_graph: torch.Tensor, other_features: torch.Tensor) -> torch.Tensor:
        fusion_graph = torch.cat([diffusion_graph, adaptive_graph], dim=-1)
        fusion_features = self.fusion_model(fusion_graph)
        return torch.cat([other_features, fusion_features], dim=-1)


class SpatialDualPathDiffusionAttention(nn.Module):
    """Spatial Dual-Path Diffusion-Attention with Cross-Spatial Interaction."""

    def __init__(
        self,
        num_nodes: int,
        input_steps: int,
        model_dim: int,
        adaptive_embedding_dim: int,
        node_dim: int,
        feed_forward_dim: int,
        num_heads: int,
        dropout: float,
        semantic_matrix: Optional[torch.Tensor],
        num_diffusive_spatial_layers: int = 1,
        num_spatial_heterogeneity_layers: int = 1,
        num_cross_spatial_layers: int = 1,
        num_graph_fusion_mlp_layers: int = 2,
        use_spatial_heterogeneity_path: bool = True,
        use_cross_spatial_interaction: bool = True,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.input_steps = input_steps
        self.model_dim = model_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.node_dim = node_dim
        self.use_spatial_heterogeneity_path = use_spatial_heterogeneity_path
        self.use_cross_spatial_interaction = use_cross_spatial_interaction

        if node_dim > 0:
            self.diffusive_structural_graph_constructor = DiffusiveStructuralGraphConstructor(num_nodes, node_dim)
            self.diffusive_structural_feature_fusion = DiffusiveStructuralFeatureFusion(
                input_dim=adaptive_embedding_dim + 2 * node_dim,
                output_dim=adaptive_embedding_dim,
                num_layers=num_graph_fusion_mlp_layers,
            )

        self.diffusive_structural_path = nn.ModuleList(
            [
                DiffusiveStructuralPath(model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_diffusive_spatial_layers)
            ]
        )


    def forward(self, temporal_hidden: torch.Tensor, transition_matrix: Sequence[torch.Tensor]) -> torch.Tensor:
        batch_size, input_steps, _, _ = temporal_hidden.shape
        diffusive_structural_hidden = temporal_hidden.clone()
        spatial_heterogeneity_hidden = temporal_hidden.clone()


        if self.use_cross_spatial_interaction:
            spatial_hidden = diffusive_structural_hidden
            for interaction_layer in self.cross_spatial_interaction:
                spatial_hidden = interaction_layer(diffusive_structural_hidden, spatial_heterogeneity_hidden, dim=2)
        else:
            spatial_hidden = diffusive_structural_hidden

        return spatial_hidden


class MultiStepPredictionHead(nn.Module):
    """Multi-Step Prediction Head for joint long-horizon prediction."""

    def __init__(
        self,
        input_steps: int,
        output_steps: int,
        num_nodes: int,
        model_dim: int,
        output_dim: int,
        use_mixed_projection: bool = True,
    ) -> None:
        super().__init__()
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.num_nodes = num_nodes
        self.model_dim = model_dim
        self.output_dim = output_dim
        self.use_mixed_projection = use_mixed_projection

        if use_mixed_projection:
            self.joint_projection = nn.Linear(input_steps * model_dim, output_steps * output_dim)
        else:
            self.temporal_projection = nn.Linear(input_steps, output_steps)
            self.output_projection = nn.Linear(model_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        output = self.output_projection(output.transpose(1, 3))
        return output


class DHTSNet(nn.Module):
    """Dual-Heterogeneity Temporal-Spatial Network."""

    def __init__(
        self,
        num_nodes: int,
        in_steps: int,
        out_steps: int,
        steps_per_day: int,
        steps_per_week: int,
        input_dim: int,
        output_dim: int,
        input_embedding_dim: int,
        tod_embedding_dim: int,
        dow_embedding_dim: int,
        feed_forward_dim: int,
        num_heads: int,
        dropout: float,
        adaptive_embedding_dim: int,
        node_dim: int,
        transition_matrix: Sequence[torch.Tensor],
        pattern_matrix: Optional[torch.Tensor],
        semantic_matrix: Optional[torch.Tensor],
        num_structural_temporal_layers: int = 1,
        num_heterogeneous_temporal_layers: int = 1,
        num_diffusive_spatial_layers: int = 1,
        num_spatial_heterogeneity_layers: int = 1,
        num_cross_temporal_layers: int = 1,
        num_cross_spatial_layers: int = 1,
        num_graph_fusion_mlp_layers: int = 2,
        use_heterogeneous_temporal_path: bool = True,
        use_spatial_heterogeneity_path: bool = True,
        use_cross_temporal_interaction: bool = True,
        use_cross_spatial_interaction: bool = True,
        use_mixed_projection: bool = True,
        # Backward-compatible aliases. They are accepted so older yaml files still run.
        num_layers_t: Optional[int] = None,
        num_layers_s: Optional[int] = None,
        num_layers_c: Optional[int] = None,
        num_layers_mlp: Optional[int] = None,
        use_temporal_heterogeneity: Optional[bool] = None,
        use_spatial_heterogeneity: Optional[bool] = None,
        use_temporal_cross: Optional[bool] = None,
        use_spatial_cross: Optional[bool] = None,
        use_mixed_proj: Optional[bool] = None,
    ) -> None:
        super().__init__()

        if num_layers_t is not None:
            num_heterogeneous_temporal_layers = num_layers_t
        if num_layers_s is not None:
            num_structural_temporal_layers = num_layers_s
            num_diffusive_spatial_layers = num_layers_s
            num_spatial_heterogeneity_layers = num_layers_s
        if num_layers_c is not None:
            num_cross_temporal_layers = num_layers_c
            num_cross_spatial_layers = num_layers_c
        if num_layers_mlp is not None:
            num_graph_fusion_mlp_layers = num_layers_mlp
        if use_temporal_heterogeneity is not None:
            use_heterogeneous_temporal_path = use_temporal_heterogeneity
        if use_spatial_heterogeneity is not None:
            use_spatial_heterogeneity_path = use_spatial_heterogeneity
        if use_temporal_cross is not None:
            use_cross_temporal_interaction = use_temporal_cross
        if use_spatial_cross is not None:
            use_cross_spatial_interaction = use_spatial_cross
        if use_mixed_proj is not None:
            use_mixed_projection = use_mixed_proj

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.transition_matrix = [torch.as_tensor(matrix, dtype=torch.float32) for matrix in transition_matrix]

        self.unified_spatiotemporal_embedding = UnifiedSpatiotemporalEmbedding(
            num_nodes=num_nodes,
            in_steps=in_steps,
            input_dim=input_dim,
            input_embedding_dim=input_embedding_dim,
            tod_embedding_dim=tod_embedding_dim,
            dow_embedding_dim=dow_embedding_dim,
            adaptive_embedding_dim=adaptive_embedding_dim,
            steps_per_day=steps_per_day,
            steps_per_week=steps_per_week,
        )
        self.model_dim = self.unified_spatiotemporal_embedding.output_dim

        self.temporal_dual_path_attention = TemporalDualPathAttention(
            model_dim=self.model_dim,
            feed_forward_dim=feed_forward_dim,
            num_heads=num_heads,
            dropout=dropout,
            pattern_matrix=pattern_matrix,
            num_structural_temporal_layers=num_structural_temporal_layers,
            num_heterogeneous_temporal_layers=num_heterogeneous_temporal_layers,
            num_cross_temporal_layers=num_cross_temporal_layers,
            use_heterogeneous_temporal_path=use_heterogeneous_temporal_path,
            use_cross_temporal_interaction=use_cross_temporal_interaction,
        )

        self.spatial_dual_path_diffusion_attention = SpatialDualPathDiffusionAttention(
            num_nodes=num_nodes,
            input_steps=in_steps,
            model_dim=self.model_dim,
            adaptive_embedding_dim=adaptive_embedding_dim,
            node_dim=node_dim,
            feed_forward_dim=feed_forward_dim,
            num_heads=num_heads,
            dropout=dropout,
            semantic_matrix=semantic_matrix,
            num_diffusive_spatial_layers=num_diffusive_spatial_layers,
            num_spatial_heterogeneity_layers=num_spatial_heterogeneity_layers,
            num_cross_spatial_layers=num_cross_spatial_layers,
            num_graph_fusion_mlp_layers=num_graph_fusion_mlp_layers,
            use_spatial_heterogeneity_path=use_spatial_heterogeneity_path,
            use_cross_spatial_interaction=use_cross_spatial_interaction,
        )

        self.multi_step_prediction_head = MultiStepPredictionHead(
            input_steps=in_steps,
            output_steps=out_steps,
            num_nodes=num_nodes,
            model_dim=self.model_dim,
            output_dim=output_dim,
            use_mixed_projection=use_mixed_projection,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.multi_step_prediction_head(spatial_hidden)

    def moe_auxiliary_loss(self) -> torch.Tensor:
        """Return the sum of stored MoE load-balancing losses from the latest forward pass."""
        losses: List[torch.Tensor] = []
        for module in self.modules():
            if isinstance(module, MixtureOfExpertsFeedForwardNetwork):
                losses.append(module.auxiliary_loss())
        if not losses:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return torch.stack([loss.to(next(self.parameters()).device) for loss in losses]).sum()
