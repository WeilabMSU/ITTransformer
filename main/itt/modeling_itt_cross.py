import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from .configuration_itt import IttConfig

ACT2FN = {
    "gelu": torch.nn.functional.gelu,
    "relu": torch.nn.functional.relu,
    "swish": lambda x: x * torch.sigmoid(x)
}

def create_nonzero_mask(input_features):
    """
    Create a mask indicating which tokens have non-zero input features.
    
    Args:
        input_features: Tensor of shape [batch_size, num_tokens, feature_dim]
    
    Returns:
        mask: Boolean tensor of shape [batch_size, num_tokens] where True indicates non-zero tokens
    """
    # Check if any feature dimension is non-zero for each token
    # Use a small epsilon to handle numerical precision issues
    epsilon = 1e-8
    return torch.norm(input_features, dim=-1) > epsilon

class IttConvLayer(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len):
        super(IttConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)  # [N, atom_fea_len]
        return out

class IttGraphEncoder(nn.Module):
    def __init__(self, config):
        super(IttGraphEncoder, self).__init__()
        self.config = config
        atom_fea_len = config.d3
        self.atom_fea_len = atom_fea_len
        self.embedding = nn.Linear(config.orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([IttConvLayer(atom_fea_len=atom_fea_len,
                                              nbr_fea_len=config.nbr_fea_len)
                                    for _ in range(config.n_conv)])
        self.max_atom_number = config.max_atom_number

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, cluster_indices):
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        
        # Pad atom_fea per structure to [max_atom_number, hidden_size]
        batch_size = len(crystal_atom_idx)
        embedding_dim = atom_fea.size(-1)
        padded_atom_fea = atom_fea.new_zeros(batch_size, self.max_atom_number, embedding_dim)
        for i, idx_map in enumerate(crystal_atom_idx):
            n_atoms = idx_map.size(0)
            n_pad = min(n_atoms, self.max_atom_number)
            padded_atom_fea[i, :n_pad, :] = atom_fea[idx_map[:n_pad]]
        return padded_atom_fea

class IttEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.project_d0 = nn.Linear(config.d0, config.hidden_size)
        self.project_d1 = nn.Linear(config.d1, config.hidden_size)
        self.project_d2 = nn.Linear(config.d2, config.hidden_size)
        self.project_atom = nn.Linear(config.d3, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings_atom, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = IttLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config
        
    def forward(self, input_part0, input_part1, input_part2, atom_fea, token_type_ids=None, position_ids=None):
        batch_size = input_part0.size(0)
        n0, n1, n2 = self.config.n0, self.config.n1, self.config.n2
        max_atom_number = self.config.max_atom_number
        part0_emb = self.project_d0(input_part0)
        part1_emb = self.project_d1(input_part1)
        part2_emb = self.project_d2(input_part2)
        atom_emb = self.project_atom(atom_fea)
        # Concatenate for position and token type ids
        seq_length = n0 + n1 + max_atom_number + n2
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_part0.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
        if token_type_ids is None:
            part0_type_ids = torch.zeros((batch_size, n0), dtype=torch.long, device=input_part0.device)
            part1_type_ids = torch.ones((batch_size, n1), dtype=torch.long, device=input_part0.device)
            part2_type_ids = torch.full((batch_size, n2), 2, dtype=torch.long, device=input_part0.device)
            atom_type_ids = torch.full((batch_size, max_atom_number), 3, dtype=torch.long, device=input_part0.device)
            token_type_ids = torch.cat([part0_type_ids, part1_type_ids, atom_type_ids, part2_type_ids], dim=1)
        # Concatenate embeddings for position and token type
        all_emb = torch.cat([part0_emb, part1_emb, atom_emb, part2_emb], dim=1)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = all_emb + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        # Split back to parts
        part0_emb = embeddings[:, :n0, :]
        part1_emb = embeddings[:, n0:n0+n1, :]
        atom_emb = embeddings[:, n0+n1:n0+n1+max_atom_number, :]
        part2_emb = embeddings[:, n0+n1+max_atom_number:, :]
        return part0_emb, part1_emb, part2_emb, atom_emb

class IttLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(IttLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class IttSelfAttention(nn.Module):
    def __init__(self, config):
        super(IttSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return (context_layer, attention_probs)

class IttSelfOutput(nn.Module):
    def __init__(self, config):
        super(IttSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = IttLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class IttAttention(nn.Module):
    def __init__(self, config):
        super(IttAttention, self).__init__()
        self.self = IttSelfAttention(config)
        self.output = IttSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        self_outputs = self.self(hidden_states, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class IttIntermediate(nn.Module):
    def __init__(self, config):
        super(IttIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class IttOutput(nn.Module):
    def __init__(self, config):
        super(IttOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = IttLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class IttCrossAttention(nn.Module):
    """
    Cross-attention module for decoder queries against encoder key-value pairs.
    Q comes from decoder sequence (part0 + part1 + per-atom graph output)
    K, V come from encoder sequence (part2)
    """
    def __init__(self, config):
        super(IttCrossAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.config = config

        # Query projection (for decoder sequence)
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        # Key and Value projections (for encoder sequence)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, decoder_hidden_states, encoder_hidden_states, attention_mask=None, head_mask=None, use_cross_attention_mask=True):
        """
        Args:
            decoder_hidden_states: [batch_size, L_out, hidden_size] - decoder sequence (part0+part1+per-atom graph output)
            encoder_hidden_states: [batch_size, L_in, hidden_size] - encoder sequence (part2)
            attention_mask: Optional mask for attention
            head_mask: Optional mask for attention heads
            use_cross_attention_mask: Whether to apply the special cross-attention mask
        """
        # Project decoder hidden states to queries
        mixed_query_layer = self.query(decoder_hidden_states)
        
        # Project encoder hidden states to keys and values
        mixed_key_layer = self.key(encoder_hidden_states)
        mixed_value_layer = self.value(encoder_hidden_states)

        # Transpose for multi-head attention
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Compute attention scores: Q * K^T / sqrt(d_k)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply cross-attention mask if requested and no custom mask provided
        if use_cross_attention_mask and attention_mask is None:
            # Create the special cross-attention mask
            cross_mask = create_cross_attention_mask(
                config=self.config,
                device=attention_scores.device
            )
            # Expand mask to match batch size and number of heads
            batch_size = attention_scores.size(0)
            num_heads = attention_scores.size(1)
            cross_mask = cross_mask.expand(batch_size, num_heads, -1, -1)
            attention_scores = attention_scores + cross_mask
        elif attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Apply softmax to get attention probabilities
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return (context_layer, attention_probs)

class IttCrossAttentionOutput(nn.Module):
    def __init__(self, config):
        super(IttCrossAttentionOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = IttLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class IttDecoderLayer(nn.Module):
    """
    Decoder layer with cross-attention followed by self-attention and feed-forward.
    """
    def __init__(self, config):
        super(IttDecoderLayer, self).__init__()
        # Cross-attention (decoder queries against encoder key-value)
        self.cross_attention = IttCrossAttention(config)
        self.cross_attention_output = IttCrossAttentionOutput(config)
        
        # Self-attention (within decoder sequence)
        self.self_attention = IttAttention(config)
        
        # Feed-forward network
        self.intermediate = IttIntermediate(config)
        self.output = IttOutput(config)

    def forward(self, decoder_hidden_states, encoder_hidden_states, 
                cross_attention_mask=None, self_attention_mask=None, 
                cross_head_mask=None, self_head_mask=None, use_cross_attention_mask=True):
        """
        Args:
            decoder_hidden_states: [batch_size, L_out, hidden_size] - decoder sequence
            encoder_hidden_states: [batch_size, L_in, hidden_size] - encoder sequence
            cross_attention_mask: Optional mask for cross-attention
            self_attention_mask: Optional mask for self-attention
            cross_head_mask: Optional head mask for cross-attention
            self_head_mask: Optional head mask for self-attention
            use_cross_attention_mask: Whether to apply the special cross-attention mask
        """
        # 1. Self-attention: within decoder sequence (standard order)
        self_attention_outputs = self.self_attention(
            decoder_hidden_states, attention_mask=self_attention_mask, head_mask=self_head_mask
        )
        self_attention_output = self_attention_outputs[0]
        
        # 2. Cross-attention: decoder queries against encoder key-value
        cross_attention_outputs = self.cross_attention(
            self_attention_output, encoder_hidden_states, 
            attention_mask=cross_attention_mask, head_mask=cross_head_mask,
            use_cross_attention_mask=use_cross_attention_mask
        )
        cross_attention_output = self.cross_attention_output(
            cross_attention_outputs[0], self_attention_output
        )
        
        # 3. Feed-forward network
        intermediate_output = self.intermediate(cross_attention_output)
        layer_output = self.output(intermediate_output, cross_attention_output)
        
        outputs = (layer_output,) + cross_attention_outputs[1:] + self_attention_outputs[1:]
        return outputs

class IttDecoder(nn.Module):
    """
    Decoder with multiple decoder layers.
    """
    def __init__(self, config):
        super(IttDecoder, self).__init__()
        self.layer = nn.ModuleList([IttDecoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, decoder_hidden_states, encoder_hidden_states, 
                cross_attention_mask=None, self_attention_mask=None, 
                cross_head_mask=None, self_head_mask=None, use_cross_attention_mask=True):
        """
        Args:
            decoder_hidden_states: [batch_size, L_out, hidden_size] - decoder sequence (part0+part1+per-atom graph output)
            encoder_hidden_states: [batch_size, L_in, hidden_size] - encoder sequence (part2)
            cross_attention_mask: Optional mask for cross-attention
            self_attention_mask: Optional mask for self-attention
            cross_head_mask: Optional head mask for cross-attention
            self_head_mask: Optional head mask for self-attention
            use_cross_attention_mask: Whether to apply the special cross-attention mask
        """
        all_hidden_states = ()
        all_cross_attentions = ()
        all_self_attentions = ()
        
        for i, layer_module in enumerate(self.layer):
            all_hidden_states = all_hidden_states + (decoder_hidden_states,)
            
            layer_outputs = layer_module(
                decoder_hidden_states, encoder_hidden_states,
                cross_attention_mask=cross_attention_mask,
                self_attention_mask=self_attention_mask,
                cross_head_mask=cross_head_mask[i] if cross_head_mask is not None else None,
                self_head_mask=self_head_mask[i] if self_head_mask is not None else None,
                use_cross_attention_mask=use_cross_attention_mask
            )
            
            decoder_hidden_states = layer_outputs[0]
            all_cross_attentions = all_cross_attentions + (layer_outputs[1],)
            all_self_attentions = all_self_attentions + (layer_outputs[2],)

        all_hidden_states = all_hidden_states + (decoder_hidden_states,)
        
        return (decoder_hidden_states, all_hidden_states, all_cross_attentions, all_self_attentions)

class IttPooler(nn.Module):
    def __init__(self, config):
        super(IttPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class IttPreTrainedModel(nn.Module):
    config_class = IttConfig
    base_model_prefix = "itt"

    def __init__(self, config, *inputs, **kwargs):
        super(IttPreTrainedModel, self).__init__()
        if not isinstance(config, IttConfig):
            raise ValueError("config should be an instance of IttConfig")
        self.config = config

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, IttLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

def create_cross_attention_mask(config, device):
    """
    Create cross-attention mask for decoder (part0+part1+per-atom graph output) attending to encoder (part2).
    
    Args:
        config: Model configuration with n0, n1, n2, n3 dimensions
        device: Device to create the mask on
    
    Returns:
        attention_mask: [1, 1, L_out, L_in] where L_out=1+7+max_atom_number and L_in=42
    """
    n0, n1, n2 = config.n0, config.n1, config.n2
    max_atom_number = config.max_atom_number
    L_out = n0 + n1 + max_atom_number  # 1 + 7 + max_atom_number
    L_in = n2  # 42
    
    # Initialize mask with zeros (no masking by default)
    mask = torch.zeros(1, 1, L_out, L_in, device=device)
    
    # For now, allow all cross-attention (no masking)
    # This can be customized based on specific requirements
    return mask

# ============================================================================
# Main Cross-Attention Modules
# ============================================================================

class IttCrossModel(IttPreTrainedModel):
    """
    Decoder-only ITT model using cross-attention between decoder (part0+part1+per-atom graph output) and part2 sequences.
    The per-atom graph output is padded to max_atom_number.
    """
    def __init__(self, config):
        super(IttCrossModel, self).__init__(config)
        self.config = config
        self.graph_cluster_encoder = IttGraphEncoder(config)
        self.embeddings = IttEmbeddings(config)
        self.decoder = IttDecoder(config)
        self.pooler = IttPooler(config)
        self.apply(self._init_weights)

    def forward(self, input_part0, input_part1, input_part2, graph_data, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, use_cross_attention_mask=True):
        input_part0 = input_part0.to(dtype=torch.float32)
        input_part1 = input_part1.to(dtype=torch.float32)
        input_part2 = input_part2.to(dtype=torch.float32)
        atom_fea, nbr_fea, nbr_fea_idx, cluster_indices, crystal_atom_idx = graph_data
        padded_atom_fea = self.graph_cluster_encoder(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, cluster_indices)  # [batch, max_atom_number, d3]

        # Use IttEmbeddings to embed all parts
        part0_emb, part1_emb, part2_emb, graph_emb = self.embeddings(
            input_part0, input_part1, input_part2, padded_atom_fea, token_type_ids, position_ids)
        # Concatenate decoder input: part0 + part1 + per-atom graph output
        decoder_hidden_states = torch.cat([part0_emb, part1_emb, graph_emb], dim=1)  # [batch, n0+n1+max_atom_number, hidden]
        # part2 is used as encoder sequence for cross-attention
        encoder_hidden_states = part2_emb  # [batch, n2, hidden]
        # Handle attention masks
        if attention_mask is not None:
            decoder_mask = torch.cat([
                attention_mask[:, :self.config.n0],
                attention_mask[:, self.config.n0:self.config.n0+self.config.n1],
                torch.ones((attention_mask.size(0), self.config.max_atom_number), device=attention_mask.device, dtype=attention_mask.dtype)
            ], dim=1)
            part2_mask = attention_mask[:, self.config.n0+self.config.n1:self.config.n0+self.config.n1+self.config.n2]
            decoder_extended_mask = decoder_mask.unsqueeze(1).unsqueeze(2)
            decoder_extended_mask = decoder_extended_mask.to(dtype=next(self.parameters()).dtype)
            decoder_extended_mask = (1.0 - decoder_extended_mask) * -10000.0
            part2_extended_mask = part2_mask.unsqueeze(1).unsqueeze(2)
            part2_extended_mask = part2_extended_mask.to(dtype=next(self.parameters()).dtype)
            part2_extended_mask = (1.0 - part2_extended_mask) * -10000.0
        else:
            decoder_extended_mask = None
            part2_extended_mask = None
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers
        decoder_outputs = self.decoder(
            decoder_hidden_states=decoder_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_mask=None,
            self_attention_mask=decoder_extended_mask,
            cross_head_mask=head_mask,
            self_head_mask=head_mask,
            use_cross_attention_mask=use_cross_attention_mask
        )
        decoder_sequence_output = decoder_outputs[0]
        pooled_output = self.pooler(decoder_sequence_output)
        outputs = (decoder_sequence_output, pooled_output,) + decoder_outputs[1:]
        return outputs

class IttCrossSequenceClassifier(IttPreTrainedModel):
    """
    Decoder-based ITT model for sequence classification using cross-attention.
    Uses part0 + part1 + per-atom graph output (padded) as decoder input, part2 as encoder.
    Pooling is controlled by classifier_head_method.
    """
    def __init__(self, config):
        super(IttCrossSequenceClassifier, self).__init__(config)
        self.num_labels = config.num_labels
        self.itt_cross = IttCrossModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_head_method = config.classifier_head_method
        self.hidden_size = config.hidden_size
        if self.classifier_head_method == 'split_mean':
            head_in_dim = self.hidden_size * 3
        else:
            head_in_dim = self.hidden_size
        self.classifier = nn.Linear(head_in_dim, self.num_labels)
        self.apply(self._init_weights)

    def forward(self, input_part0, input_part1, input_part2, graph_data,
                attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None, use_cross_attention_mask=False):
        outputs = self.itt_cross(
            input_part0=input_part0,
            input_part1=input_part1,
            input_part2=input_part2,
            graph_data=graph_data,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cross_attention_mask=use_cross_attention_mask
        )
        sequence_output = outputs[0]
        n0, n1, n2, n3 = self.config.n0, self.config.n1, self.config.n2, self.config.n3
        if self.classifier_head_method == 'cls':
            pooled = self.dropout(sequence_output[:, 0, :])
        elif self.classifier_head_method == 'mean':
            pooled = self.dropout(torch.mean(sequence_output, dim=1))
        elif self.classifier_head_method == 'split_mean':
            part0_emb = sequence_output[:, :n0, :]
            part1_emb = sequence_output[:, n0:n0+n1, :]
            graph_emb = sequence_output[:, n0+n1:, :]  # [batch, max_atom_number, hidden]
            part0_pooled = torch.mean(part0_emb, dim=1)
            part1_pooled = torch.mean(part1_emb, dim=1)
            graph_pooled = torch.mean(graph_emb, dim=1)
            pooled = torch.cat((part0_pooled, part1_pooled, graph_pooled), dim=1)
            pooled = self.dropout(pooled)
        else:
            raise ValueError(f"Unknown classifier_head_method: {self.classifier_head_method}")
        logits = self.classifier(pooled)
        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                if labels.dim() == 2 and labels.size(1) == 1:
                    labels = labels.squeeze(-1)
                loss = loss_fct(logits, labels)
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs

class IttForCrossJointPredictionLM(IttPreTrainedModel):
    """
    Pretraining head for Cross model: joint prediction of volume (regression) and per-atom binary cluster classification.
    - Task 1: Regression (volume) using the first token (part0) embedding.
    - Task 2: Binary classification (per-atom cluster group) using the graph output (padded per-atom features).
              Atoms are classified as either belonging to clusters 4 and 6 (class 1) or other clusters (class 0).
    - Loss is the mean of the two task losses.
    Args:
        config: IttConfig
    """
    def __init__(self, config):
        super().__init__(config)
        self.itt_cross = IttCrossModel(config)
        self.hidden_size = config.hidden_size
        self.volume_head = nn.Linear(self.hidden_size, 1)
        self.cluster_head = nn.Linear(self.hidden_size, 2)  # Binary classification: 2 classes
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.apply(self._init_weights)

    def forward(self, input_part0, input_part1, input_part2, graph_data,
                attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                volume_labels=None, cluster_labels=None, use_cross_attention_mask=False):
        """
        Args:
            input_part0, input_part1, input_part2: input features
            graph_data: tuple of graph features
            volume_labels: [batch] regression targets
            cluster_labels: [batch, max_atom_number] int64, per-atom binary labels (0 or 1, -1 for padding)
                           where 1 indicates clusters 4 and 6, 0 indicates other clusters
        Returns:
            dict with keys: loss, volume_pred, cluster_logits, loss_volume, loss_cluster
        """
        outputs = self.itt_cross(
            input_part0=input_part0,
            input_part1=input_part1,
            input_part2=input_part2,
            graph_data=graph_data,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cross_attention_mask=use_cross_attention_mask
        )
        sequence_output = outputs[0]  # [batch, seq_len, hidden]
        # Get the first token (part0) embedding for volume prediction
        part0_emb = sequence_output[:, 0, :]  # [batch, hidden]
        part0_emb = self.dropout(part0_emb)
        volume_pred = self.volume_head(part0_emb).squeeze(-1)  # [batch]
        # Use the atom features from the last layer's outputs (decoder output), corresponding to the per-atom graph tokens
        n0, n1 = self.config.n0, self.config.n1
        atom_emb = sequence_output[:, n0+n1::, :]  # [batch, max_atom_number, hidden]
        cluster_logits = self.cluster_head(atom_emb)  # [batch, max_atom_number, 2]
        loss = None
        loss_volume = None
        loss_cluster = None
        if (volume_labels is not None) and (cluster_labels is not None):
            # Volume regression loss
            loss_fct_reg = nn.MSELoss()
            loss_volume = loss_fct_reg(volume_pred, volume_labels.float())
            # Cluster binary classification loss (ignore padding atoms with label -1)
            # cluster_labels: [batch, max_atom_number], cluster_logits: [batch, max_atom_number, 2]
            active_mask = (cluster_labels != -1)
            if active_mask.sum() > 0:
                cluster_logits_flat = cluster_logits[active_mask]  # [num_active, 2]
                cluster_labels_flat = cluster_labels[active_mask]  # [num_active]
                loss_fct_cls = nn.CrossEntropyLoss()
                loss_cluster = loss_fct_cls(cluster_logits_flat, cluster_labels_flat)
            else:
                loss_cluster = torch.tensor(0.0, device=volume_pred.device)
            loss = (loss_volume + loss_cluster) / 2.0
        return {
            'loss': loss,
            'volume_pred': volume_pred,
            'cluster_logits': cluster_logits,
            'loss_volume': loss_volume,
            'loss_cluster': loss_cluster,
            'sequence_output': sequence_output,
        }

class IttForCrossMultitaskPredictionLM(IttPreTrainedModel):
    """
    Pretraining head for Cross model: joint prediction of volume (regression) and per-atom binary cluster classification.
    - Task 1: Regression (volume) using the first token (part0) embedding.
    - Task 2: Regression (average electronegativity) using the first token (part0) embedding.
    - Loss is the mean of the two task losses.
    Args:
        config: IttConfig
    """
    def __init__(self, config):
        super().__init__(config)
        self.itt_cross = IttCrossModel(config)
        self.hidden_size = config.hidden_size
        self.volume_head = nn.Linear(self.hidden_size, 1)
        self.electronegativity_head = nn.Linear(self.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.apply(self._init_weights)

    def forward(self, input_part0, input_part1, input_part2, graph_data,
                attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                volume_labels=None, electronegativity_labels=None, use_cross_attention_mask=False):
        """
        Args:
            input_part0, input_part1, input_part2: input features
            graph_data: tuple of graph features
            volume_labels: [batch] regression targets
            cluster_labels: [batch, max_atom_number] int64, per-atom binary labels (0 or 1, -1 for padding)
                           where 1 indicates clusters 4 and 6, 0 indicates other clusters
        Returns:
            dict with keys: loss, volume_pred, cluster_logits, loss_volume, loss_cluster
        """
        outputs = self.itt_cross(
            input_part0=input_part0,
            input_part1=input_part1,
            input_part2=input_part2,
            graph_data=graph_data,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cross_attention_mask=use_cross_attention_mask
        )
        sequence_output = outputs[0]  # [batch, seq_len, hidden]
        # Get the first token (part0) embedding for volume prediction
        part0_emb = sequence_output[:, 0, :]  # [batch, hidden]
        part0_emb = self.dropout(part0_emb)
        volume_pred = self.volume_head(part0_emb).squeeze(-1)  # [batch]
        electronegativity_pred = self.electronegativity_head(part0_emb).squeeze(-1)  # [batch]
        loss = None
        loss_volume = None
        loss_electronegativity = None
        if (volume_labels is not None) and (electronegativity_labels is not None):
            # Volume regression loss
            loss_fct_reg = nn.MSELoss()
            loss_volume = loss_fct_reg(volume_pred, volume_labels.float())
            # Electronegativity regression loss
            loss_fct_reg = nn.MSELoss()
            loss_electronegativity = loss_fct_reg(electronegativity_pred, electronegativity_labels.float())
            loss = (loss_volume + loss_electronegativity) / 2.0
        return {
            'loss': loss,
            'volume_pred': volume_pred,
            'electronegativity_pred': electronegativity_pred,
            'loss_volume': loss_volume,
            'loss_electronegativity': loss_electronegativity,
            'sequence_output': sequence_output,
        }

class IttForCrossMaskedAndAtomLM(IttPreTrainedModel):
    """
    Pretraining head for Cross model: masked language modeling for part0 and per-atom cluster classification.
    - Task 1: Masked language modeling for part0 (first token) with configurable mask ratio.
    - Task 2: Multiclass classification (per-atom cluster index) using the atom features from last layer outputs.
    - Loss is the mean of the two task losses.
    Args:
        config: IttConfig
        mask_ratio: float, ratio of tokens to mask in part0 (default: 0.15)
    """
    def __init__(self, config, mask_ratio=0.15):
        super().__init__(config)
        self.itt_cross = IttCrossModel(config)
        self.hidden_size = config.hidden_size
        self.mask_ratio = mask_ratio
        self.cluster_head = nn.Linear(self.hidden_size, config.n3)  # n3: number of clusters/classes
        self.masked_head = nn.Linear(self.hidden_size, config.d0)  # Map back to original part0 dimension
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.apply(self._init_weights)

    def forward(self, input_part0, input_part1, input_part2, graph_data,
                attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                mask_info=None, cluster_labels=None, use_cross_attention_mask=False):
        """
        Args:
            input_part0, input_part1, input_part2: input features
            graph_data: tuple of graph features
            mask_info: dict with 'part0' key containing (mask, target) for masked language modeling
            cluster_labels: [batch, max_atom_number] int64, per-atom cluster indices (0..n3-1, -1 for padding)
        Returns:
            dict with keys: loss, masked_loss, cluster_loss, masked_logits, cluster_logits
        """
        outputs = self.itt_cross(
            input_part0=input_part0,
            input_part1=input_part1,
            input_part2=input_part2,
            graph_data=graph_data,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cross_attention_mask=use_cross_attention_mask
        )
        sequence_output = outputs[0]  # [batch, seq_len, hidden]
        
        # Task 1: Masked language modeling for part0 (first token)
        part0_emb = sequence_output[:, 0, :]  # [batch, hidden]
        masked_logits = self.masked_head(part0_emb)  # [batch, d0] - map back to original part0 dimension
        
        # Task 2: Per-atom cluster classification using atom features from last layer outputs
        n0, n1 = self.config.n0, self.config.n1
        atom_emb = sequence_output[:, n0+n1:, :]  # [batch, max_atom_number, hidden]
        cluster_logits = self.cluster_head(atom_emb)  # [batch, max_atom_number, n3]
        
        loss = None
        masked_loss = None
        cluster_loss = None
        
        if mask_info is not None and 'part0' in mask_info:
            # Masked language modeling loss for part0
            mask, target = mask_info['part0']
            # mask: [batch, d0], target: [batch, d0], masked_logits: [batch, d0]
            if mask.sum() > 0:
                # Flatten tensors for boolean indexing
                mask_flat = mask.view(-1)  # [batch * d0]
                masked_logits_flat = masked_logits.view(-1)  # [batch * d0]
                target_flat = target.view(-1)  # [batch * d0]
                masked_pred = masked_logits_flat[mask_flat]  # [num_masked]
                masked_target = target_flat[mask_flat]  # [num_masked]
                loss_fct_masked = nn.MSELoss()
                masked_loss = loss_fct_masked(masked_pred, masked_target)
            else:
                masked_loss = torch.tensor(0.0, device=masked_logits.device)
        
        if cluster_labels is not None:
            # Cluster classification loss (ignore padding atoms with label -1)
            active_mask = (cluster_labels != -1)
            if active_mask.sum() > 0:
                cluster_logits_flat = cluster_logits[active_mask]  # [num_active, n3]
                cluster_labels_flat = cluster_labels[active_mask]  # [num_active]
                loss_fct_cls = nn.CrossEntropyLoss()
                cluster_loss = loss_fct_cls(cluster_logits_flat, cluster_labels_flat)
            else:
                cluster_loss = torch.tensor(0.0, device=masked_logits.device)
        
        # Compute total loss
        if masked_loss is not None and cluster_loss is not None:
            loss = (masked_loss + cluster_loss) / 2.0
        elif masked_loss is not None:
            loss = masked_loss
        elif cluster_loss is not None:
            loss = cluster_loss
        
        return {
            'loss': loss,
            'masked_loss': masked_loss,
            'cluster_loss': cluster_loss,
            'masked_logits': masked_logits,
            'cluster_logits': cluster_logits,
            'sequence_output': sequence_output,
        }

class IttForCrossMaskedLM(IttPreTrainedModel):
    """
    Pretraining head for Cross model: masked language modeling for part0 only.
    - Task: Masked language modeling for part0 (first token) with configurable mask ratio.
    - Loss is the masked language modeling loss.
    Args:
        config: IttConfig
        mask_ratio: float, ratio of features to mask in part0 (default: 0.15)
    """
    def __init__(self, config, mask_ratio=0.15):
        super().__init__(config)
        self.itt_cross = IttCrossModel(config)
        self.mask_ratio = mask_ratio
        self.masked_head = nn.Linear(config.hidden_size, config.d0)  # Map back to original part0 dimension
        self.apply(self._init_weights)

    def forward(self, input_part0, input_part1, input_part2, graph_data,
                attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                mask_info=None, use_cross_attention_mask=False):
        """
        Args:
            input_part0, input_part1, input_part2: input features
            graph_data: tuple of graph features
            mask_info: dict with 'part0' key containing (mask, target) for masked language modeling
        Returns:
            dict with keys: loss, masked_loss, masked_logits
        """
        outputs = self.itt_cross(
            input_part0=input_part0,
            input_part1=input_part1,
            input_part2=input_part2,
            graph_data=graph_data,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cross_attention_mask=use_cross_attention_mask
        )
        sequence_output = outputs[0]  # [batch, seq_len, hidden]
        
        # Masked language modeling for part0 (first token)
        part0_emb = sequence_output[:, 0, :]  # [batch, hidden]
        masked_logits = self.masked_head(part0_emb)  # [batch, d0] - map back to original part0 dimension
        
        loss = None
        masked_loss = None
        
        if mask_info is not None and 'part0' in mask_info:
            # Masked language modeling loss for part0
            mask, target = mask_info['part0']
            # mask: [batch, d0], target: [batch, d0], masked_logits: [batch, d0]
            if mask.sum() > 0:
                # Flatten tensors for boolean indexing
                mask_flat = mask.view(-1)  # [batch * d0]
                masked_logits_flat = masked_logits.view(-1)  # [batch * d0]
                target_flat = target.view(-1)  # [batch * d0]
                masked_pred = masked_logits_flat[mask_flat]  # [num_masked]
                masked_target = target_flat[mask_flat]  # [num_masked]
                loss_fct_masked = nn.MSELoss()
                masked_loss = loss_fct_masked(masked_pred, masked_target)
            else:
                masked_loss = torch.tensor(0.0, device=masked_logits.device)
        
        # Total loss is just the masked loss
        loss = masked_loss
        
        return {
            'loss': loss,
            'masked_loss': masked_loss,
            'masked_logits': masked_logits,
            'sequence_output': sequence_output,
        } 