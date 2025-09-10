
class IttConfig:
    def __init__(self,
                 hidden_size=256,
                 num_hidden_layers=8,
                 num_attention_heads=8,
                 intermediate_size=1024,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 type_vocab_size=4, # 0: part0, 1: part1, 2: part2, 3: atom
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 pad_token_id=0,
                 gradient_checkpointing=False,
                 d0=750,  # dimension of part0 features
                 d1=750,  # dimension of part1 features
                 d2=500,  # dimension of part2 features
                 d3=128,   # dimension of cluster-level embeddings from graph module, original 64
                 n0=1,    # number of tokens in part0
                 n1=7,  # number of tokens in part1
                 n2=42,  # number of tokens in part2
                 n3=7,    # number of clusters
                 orig_atom_fea_len=92, # input dimension for atom features in graph module
                 nbr_fea_len=41, # input dimension for neighbor features in graph module
                 n_conv=4,
                 classifier_head_method="cls",
                 num_labels=1,
                 max_atom_number=256,
                 **kwargs):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.gradient_checkpointing = gradient_checkpointing
        self.d0 = d0
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.n0 = n0
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.orig_atom_fea_len = orig_atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.n_conv = n_conv
        self.classifier_head_method = classifier_head_method
        self.num_labels = num_labels
        self.max_atom_number = max_atom_number
        self.max_position_embeddings = n0 + n1 + n2 + n3
        self.max_position_embeddings_atom = n0 + n1 + n2 + max_atom_number
        for key, value in kwargs.items():
            setattr(self, key, value) 

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = self.__dict__.copy()
        return output 