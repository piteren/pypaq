from typing import Optional
import torch

from pypaq.torchness.types import ACT, INI, TNS, DTNS
from pypaq.torchness.base_elements import bert_initializer, my_initializer, TorchnessException
from pypaq.torchness.layers import LayDense, TF_Dropout, LayConv1D, zeroes


# Block (Layer) of EncDRT (LN > Dense > drop > RES)
class LayBlockDRT(torch.nn.Module):

    def __init__(
            self,
            in_width: int,
            do_scaled_dns: bool=    False,          # two denses (True) or single dense (False)
            dns_scale: int=         4,              # up-scale for first dense of two
            activation: ACT=        torch.nn.ReLU,
            lay_dropout: float=     0.0,            # dropout after dense/s
            residual: bool=         True,           # residual yes/no
            res_dropout: float=     0.0,            # dropout on residual connection
            device=                 None,
            dtype=                  None,
            initializer: INI=       None):

        super(LayBlockDRT, self).__init__()

        if initializer is None: initializer = my_initializer

        self.ln_in = torch.nn.LayerNorm(
            normalized_shape=   in_width,
            device=             device,
            dtype=              dtype)

        self.denses = []
        if do_scaled_dns:
            # dense (scale up) with activation
            self.denses.append(LayDense(
                in_features=    in_width,
                out_features=   in_width * dns_scale,
                activation=     activation,
                bias=           True,
                device=         device,
                dtype=          dtype,
                initializer=    initializer))
            # dense (scale down) without activation
            self.denses.append(LayDense(
                in_features=    in_width * dns_scale,
                out_features=   in_width,
                activation=     None,
                bias=           True,
                device=         device,
                dtype=          dtype,
                initializer=    initializer))
        else:
            # just single dense, with activation
            self.denses.append(LayDense(
                in_features=    in_width,
                out_features=   in_width,
                activation=     activation,
                bias=           True,
                device=         device,
                dtype=          dtype,
                initializer=    initializer))
        for dix, l in enumerate(self.denses): self.add_module(f'dense{dix}', l)

        self.drop_lay = torch.nn.Dropout(p=lay_dropout) if lay_dropout else None

        self.add_res = residual

        self.drop_res = torch.nn.Dropout(p=res_dropout) if res_dropout else None

    def forward(self, input:TNS) -> DTNS:

        out = self.ln_in(input)

        dense = self.denses[0]
        out = dense(out)
        zsL = [zeroes(out)]

        if len(self.denses) > 1: # there is second one, without activation
            dense = self.denses[1]
            out = dense(out)

        if self.drop_lay:
            out = self.drop_lay(out)

        if self.add_res:
            if self.drop_res: x = self.drop_res(input)
            out += input # residual

        return {
            'out':  out,
            'zsL':  zsL}


# Deep Residual encoder based on stacked LayBlockDRT
class EncDRT(torch.nn.Module):

    def __init__(
            self,
            in_width: int,
            in_dropout: float=          0.0,            # dropout on input
            shared_lays: bool=          False,          # shared variables in enc_layers
            n_layers: int=              6,
            lay_width: Optional[int]=   None,           # for None matches input width
            do_scaled_dns: bool=        True,
            dns_scale: int=             4,              # scale(*) of first dense
            activation: ACT=            torch.nn.ReLU,  # gelu is really worth a try
            lay_dropout: float=         0.0,            # dropout after two denses
            residual: bool=             True,           # residual yes/no
            res_dropout: float=         0.0,            # dropout on residual connection
            device=                     None,
            dtype=                      None,
            initializer: INI=           None):

        super(EncDRT, self).__init__()

        if initializer is None: initializer = my_initializer

        self.in_drop_lay = torch.nn.Dropout(p=in_dropout) if in_dropout else None

        self.in_width = in_width
        self.lay_width = lay_width or self.in_width
        self.projection_lay = LayDense(
            in_features=    self.in_width,
            out_features=   self.lay_width,
            activation=     None,
            bias=           False,
            device=         device,
            dtype=          dtype,
            initializer=    initializer) if self.lay_width != self.in_width else None

        self.ln_in = torch.nn.LayerNorm(
            normalized_shape=   self.lay_width,
            device=             device,
            dtype=              dtype)

        num_layers_to_build = 1 if shared_lays else n_layers
        self.drt_lays = [LayBlockDRT(
            in_width=       self.lay_width,
            do_scaled_dns=  do_scaled_dns,
            dns_scale=      dns_scale,
            activation=     activation,
            lay_dropout=    lay_dropout,
            residual=       residual,
            res_dropout=    res_dropout,
            device=         device,
            dtype=          dtype,
            initializer=    initializer
        ) for _ in range(num_layers_to_build)]
        for lix,lay in enumerate(self.drt_lays): self.add_module(f'lay_drt_{lix}',lay)
        if shared_lays and n_layers > 1: self.drt_lays *= n_layers

    def forward(self, input:TNS) -> DTNS:

        zsL = []

        out = input

        if self.in_drop_lay: # input dropout
            out = self.in_drop_lay(out)

        if self.projection_lay: # input projection, no activation <- do not catch zeroes
            out = self.projection_lay(out)

        out = self.ln_in(out)

        for drt_lay in self.drt_lays:
            lay_out = drt_lay(out)
            out = lay_out['out']
            zsL += lay_out['zsL']

        return {
            'out':  out,
            'zsL':  zsL}


# Block (Layer) of EncCNN (LN > CNN > act > drop > RES > LayBlockDRT), number of parameters: kernel*in_features*n_filters
class LayBlockCNN(torch.nn.Module):

    def __init__(
            self,
            n_filters: int,                             # num of filters
            kernel_size :int=           3,              # layer kernel
            activation: ACT=            torch.nn.ReLU,  # global enc activation func
            lay_dropout: float=         0.0,
            res_dropout: float=         0.0,
            # lay_DRT
            do_ldrt=                    False,          # lay DRT - build or not
            ldrt_do_scaled_dns: bool=   True,
            ldrt_dns_scale: int=        4,
            ldrt_drop: float or None=   0.0,
            ldrt_residual: bool=        True,
            ldrt_res_dropout: float=    0.0,
            # other
            device=                     None,
            dtype=                      None,
            initializer: INI=           None):

        super(LayBlockCNN, self).__init__()

        if kernel_size % 2 != 1: raise TorchnessException('LayBlockCNN kernel_size cannot be even number')
        self.kernel_size = kernel_size

        self.lay_ln = torch.nn.LayerNorm(
            normalized_shape=   n_filters,
            device=             device,
            dtype=              dtype)

        self.lay_conv1D = LayConv1D(
            in_features=    n_filters,
            n_filters=      n_filters,
            kernel_size=    self.kernel_size,
            padding=        'valid',
            device=         device,
            dtype=          dtype,
            activation=     None,
            initializer=    initializer)

        self.activation = activation() if activation else None

        self.lay_drop = torch.nn.Dropout(p=lay_dropout) if lay_dropout else None

        self.res_drop = torch.nn.Dropout(p=res_dropout) if res_dropout else None

        self.lay_DRT = LayBlockDRT(
            in_width=       n_filters,
            do_scaled_dns=  ldrt_do_scaled_dns,
            dns_scale=      ldrt_dns_scale,
            activation=     activation,
            lay_dropout=    ldrt_drop,
            residual=       ldrt_residual,
            res_dropout=    ldrt_res_dropout,
            device=         device,
            dtype=          dtype,
            initializer=    initializer) if do_ldrt else None

    def forward(self, input:TNS, history:Optional[TNS]=None) -> DTNS:

        zsL = []

        out = self.lay_ln(input)

        if history is None:
            in_sh = list(input.shape)
            pad_width = int((self.kernel_size-1)/2)
            in_sh[-2] = pad_width
            pad = torch.zeros(in_sh)
            conc = [pad, out, pad] # pad both sides (encoder)
        else:
            conc = [history, out] # concatenate with history (casual encoder)
        out = torch.concat(conc, dim=-2)

        out = self.lay_conv1D(out)

        if self.activation:
            out = self.activation(out)
            zsL.append(zeroes(out))

        if self.lay_drop:
            out = self.lay_drop(out)

        if self.res_drop:
            input = self.res_drop(input)

        out += input  # RES

        if self.lay_DRT:
            lay_out = self.lay_DRT(out)
            out = lay_out['out']
            zsL += lay_out['zsL']

        return {
            'out':      out,
            'state':    torch.split(input, split_size_or_sections=self.kernel_size-1, dim=-2)[-1],
            'zsL':      zsL}


# CNN 1D Encoder (for sequences), number of parameters: projection + n_layers*LayBlockCNN
class EncCNN(torch.nn.Module):

    def __init__(
            self,
            in_features: int,                           # input num of channels
            time_drop: float=           0.0,
            feat_drop: float=           0.0,
            # layer
            shared_lays: bool=          False,          # shared variables in enc_layers
            n_layers :int=              6,              # num of layers
            kernel_size :int=           3,              # layer kernel
            n_filters :Optional[int]=   None,           # num of filters
            activation: ACT=            torch.nn.ReLU,  # global enc activation func
            lay_dropout: float=         0.0,
            res_dropout: float=         0.0,
            # lay_DRT
            do_ldrt=                    False,          # lay DRT - build or not
            ldrt_do_scaled_dns: bool=   True,
            ldrt_dns_scale: int=        4,
            ldrt_drop: float or None=   0.0,
            ldrt_residual: bool=        True,
            ldrt_res_dropout: float=    0.0,
            # other
            device=                     None,
            dtype=                      None,
            initializer: INI=           None):

        super(EncCNN, self).__init__()

        self.in_features = in_features
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.n_filters = n_filters or self.in_features

        self.in_TFdrop_lay = TF_Dropout(
            time_drop=  time_drop,
            feat_drop=  feat_drop) if time_drop or feat_drop else None

        self.projection_lay = LayDense(
            in_features=    self.in_features,
            out_features=   self.n_filters,
            activation=     None,
            bias=           False,
            device=         device,
            dtype=          dtype,
            initializer=    initializer) if self.in_features != self.n_filters else None

        num_blocks_to_build = 1 if shared_lays else self.n_layers

        self.blocks = [LayBlockCNN(
            n_filters=          self.n_filters,
            kernel_size=        self.kernel_size,
            activation=         activation,
            lay_dropout=        lay_dropout,
            res_dropout=        res_dropout,
            do_ldrt=            do_ldrt,
            ldrt_do_scaled_dns= ldrt_do_scaled_dns,
            ldrt_dns_scale=     ldrt_dns_scale,
            ldrt_drop=          ldrt_drop,
            ldrt_residual=      ldrt_residual,
            ldrt_res_dropout=   ldrt_res_dropout,
            device=             device,
            dtype=              dtype,
            initializer=        initializer) for _ in range(num_blocks_to_build)]

        for bix,block in enumerate(self.blocks): self.add_module(f'block_{bix}',block)

        if shared_lays and self.n_layers > 1: self.blocks *= self.n_layers

        self.out_ln = torch.nn.LayerNorm(
            normalized_shape=   self.n_filters,
            device=             device,
            dtype=              dtype)

    # prepares initial history for casual mode, history has shape [.., n_layers, kernel_size-1, n_filters]
    def get_zero_history(self, input:TNS) -> TNS:
        in_sh = list(input.shape)
        in_sh.insert(-2, self.n_layers)
        in_sh[-2] = self.kernel_size - 1
        in_sh[-1] = self.n_filters
        return torch.zeros(in_sh)

    def forward(self, input:TNS, history:Optional[TNS]=None) -> DTNS:

        states = []  # here we will store block states to concatenate them finally
        zsL = []

        if self.in_TFdrop_lay:
            input = self.in_TFdrop_lay(input)

        if self.projection_lay:
            input = self.projection_lay(input)

        output = input  # for 0 layers case
        histories = torch.split(history,1,dim=-3) if history is not None else [None]*self.n_layers
        for block,hist in zip(self.blocks, histories):
            if hist is not None: hist = torch.squeeze(hist, dim=-3)
            block_out = block(output, history=hist)
            output = block_out['out']
            states.append(torch.unsqueeze(block_out['state'], dim=-3))
            zsL += block_out['zsL']

        output = self.out_ln(output)

        return {
            'out':      output,
            'state':    torch.concat(states,dim=-3),
            'zsL':      zsL}


# QKV_linear_projection + QKV_scaled_dot_product_attention + linear_out_projection
class MyMHA(torch.nn.MultiheadAttention):

    # replaces xavier with BERT
    def _reset_parameters(self):

        if self._qkv_same_embed_dim:
            bert_initializer(self.in_proj_weight)
        else:
            bert_initializer(self.q_proj_weight)
            bert_initializer(self.k_proj_weight)
            bert_initializer(self.v_proj_weight)
        bert_initializer(self.out_proj.weight)

        if self.in_proj_bias is not None:
            torch.nn.init.zeros_(self.in_proj_bias)
            torch.nn.init.zeros_(self.out_proj.bias)
        if self.bias_k is not None:
            bert_initializer(self.bias_k)
        if self.bias_v is not None:
            bert_initializer(self.bias_v)


# Block (Layer) of EncTNS (based on torch.nn.modules.transformer.TransformerEncoderLayer; )
class LayBlockTNS(torch.nn.Module):

    __constants__ = ['batch_first', 'norm_first']

    def __init__(
            self,
            d_model: int=               512,
            nhead: int=                 8,
            dns_scale: int=             4,              # up-scale for first dense of two
            dropout: float=             0.1,
            dropout_att: float=         0.0,            # in original (torch.nn..) implementation dropout_att == dropout
            activation: ACT=            torch.nn.ReLU,
            #layer_norm_eps: float = 1e-5,              # TODO: remove
            #batch_first: bool = False,                 # TODO: remove
            #norm_first: bool = False,                  # TODO: remove
            device=                     None,
            dtype=                      None,
            initializer: INI=           None            # TODO: check orig initializer
    ):

        #factory_kwargs = {'device': device, 'dtype': dtype} # TODO: remove

        super(LayBlockTNS, self).__init__()

        self.norm1 = torch.nn.LayerNorm(
            normalized_shape=   d_model,
            device=             device,
            dtype=              dtype)

        self.self_attn = MyMHA(
            embed_dim=      d_model,
            num_heads=      nhead,
            dropout=        dropout_att,
            bias=           True,
            add_bias_kv=    False,
            add_zero_attn=  False,
            kdim=           None,
            vdim=           None,
            batch_first=    True,
            device=         device,
            dtype=          dtype)

        self.dropout1 = torch.nn.Dropout(p=dropout) if dropout else None

        self.lay_drt = LayBlockDRT(
            in_width=       d_model,
            do_scaled_dns=  True,
            dns_scale=      dns_scale,
            activation=     activation,
            lay_dropout=    dropout,
            residual=       True,
            res_dropout=    0.0, # TODO: use
            device=         device,
            dtype=          dtype,
            initializer=    bert_initializer)

        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu:
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu:
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation # TODO: call () __init__

    def __setstate__(self, state):
        super(LayBlockTNS, self).__setstate__(state)
        # TODO: why such activation policy
        if not hasattr(self, 'activation'):
            self.activation = F.relu

    def forward(
            self,
            src: torch.Tensor,
            src_mask: Optional[torch.Tensor]=               None,
            src_key_padding_mask: Optional[torch.Tensor]=   None) -> torch.Tensor:

        x = self.norm1(src) # norm first https://arxiv.org/pdf/2002.04745v1.pdf

        x = self.self_attn(
            query=              x,
            key=                x,
            value=              x,
            key_padding_mask=   src_key_padding_mask,
            need_weights=       False,
            attn_mask=          src_mask)[0]

        if self.dropout1:
            x = self.dropout1(x)

        sa_block_out = x + src # first res #TODO: add residual dropout

        x = sa_block_out + self._ff_block(self.norm2(sa_block_out))

        return x

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


# Transformer Encoder (based on torch.nn.modules.transformer.TransformerEncoder)
class EncTNS(torch.nn.Module):

    __constants__ = ['norm']

    def __init__(
            self,
            num_layers,
            # add task attention (TA) mode
            max_seq_len: Optional[int]=     None,   # when int given adds positional embeddings (PE) to seq
            # norm=None, #TODO <- disable, hardcode
            enable_nested_tensor=False):
        super(EncTNS, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.enable_nested_tensor = enable_nested_tensor

    def forward(self, src: torch.Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        if isinstance(first_layer, torch.nn.TransformerEncoderLayer):
            if (not first_layer.norm_first and not first_layer.training and
                    first_layer.self_attn.batch_first and
                    first_layer.self_attn._qkv_same_embed_dim and first_layer.activation_relu_or_gelu and
                    first_layer.norm1.eps == first_layer.norm2.eps and
                    src.dim() == 3 and self.enable_nested_tensor) :
                if src_key_padding_mask is not None and not output.is_nested and mask is None:
                    tensor_args = (
                        src,
                        first_layer.self_attn.in_proj_weight,
                        first_layer.self_attn.in_proj_bias,
                        first_layer.self_attn.out_proj.weight,
                        first_layer.self_attn.out_proj.bias,
                        first_layer.norm1.weight,
                        first_layer.norm1.bias,
                        first_layer.norm2.weight,
                        first_layer.norm2.bias,
                        first_layer.linear1.weight,
                        first_layer.linear1.bias,
                        first_layer.linear2.weight,
                        first_layer.linear2.bias,
                    )
                    if not torch.overrides.has_torch_function(tensor_args):
                        if not torch.is_grad_enabled() or all([not x.requires_grad for x in tensor_args]):
                            if output.is_cuda or 'cpu' in str(output.device):
                                convert_to_nested = True
                                output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not())

        for mod in self.layers:
            if convert_to_nested:
                output = mod(output, src_mask=mask)
            else:
                output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if convert_to_nested:
            output = output.to_padded_tensor(0.)

        if self.norm is not None:
            output = self.norm(output)

        return output