"""

 2019 (c) piteren

    some encoders

"""

from pypaq.neuralmess.get_tf import tf
from pypaq.neuralmess.base_elements import my_initializer, list_of_layers
from pypaq.neuralmess.layers import zeroes, lay_dense, lay_conv1D, attn, tf_drop

# TODO: probably in tf 1.15.2 histogram 'family=' should be replaced with 'description=' << check it on histogram case


# DRT layer (based on Transformer scaling two denses concept)
def lay_DRT(
        input,
        name=           'lay_DRT',  # scope name, be careful when stacked since auto_reuse
        hist_name=      None,       # family name of histogram
        do_scaled_dns=  False,      # two denses (True) or single dense (False)
        dns_scale=      4,          # scale for two denses
        activation=     tf.nn.relu, # gelu is really worth a try
        lay_dropout=    0.0,        # dropout after dense/s
        add_res=        True,       # residual yes/no
        res_dropout=    0.0,        # dropout on residual connection
        training_flag=  None,       # training flag tensor (for dropout)
        initializer=    None,
        seed=           12321):

    if not hist_name: hist_name = name
    lay_width = input.shape[-1]
    if initializer is None: initializer = my_initializer(seed)
    hist_summ = []

    with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):

        # layer_norm
        output = tf.keras.layers.LayerNormalization(axis=-1)(input)

        hist_summ.append(tf.summary.histogram('a_input', output, family=hist_name))

        # two denses (up-scale >> down-scale - transformer like)
        if do_scaled_dns:

            # dense (scale up), no activation
            output = lay_dense(
                input=          output,
                units=          int(lay_width * dns_scale),
                activation=     None,
                use_bias=       True,
                initializer=    initializer,
                seed=           seed,
                name=           'denseS')
            hist_summ.append(tf.summary.histogram('b_denseUp', output, family=hist_name))

            # activation
            output = activation(output)
            zsL = [zeroes(output)]  # zeroes list
            hist_summ.append(tf.summary.histogram('c_activation', output, family=hist_name))

            # dense (scale down), no activation
            output = lay_dense(
                input=          output,
                units=          lay_width,
                activation=     None,
                name=           'DRTdenseNA',
                use_bias=       True,
                initializer=    initializer,
                seed=           seed)
            hist_summ.append(tf.summary.histogram('d_denseDown', output, family=hist_name))

        # just dense
        else:
            output = lay_dense(
                input=          output,
                units=          lay_width,
                activation=     activation,
                use_bias=       True,
                initializer=    initializer,
                seed=           seed,
                name=           'dense')
            zsL = [zeroes(output)]  # zeroes list

        # layer dropout
        if lay_dropout:
            output = tf.layers.dropout(
                inputs=     output,
                rate=       lay_dropout,
                training=   training_flag,
                seed=       seed)

        if add_res:
            # residual dropout
            if res_dropout:
                input = tf.layers.dropout(
                    inputs=     input,
                    rate=       res_dropout,
                    training=   training_flag,
                    seed=       seed)
            output += input # residual
            hist_summ.append(tf.summary.histogram('e_residual', output, family=hist_name))

    return{
        'output':       output,
        'hist_summ':    hist_summ,
        'zeroes':       zsL}

# Deep Residual encoder based on stacked lay_DRT
def enc_DRT(
        input,
        name=               'enc_DRT',
        in_dropout=         0.0,        # dropout on input
        shared_lays :bool=  False,      # shared variables in enc_layers
        n_layers=           6,
        lay_width :int=     None,       # for None matches input width
        do_scaled_dns=      True,
        dns_scale=          4,          # scale(*) of first dense
        activation=         tf.nn.relu, # gelu is really worth a try
        lay_dropout=        0.0,        # dropout after two denses
        add_res=            True,
        res_dropout=        0.0,        # dropout on residual
        training_flag=      None,       # training flag tensor (for dropout)
        initializer=        None,
        seed=               12321,
        n_hist=             4,          # number of histogram layers (for TB)
        verb=               0):

    input_width = input.shape[-1]
    lay_width_matched = ''
    if lay_width is None:
        lay_width = input_width
        lay_width_matched = '(lay_width taken form input width)'
    if verb>0: print(f'\nBuilding DRTencoder ({n_layers}x{lay_width}) {lay_width_matched}...')

    if initializer is None: initializer = my_initializer(seed)

    hist_summ = []
    hist_layers = list_of_layers(n_layers, n_select=n_hist)
    if verb>1: print(' > histogram layers of DRTencoder:', hist_layers)

    zsL = [] # zeroes list
    with tf.variable_scope(name):

        # input dropout
        if in_dropout:
            input = tf.layers.dropout(
                inputs=     input,
                rate=       res_dropout,
                training=   training_flag,
                seed=       seed)

        # input projection, no activation
        if input_width != lay_width:
            input = lay_dense(
                input=          input,
                units=          lay_width,
                use_bias=       False,
                initializer=    initializer,
                seed=           seed)
            if verb>0: print(f'projected input to layWidth({lay_width}) since it differs({input_width})')

        input = tf.keras.layers.LayerNormalization(axis=-1)(input) # input layer_norm

        output = input # for 0 layers case
        for nL in range(n_layers):

            lay_name = f'DRLay_{nL}' if not shared_lays else 'DRLay_shared'
            lay_out = lay_DRT(
                input=          output,
                name=           lay_name,
                hist_name=      name,
                do_scaled_dns=  do_scaled_dns,
                dns_scale=      dns_scale,
                activation=     activation,
                lay_dropout=    lay_dropout,
                add_res=        add_res,
                res_dropout=    res_dropout,
                training_flag=  training_flag,
                initializer=    initializer,
                seed=           seed)

            output = lay_out['output']
            if nL in hist_layers: hist_summ.append(lay_out['hist_summ'])
            zsL += lay_out['zeroes']

    return {
        'output':       output,
        'hist_summ':    hist_summ,
        'zeroes':       zsL}

# CNN 1D encoder (for sequences, LN > CNN > act > drop > RES), number of parameters: n_layers*kernel*input.width*n_filters
def enc_CNN(
        input :tf.Tensor,
        history :tf.Tensor=                 None,       # optional history(state) tensor with shape [bsz, n_layers ,kernel-1, n_filters], >> masked cnn
        name=                               'enc_CNN',
        time_drop=                          0.0,
        feat_drop=                          0.0,
        # layer
        shared_lays :bool=                  False,      # shared variables in enc_layers
        n_layers :int=                      12,         # num of layers
        kernel :int=                        3,          # layer kernel
        n_filters :int=                     128,        # num of filters
        activation=                         tf.nn.relu, # global enc activation func, gelu is really worth a try
        lay_drop : float or None=           0.0,
        # lay_DRT
        do_ldrt=                            False,      # lay DRT - build or not
        ldrt_scaled_dns=                    True,       # lay DRT - build scaled two denses
        ldrt_scale: int or None=            4,          # lay DRT scale of first dense
        ldrt_drop: float or None=           0.0,        # lay DRT dropout
        ldrt_res=                           True,       # lay DRT - do residual connection
        ldrt_res_drop :float or None=       0.0,        # lay DRT residual dropout

        training_flag :tf.Tensor or bool=   None,       # dropout training flag tensor
        initializer=                        None,
        seed :int=                          12321,
        n_hist :int=                        4,          # number of histogram layers
        verb=                               0):

    if verb>0: print(f'\n *** enc_CNN *** Building {name} ({n_layers}x{n_filters})...')

    if initializer is None: initializer = my_initializer(seed)

    # manage history
    history_lays = None
    if history is not None:
        history_lays = tf.unstack(history, axis=-3)
        if verb>1: print(f' > state_lays len {len(history_lays)} of: {history_lays[0]}')

    hist_summ = []
    hist_layers = list_of_layers(n_layers, n_select=n_hist)
    if verb>1: print(f' > histogram layers of cnn encoder: {hist_layers}')

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        input_lays = [] # here we will store inputs of the following layers to extract the state (history)
        zsL = [] # zeroes

        # tf dropout
        if time_drop or feat_drop:
            input = tf_drop(
                input=      input,
                time_drop=  time_drop,
                feat_drop=  feat_drop,
                train_flag= training_flag,
                seed=       seed)

        # input projection - to match n_filters and input width
        if verb>1: print(f' > encoder input: {input}')
        if input.shape[-1] != n_filters:
            input = lay_dense(
                input=          input,
                units=          n_filters,
                name=           'enc_input_projection',
                initializer=    initializer)
            if verb>1: print(f' > encoder projected input: {input}')

        output = input # for 0 layers case
        sub_output = input # first input
        for depth in range(n_layers):

            lay_name = f'enc_CNN_lay_{depth}' if not shared_lays else 'enc_CNN_lay_shared'
            if verb>1: print(f'<< layer {lay_name}:')

            lay_input = tf.concat([history_lays[depth], sub_output], axis=-2) if history_lays else sub_output
            if verb>1:
                print(f' > sub_output (previous): {sub_output}')
                print(f' > lay_input (eventually padded): {lay_input}')
            input_lays.append(lay_input)

            hist_lay = depth in hist_layers

            with tf.variable_scope(lay_name):

                if hist_lay: hist_summ.append(tf.summary.histogram('a_lay_in', lay_input, family=name))

                # LN
                lay_input = tf.keras.layers.LayerNormalization(axis=-1)(lay_input)
                if hist_lay: hist_summ.append(tf.summary.histogram('b_LN', lay_input, family=name))

                # conv no activation
                output = lay_conv1D(
                    input=          lay_input,
                    name=           'conv1D',
                    kernels=        kernel,
                    filters=        n_filters,
                    activation=     None,
                    initializer=    initializer,
                    padding=        'same' if history is None else 'valid',
                    seed=           seed,
                    verb=           0)
                if hist_lay: hist_summ.append(tf.summary.histogram('c_cnn', output, family=name))

                # activation
                if activation:
                    output = activation(output)
                    zsL += [zeroes(output)]  # catch zeroes
                    if hist_lay: hist_summ.append(tf.summary.histogram('d_activation', output, family=name))

                # dropout
                if lay_drop:
                    output = tf.layers.dropout(
                        inputs=     output,
                        rate=       lay_drop,
                        training=   training_flag,
                        seed=       seed)
                    if hist_lay: hist_summ.append(tf.summary.histogram('e_drop', output, family=name))

                # RES, here we take sub_output, since lay_input may be padded by history
                output += sub_output
                if hist_lay: hist_summ.append(tf.summary.histogram('f_residual', output, family=name))

                if verb>1: print(f' > output (layer): {output}')

                if do_ldrt:
                    lay_out = lay_DRT(
                        input=          output,
                        name=           lay_name + '_lay_DRT',
                        hist_name=      name,
                        do_scaled_dns=  ldrt_scaled_dns,
                        dns_scale=      ldrt_scale,
                        activation=     activation,
                        lay_dropout=    ldrt_drop,
                        add_res=        ldrt_res,
                        res_dropout=    ldrt_res_drop,
                        training_flag=  training_flag,
                        initializer=    initializer,
                        seed=           seed)
                    output = lay_out['output']
                    zsL += lay_out['zeroes']
                    if hist_lay: hist_summ.append(lay_out['hist_summ'])

                sub_output = output

    output = tf.keras.layers.LayerNormalization(axis=-1)(output) # final LN

    # prepare fin_state
    fin_state = None
    if history is not None:
        state = tf.stack(input_lays, axis=-3)
        if verb>1: print(f' > state (stacked): {state}')
        fin_state = tf.split(state, num_or_size_splits=[-1, kernel-1], axis=-2)[1]
        if verb>1: print(f' > fin_state (split): {fin_state}')

    if verb>1: print(f' > {name} output: {output}')
    return {
        'output':       output,
        'state':        fin_state, # history for next
        'hist_summ':    hist_summ,
        'zeroes':       zsL}

# TODO: is family_name in tblock correctly used (following tblocks...)?
# transformer encoder
def enc_TNS(
        in_seq,                                         # input sequence embeddings [batch, seq, emb]
        name=                               'enc_TNS',
        seq_out :bool=                      True,       # transformer seq2seq, if False seq2one (Task Attention Transformer)
        add_PE :bool=                       True,       # add positional embeddings
        do_LN :bool=                        True,       # do layer norm in blocks
        shared_lays :bool=                  False,      # shared variables in blocks
        n_blocks=                           12,
        n_heads=                            8,
        dense_mul :int or float=            4,          # dense (after att) scale
        activation=                         tf.nn.relu, # dense activation
        max_seq_len=                        100,        # used only to set shape (axis 0) of positional embeddings
        dropout=                            0.0,        # dropout of FC after attention
        dropout_att=                        0.0,        # dropout of attention probabilities
        training_flag :tf.Tensor or bool=   None,       # training flag (bool or tensor) for dropout
        initializer=                        None,
        seed=                               12321,
        n_hist=                             4,          # number of histogram layers
        verb=                               0):

    if initializer is None: initializer = tf.truncated_normal_initializer(stddev=0.01, seed=seed)

    # split feats(-1) for heads
    def split_heads(x, n_heads):
        x = tf.split(x, n_heads, axis=-1) # list of tensors
        return tf.stack(x, axis=-3)  # [batch, head, seq, feats]

    # merge heads over feats(-1)
    def merge_heads(x):
        x = tf.unstack(x, axis=-3)
        return tf.concat(x, axis=-1)

    # multi_head_attention for input
    def mh_attn(
            in_seq,         # input sequence [batch, seq, feats]
            query,          # None for self attention, otherwise TAT [batch,1,feats]
            n_heads,
            dropout_att,
            training_flag,
            initializer,
            seed):

        # input projection of in_seq for KQV or KV(if query)
        width = in_seq.shape[-1].value
        proj_size = 3 if query is None else 2
        c = lay_dense(
            input=          in_seq,  # [batch, seq, feats]
            units=          width*proj_size,
            name=           'mhProj',
            activation=     None,
            initializer=    initializer,
            seed=           seed)
        ins_split = tf.split(c, proj_size, axis=-1) # split projected

        if query is not None:
            #q = query # projection for Q is not needed (at least with 1 head) # TODO: here previous version without projection
            q = lay_dense(
                input=          query,  # [batch, seq, feats]
                units=          width,
                name=           'mhProj_query',
                activation=     None,
                initializer=    initializer,
                seed=           seed)
            k,v = ins_split
        else:
            q,k,v = ins_split
        q = split_heads(q, n_heads)
        k = split_heads(k, n_heads)
        v = split_heads(v, n_heads)

        # attention
        att_out = attn(
            q=          q,
            k=          k,
            v=          v,
            dropout=    dropout_att,
            drop_flag=  training_flag,
            seed=       seed)
        a = att_out['attention']
        a = merge_heads(a)
        return {
            'attention':    a,
            'att_vals':     att_out['att_weights']}

    # transformer block
    def tblock(
            in_seq,
            task_query,
            do_LN,
            n_heads,
            dropout_att,
            dense_mul,
            dense_activation,
            dropout,
            training_flag,
            initializer,
            family_name,
            seed):

        hist_summ = []

        output = in_seq
        if task_query is None:
            hist_summ.append(tf.summary.histogram('a_inputSeq', output, family=family_name))
            # layer norm on seq
            if do_LN:
                output = tf.keras.layers.LayerNormalization(axis=-1)(output)
                hist_summ.append(tf.summary.histogram('b_inputSeqLN', output, family=family_name))
        else:
            hist_summ.append(tf.summary.histogram('a_inTaskQuery', task_query, family=family_name))
            # layer norm on task_query
            if do_LN:
                task_query = tf.keras.layers.LayerNormalization(axis=-1)(task_query)
                hist_summ.append(tf.summary.histogram('b_taskQueryLN', task_query, family=family_name))

        # multi-head self attention
        mha_out = mh_attn(
            in_seq=         output,
            query=          task_query,
            n_heads=        n_heads,
            dropout_att=    dropout_att,
            training_flag=  training_flag,
            initializer=    initializer,
            seed=           seed)
        output =    mha_out['attention']
        att_vals =  mha_out['att_vals']
        hist_summ.append(tf.summary.histogram('c_mhAttn', output, family=family_name))

        # dense without activation
        output = lay_dense(
            input=          output,
            units=          output.shape[-1].value,
            name=           'afterAttProj',
            initializer=    initializer,
            seed=           seed)
        hist_summ.append(tf.summary.histogram('d_denseAftAtt', output, family=family_name))

        if dropout:
            output = tf.layers.dropout(
                inputs=     output,
                rate=       dropout,
                training=   training_flag,
                seed=       seed)

        # residual 1
        if task_query is None:
            res1_out = in_seq + output
            hist_summ.append(tf.summary.histogram('e_res_onInputSeq', res1_out, family=family_name))
        else:
            res1_out = task_query + output
            hist_summ.append(tf.summary.histogram('e_res_onTaskQuery', res1_out, family=family_name))

        out_drtlay = lay_DRT(
            input=          res1_out,
            do_scaled_dns=  True,
            dns_scale=      dense_mul,
            activation=     dense_activation,
            lay_dropout=    dropout,
            res_dropout=    0.0, # TODO: make experiments
            training_flag=  training_flag,
            initializer=    initializer,
            seed=           seed)
        output =        out_drtlay['output']
        hist_summ +=    out_drtlay['hist_summ']
        zsL =           out_drtlay['zeroes']

        return {
            'output':       output,
            'hist_summ':    hist_summ,
            'att_vals':     att_vals,
            'zeroes':       zsL}

    width = in_seq.shape[-1]        # sequence width (feats)
    seq_len = tf.shape(in_seq)[-2]  # sequence length (time)

    if verb>0:
        print(f'\nBuilding {name} (transformer encoder) ({n_blocks}x{width}, denseMul {dense_mul:.1f}), ')
        print(f' > dropout: {dropout:.2f} {dropout_att:.2f}(att)')
        print(' > seq2seq mode...') if seq_out else print(' > task attention mode...')

    hist_layers = list_of_layers(n_blocks, n_select=n_hist)
    if verb>1: print(' > histogram layers of transformer encoder:', hist_layers)

    with tf.variable_scope(name):

        hist_summ = []  # list of histogram summaries

        if verb>1: print(' > transformer input', in_seq)
        hist_summ.append(tf.summary.histogram('a_transformerInput', in_seq, family=name))

        # init task_query (for first block - input averaged over time (seq))
        task_query = None
        if not seq_out:
            if do_LN: in_seq = tf.keras.layers.LayerNormalization(axis=-1)(in_seq)
            task_query = tf.reduce_mean(in_seq, axis=-2, keep_dims=True)# [batch,1,feats]
            """ experimental variable for task_query
            task_query = tf.get_variable(
                name=           'task_query',
                shape=          [1,width],
                initializer=    initializer)
            """
            if verb>1: print(' > first task_query (reduced input) for TAT', task_query)

        # positional embedding
        if add_PE:
            pos_emb_var = tf.get_variable(
                name=           'tnsPosEmb',
                shape=          [max_seq_len, width],
                initializer=    initializer)
            in_seq += tf.nn.embedding_lookup(params=pos_emb_var, ids=tf.range(seq_len))
            if verb>1: print(' > added positional embedding to the input...')
            hist_summ.append(tf.summary.histogram('b_transformerPosEmbInput', in_seq, family=name))

        if verb>1: print(f' > building {n_blocks} blocks of transformer...')
        att_vals = [] # list of block attention values
        zsL = []
        block_output = None
        for nB in range(n_blocks):
            hist_lay = nB in hist_layers
            lay_name = f'block_{nB}' if not shared_lays else 'block_shared'
            with tf.variable_scope(lay_name, reuse=tf.AUTO_REUSE):
                bo_dict = tblock(
                    in_seq=             in_seq,
                    task_query=         task_query,
                    do_LN=              do_LN,
                    n_heads=            n_heads,
                    dropout_att=        dropout_att,
                    dense_mul=          dense_mul,
                    dense_activation=   activation,
                    dropout=            dropout,
                    training_flag=      training_flag,
                    initializer=        initializer,
                    family_name=        name,
                    seed=               seed)
                block_output = bo_dict['output']
                if task_query is None: in_seq = block_output
                else: task_query = block_output

                zsL +=                      bo_dict['zeroes']
                if hist_lay: hist_summ +=   bo_dict['hist_summ']
                att_block_vals =            bo_dict['att_vals'] #[batch,head,query_n or seq,seq]
                att_vals.append(att_block_vals)

        if task_query is None:  output = block_output
        else:                   output = tf.squeeze(task_query, axis=-2)

        if do_LN: output = tf.keras.layers.LayerNormalization(axis=-1)(output)

        hist_summ.append(tf.summary.histogram('c_transformer_out', output, family=name))

    if verb>1: print(f' > {name} output {output}')
    return {
        'output':       output,
        'hist_summ':    hist_summ,
        'att_vals':     att_vals,
        'zeroes':       zsL}


if __name__ == "__main__":

    n_lays = 3
    width = 50
    kernel = 5

    input_PH = tf.placeholder(
        name=   'input_PH',
        dtype=  tf.float32,
        shape=  [None, 20, width])  # [bsz,seq,width]

    # uncomment for masked CNN
    #"""
    state_PH = tf.placeholder(
        name=   'state_PH',
        dtype=  tf.float32,
        shape=  [None, n_lays, kernel-1, width])  # [bsz,n_lay,2,width]
    #"""
    #state_PH = None

    encoder = enc_CNN(
        input=      input_PH,
        history=    state_PH,
        n_layers=   n_lays,
        kernel=     kernel,
        n_filters=  width,
        verb=       2)

