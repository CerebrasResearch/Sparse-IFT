from math import cos, pi

import torch


def _drop_connect(
    module, name_w,
    dc_rate, dc_init_rate=None,
    dc_begin_iteration=0, dc_end_iteration=None,
    dc_inter_iteration=None, dc_inter_rate=None,
    dc_sched='constant',
    dc_drop_std=None,
    dc_binomialsamp=False,
    unscale=False):
    """
    Helper for `WeightDrop` ie dropconnect.
    """
    with torch.no_grad():
        w = getattr(module, name_w)
        del module._parameters[name_w]
        module.register_parameter(name_w + '_raw', torch.nn.Parameter(w))
        # set initial weight
        raw_w = getattr(module, name_w + '_raw')
        setattr(module, name_w, raw_w.clone())

        original_module_forward = module.forward

        module.unscale = unscale

        if dc_init_rate is None:
            assert dc_sched == 'constant'
            dc_init_rate = dc_rate

        if dc_sched == 'bilinear':
            assert dc_inter_iteration != None
            assert dc_inter_rate != None

        module.dc_rate = dc_init_rate

    def forward(*args, **kwargs):
        raw_w = getattr(module, name_w + '_raw')
        w = raw_w
        if module.training:
            if dc_drop_std is None and not dc_binomialsamp:
                w = torch.nn.functional.dropout(
                    w, p=module.dc_rate, training=module.training
                )
                if module.unscale:
                    w = w * (1 - module.dc_rate)
            else:
                with torch.no_grad():
                    numel = w.numel()
                    num_dense_elem = numel - int(module.dc_rate * numel)

                    p = w.clone().detach().abs().view(-1)
                    if dc_binomialsamp:
                        raise NotImplementedError
                        # NOTE No way to enforce drop fraction # elements
                        mask = torch.distributions.binomial.Binomial(total_count=1, probs=(p / torch.max(p))).sample().view(w.shape).contiguous()
                    else:
                        # add noise to top-k effectively adding stochasticity to it.
                        # if signal washes out noise, then its effectively top-k
                        # if noise washes out signal, then its completely random. 
                        _, indices = torch.topk(
                            p / torch.max(p) + dc_drop_std * torch.randn_like(p),
                            num_dense_elem)

                        mask = torch.zeros_like(
                            p, dtype=torch.bool
                        ).scatter(0, indices, True).view(w.shape).contiguous()
                
                # mult by mask and scale by droprate
                w = w * mask
                if not module.unscale:
                    w = w * (1 / (1 - module.dc_rate))

        if isinstance(w, torch.nn.Parameter):
            w = w.clone()

        setattr(module, name_w, w)

        return original_module_forward(*args, **kwargs)

    setattr(module, 'forward', forward)

    def update_drop_connect_rate(cur_iter):

        if dc_sched == 'constant':
            assert dc_init_rate == dc_rate

        elif dc_sched == 'linear':
            if dc_begin_iteration > cur_iter:
                module.dc_rate = dc_init_rate
            elif dc_begin_iteration <= cur_iter < dc_end_iteration:
                sched_len = dc_end_iteration - dc_begin_iteration
                itrs_into_sched = cur_iter - dc_begin_iteration
                frac_sched_togo = 1 - itrs_into_sched / sched_len
                _dc_rate = (
                    frac_sched_togo * (dc_init_rate - dc_rate)
                    + dc_rate
                )
                module.dc_rate = _dc_rate

            elif cur_iter >= dc_end_iteration:
                module.dc_rate = dc_rate

        elif dc_sched == 'bilinear':
            if dc_begin_iteration > cur_iter:
                module.dc_rate = dc_init_rate

            elif dc_begin_iteration <= cur_iter < dc_inter_iteration:
                sched_len = dc_inter_iteration - dc_begin_iteration
                itrs_into_sched = cur_iter - dc_begin_iteration
                frac_sched_togo = 1 - itrs_into_sched / sched_len
                _dc_rate = (
                    frac_sched_togo * (dc_init_rate - dc_inter_rate)
                    + dc_inter_rate
                )
                module.dc_rate = _dc_rate

            elif dc_inter_iteration <= cur_iter < dc_end_iteration:
                sched_len = dc_end_iteration - dc_inter_iteration
                itrs_into_sched = cur_iter - dc_inter_iteration
                frac_sched_togo = 1 - itrs_into_sched / sched_len
                _dc_rate = (
                    frac_sched_togo * (dc_inter_rate - dc_rate)
                    + dc_rate
                )
                module.dc_rate = _dc_rate
            
            elif cur_iter >= dc_end_iteration:
                module.dc_rate = dc_rate

        elif dc_sched == 'cosine':
            if dc_begin_iteration > cur_iter:
                module.dc_rate = dc_init_rate
            elif dc_begin_iteration <= cur_iter < dc_end_iteration:
                sched_len = dc_end_iteration - dc_begin_iteration
                itrs_into_sched = cur_iter - dc_begin_iteration

                m = cos(pi * itrs_into_sched / sched_len) * 0.5 + 0.5
                _dc_rate = m * (dc_init_rate - dc_rate) + dc_rate
                module.dc_rate = _dc_rate

            elif cur_iter >= dc_end_iteration:
                module.dc_rate = dc_rate

        else:
            raise NotImplementedError(
                f'{dc_sched} sparsity schedual not implemented'
            )

        return module.dc_rate

    setattr(module, 'update_drop_connect_rate', update_drop_connect_rate)    

    def train(mode: bool = True):
        r"""Override module train(mode) func to include setting fwd params for eval mode.
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        if not mode:
            # If eval mode is being envoked
            with torch.no_grad():
                raw_w = getattr(module, name_w + '_raw')
                setattr(module, name_w, raw_w.clone())

        module.training = mode
        for module_child in module.children():
            module_child.train(mode)
        return module
    
    setattr(module, 'train', train)


def init_drop_connect(
        modules,
        dc_rate=0.5, dc_init_rate=None,
        dc_begin_iteration=0, dc_end_iteration=None,
        dc_inter_iteration=None, dc_inter_rate=None,
        dc_sched='constant',
        dc_drop_std=None,
        dc_binomialsamp=False,
        desc={
            torch.nn.Linear: 'weight',
            torch.nn.Conv2d: 'weight',
        },
        unscale=False,
    ):
    for module in modules:
        _drop_connect(
            module,
            desc[type(module)],
            dc_rate=dc_rate,
            dc_init_rate=dc_init_rate,
            dc_begin_iteration=dc_begin_iteration,
            dc_end_iteration=dc_end_iteration,
            dc_inter_iteration=dc_inter_iteration,
            dc_inter_rate=dc_inter_rate,
            dc_sched=dc_sched,
            dc_drop_std=dc_drop_std,
            dc_binomialsamp=dc_binomialsamp,
            unscale=unscale)


def set_drop_connect_rate(model, dc_rate=0.):
    for module in model.modules():
        if hasattr(module, 'dc_rate'):
            setattr(module, 'dc_rate', dc_rate)


if __name__ == '__main__':
    from utils import generic_cv_should_dropconnect
    
    import argparse
    args = argparse.Namespace()
    args.input_size = [3, 32, 32]
    args.num_classes = 2


    b, h = 4, 8
    itr = 2
    model = torch.nn.Sequential(
        torch.nn.Linear(args.input_size[0], h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, args.num_classes),
    )
    model.train()

    desc = {
        torch.nn.Linear: 'weight',
        torch.nn.Conv2d: 'weight',}
    dc_modules = [
        m for n, m in model.named_modules() if generic_cv_should_dropconnect(n, m, args, desc=desc)
    ]
    print(dc_modules)

    init_drop_connect(
        dc_modules, 
        dc_rate=0.5, dc_init_rate=None,
        dc_begin_iteration=0, dc_end_iteration=None, dc_sched='constant',
        desc=desc,
        unscale=False,)

    set_drop_connect_rate(model, dc_rate=0.75)

    print('named parameters')
    print(list(model.named_parameters()))

    for itr in range(itr):
        x = torch.randn(b, args.input_size[0])

        y = model(x)
        loss = y.sum() # proxy loss
        loss.backward()

    print('weights0')
    print([p for n, p in model.named_parameters() if 'weight' in n])
    print('weights1')
    print([m.weight for m in model.modules() if hasattr(m, 'weight')])
    print('grad')
    print([p.grad for n, p in model.named_parameters() if 'weight' in n])

    model.eval()
    print('eval weight')
    print([m.weight for m in model.modules() if hasattr(m, 'weight')])

    for itr in range(itr):
        x = torch.randn(b, args.input_size[0])

        y = model(x)
        loss = y.sum() # proxy loss

    print('eval weight')
    print([m.weight for m in model.modules() if hasattr(m, 'weight')])
