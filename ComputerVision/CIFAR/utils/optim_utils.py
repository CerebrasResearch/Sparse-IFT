import torch
import torch.optim as optim


def set_weight_decay(
    model, weight_decay, no_weight_decay_list=(),
):
    """
    Sets weight decay for the parameters of a model, excluding specified parameters.

    Args:
        model: The neural network model.
        weight_decay (float): The weight decay value.
        no_weight_decay_list (tuple, optional): List of parameter names to exclude from weight decay.
    """
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if (
            param.ndim <= 1
            or name.endswith(".bias")
            or name in no_weight_decay_list
        ):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay, 'weight_decay': weight_decay},
    ]


def get_optimizer_and_lr_scheduler(args, model):
    """
    Constructs and returns an optimizer and a learning rate scheduler 
    based on the provided arguments.

    Parameters:
        args: The command-line arguments or configuration containing 
                optimization-related settings.
        model: The neural network model.

    Returns:
        optimizer: The optimization algorithm for updating model
            parameters.
        lr_scheduler: The learning rate scheduler to adjust 
            learning rates during training.
    """
    # get optimizer
    weight_decay = None

    if args.filter_norm_and_bias:
        print("No weight decay on norms and biases.")
        no_weight_decay = {}
        if hasattr(model, 'no_weight_decay'):
            no_weight_decay = model.no_weight_decay()

        parameters = set_weight_decay(
            model,
            weight_decay=args.weight_decay,
            no_weight_decay_list=no_weight_decay,
        )
        assert len(parameters) > 0, "Decays not applied properly, recheck!!"
        weight_decay = 0.0
    else:
        parameters = model.parameters()
        weight_decay = args.weight_decay

    opt_name = args.optimizer.lower()
    optimizer = None
    if opt_name == "sgd":
        optimizer = optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=weight_decay,
            nesterov=args.nesterov,
        )
    elif opt_name == "rmsprop":
        optimizer = optim.RMSprop(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=weight_decay,
            eps=0.0316,
            alpha=0.9,
        )
    elif opt_name == "adam":
        optimizer = optim.AdamW(
            parameters, lr=args.lr, weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Expect one of sgd, rmsprop or adam, got {opt_name}")

    print("-" * 60)
    print(f"Created {opt_name} optimizer")

    # get lr_scheduler
    scheduler = args.lr_scheduler.lower()
    if scheduler == "step":
        main_lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
        )
    elif scheduler == "cosineannealing":
        main_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs - args.lr_warmup_epochs,
            eta_min=args.lr_min,
        )
    elif scheduler == "multistep":
        main_lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma
        )
    elif scheduler == "exponential":
        main_lr_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.lr_gamma
        )
    else:
        raise RuntimeError(
            f"Invalid lr scheduler {scheduler} not supported currently!!"
        )

    lr_str = f"Created {scheduler} lr scheduler"

    if args.lr_warmup_epochs > 0:
        warmup_method = args.lr_warmup_method.lower()
        if warmup_method == "linear":
            warmup_lr_scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=args.lr_warmup_decay,
                total_iters=args.lr_warmup_epochs,
            )
        elif warmup_method == "constant":
            warmup_lr_scheduler = optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=args.lr_warmup_decay,
                total_iters=args.lr_warmup_epochs,
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{warmup_method}' not supported currently!!"
            )
        lr_str += " with a {warmup_method} warmup."
        lr_scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, main_lr_scheduler],
            milestones=[args.lr_warmup_epochs],
        )
    else:
        lr_str += "."
        lr_scheduler = main_lr_scheduler

    print(f"{lr_str}")

    return optimizer, lr_scheduler


def compute_explicit_weight_decay_low_rank(
    model, weight_decay, no_wd_bn_bias, apply_on_folded=True
):
    '''Computes weight-decay on the reconstructed weight-tensor for 
        stochastic low-rank factorized layers.

    Args:
        model (torch.nn.Module): DNN model.
        weight_decay (float): magnitude of weight-decay.
        no_wd_bn_bias (bool): if True, we do not apply weight-decay 
            on bn and bias.
        apply_on_folded (bool): if True, we apply weight-decay by 
            folding low-rank layers.
    '''

    loss_weight_decay = torch.tensor(0.0, requires_grad=True)
    model_parameters = dict(model.named_parameters())
    model_modules = dict(model.named_modules())
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1:
            # Catches Bias, and BN params
            if no_wd_bn_bias:
                # Skip weight-decay on BN and bias
                continue
            loss_weight_decay = (
                loss_weight_decay + weight_decay * 0.5 * (param ** 2).sum()
            )
        else:
            # Catches all the weight-tensors
            if apply_on_folded:
                if 'U_' in name:
                    # We will fetch value of weight-decay computed and stored in the module
                    name_split = name.split('.')
                    module_name = '.'.join(name_split[:3])
                    _weight_norm = model_modules[
                        module_name
                    ].dynamic_weight_norm
                    loss_weight_decay = (
                        loss_weight_decay + weight_decay * 0.5 * _weight_norm
                    )
                    # loss_weight_decay = (
                    #     loss_weight_decay
                    #     + weight_decay * 0.5 * (param ** 2).sum()
                    # )
                elif 'V_' in name:
                    # We do not apply weight-decay on V component, we apply on W 
                    # constructed by folding U and V.
                    continue
                    # loss_weight_decay = (
                    #     loss_weight_decay
                    #     + weight_decay * 0.5 * (param ** 2).sum()
                    # )
                elif 'sparse_conv' in name:
                    loss_weight_decay = (
                        loss_weight_decay
                        + weight_decay * 0.5 * (param ** 2).sum()
                    )
                    # We do not apply weight-devay on sparse_conv component,
                    # we apply on W constructed by folding UV and sparse(W).
                    # continue
                    # loss_weight_decay = loss_weight_decay + weight_decay*0.5*(param**2).sum()
                else:
                    # On all other parameters we apply weight-decay
                    loss_weight_decay = (
                        loss_weight_decay
                        + weight_decay * 0.5 * (param ** 2).sum()
                    )
            else:
                loss_weight_decay = (
                    loss_weight_decay + weight_decay * 0.5 * (param ** 2).sum()
                )
    return loss_weight_decay
