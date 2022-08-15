import os
import torch
import torch.nn as nn

def load(models: dict, opt, log_dir, epoch, logger):
    """
    load the models
    :param models: models to load {'name': model}, whose pth file named {name}_epoch-{epc}.pth
    :param log_dir: where the pth files are saved
    :param epoch: epoch number to load. if negative, load the latest
    :return:
    """

    # TODO: move to utils
    def _detect_latest(prefix, suffix):
        """
        detect the latest file in log_dir with format <prefix><epoch><suffix>
        :param prefix:
        :param suffix:
        :return: epoch, if here's no checkpoints, return a negative value
        """
        checkpoints = os.listdir(log_dir)
        checkpoints = [f for f in checkpoints if f.startswith(prefix) and f.endswith(suffix)]
        checkpoints = [int(f[len(prefix):-len(suffix)]) for f in checkpoints]
        checkpoints = sorted(checkpoints)
        _epoch = checkpoints[-1] if len(checkpoints) > 0 else None
        return _epoch

    def _remove_dist_module_prefix(k: str):
        return k[7:] if k.startswith('module.') else k

    def _load(model, epoch, prefix, suffix, name):
        if model:
            if epoch < 0:
                epoch = _detect_latest(prefix, suffix)
            if epoch is not None:
                ckpt = torch.load(os.path.join(log_dir, '{}{}{}'.format(prefix, epoch, suffix)),map_location='cpu')
                ckpt = {_remove_dist_module_prefix(k): v for k, v in ckpt.items()}
                kwargs = {}
                if isinstance(model, nn.Module):
                    kwargs['strict'] = False
                model.load_state_dict(ckpt, **kwargs)
                logger.info("loaded {} epoch: {}".format(name, epoch))

    for m_name, m in models.items():
        prefix = m_name + '_epoch-'
        _load(m, epoch, prefix, '.pth', m_name)
        models[m_name] = m

    _load(opt, epoch, 'opt_epoch-', '.pth', 'opt')