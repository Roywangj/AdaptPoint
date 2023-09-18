#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/8 16:44
# @Author  : wangjie
from openpoints.utils import registry
ADAPTMODELS = registry.Registry('adaptmodels')

def build_adaptpointmodels_from_cfg(cfg, **kwargs):
    """
    Build a criterion (loss function), defined by cfg.NAME.
    Args:
        cfg (eDICT):
    Returns:
        criterion: a constructed loss function specified by cfg.NAME
    """
    return ADAPTMODELS.build(cfg, **kwargs)