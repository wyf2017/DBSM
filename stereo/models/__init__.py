#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys
import os
sys.path.append(os.path.dirname(__file__))


def model_by_name(args):
    
    name = args.arch.lower()

    # DispNet[S/C]
    if('DispNet'.lower() in name):
        from .DispNet import get_model_by_name
        model = get_model_by_name(args)

    # WSMCnet
    elif('WSMCnet'.lower() in name):
        from .WSMCnet import get_model_by_name
        model = get_model_by_name(args)

    # MBFnet
    elif('MBFnet'.lower() in name):
        from .MBFnet import get_model_by_name
        model = get_model_by_name(args)

    # MTLnet
    elif('MTLnet'.lower() in name.lower()):
        from .MTLnet import get_model_by_name
        model = get_model_by_name(args)

    # Unsupported model
    else:
        raise Exception('Unsupported model: ' + name)
    
    return model



