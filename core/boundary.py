#!/usr/bin/env python3
# core/boundary.py

def apply_periodic_yz(pr, ng):
    pr[:, :, 0:ng, :] = pr[:, :, -2*ng:-ng, :]
    pr[:, :, -ng:, :] = pr[:, :, ng:2*ng, :]
    pr[:, :, :, 0:ng] = pr[:, :, :, -2*ng:-ng]
    pr[:, :, :, -ng:] = pr[:, :, :, ng:2*ng]

def apply_outflow_right_x(pr, ng):
    pr[:, -ng:, :, :] = pr[:, -ng-1:-ng, :, :]
