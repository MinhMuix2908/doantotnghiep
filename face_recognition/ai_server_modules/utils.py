import rpyc
import numpy as np

def cvtNetrefToNumpy(obj):
    obj = rpyc.classic.obtain(obj)
    return obj