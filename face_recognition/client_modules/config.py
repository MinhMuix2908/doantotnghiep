import anyconfig
import munch

cfg = anyconfig.load('settings.yaml')
cfg = munch.munchify(cfg)