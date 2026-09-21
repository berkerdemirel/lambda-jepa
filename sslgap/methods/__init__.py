from sslgap.methods.base import Frame, SSLMethod  # noqa: F401
from sslgap.methods.byol import BYOL  # noqa: F401
from sslgap.methods.dino import DINO  # noqa: F401
from sslgap.methods.lambdajepa import LambdaJEPA  # noqa: F401
from sslgap.methods.ijepa import IJEPA  # noqa: F401
from sslgap.methods.lejepa import LeJEPA  # noqa: F401
from sslgap.methods.visreg import VISReg as VISRegHouse  # noqa: F401
from sslgap.methods.mae import MAE  # noqa: F401
from sslgap.methods.pivot import Pivot  # noqa: F401
from sslgap.methods.simclr import SimCLR  # noqa: F401
from sslgap.methods.supervised import DeiTLite  # noqa: F401
from sslgap.methods.vicreg import VICReg  # noqa: F401

METHODS = {m.name: m for m in (LeJEPA, SimCLR, VICReg, BYOL, DINO, MAE, IJEPA, DeiTLite, Pivot, LambdaJEPA, VISRegHouse)}
