from sslgap.methods.base import Frame, SSLMethod
from sslgap.methods.byol import BYOL
from sslgap.methods.dino import DINO
from sslgap.methods.lambdajepa import LambdaJEPA
from sslgap.methods.lejepa import LeJEPA
from sslgap.methods.visreg import VISReg as VISRegHouse
from sslgap.methods.simclr import SimCLR
from sslgap.methods.vicreg import VICReg

METHODS = {m.name: m for m in (LeJEPA, SimCLR, VICReg, BYOL, DINO, LambdaJEPA, VISRegHouse)}
