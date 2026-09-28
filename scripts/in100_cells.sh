#!/bin/bash
# ImageNet-100 controlled cells: six methods as released and with SACReg at the backbone, lambda-JEPA with and without it.
#   bash scripts/in100_cells.sh list                  # the 14 cells and their overrides
#   bash scripts/in100_cells.sh train <cell> [seed]   # one cell on one GPU (resumable) -> outputs/<run_id>_ep100.pt
#   bash scripts/in100_cells.sh land <run_id>         # extract -> linear/kNN probes, depth metrics, augmentation thickness
set -e
cd "$(dirname "$0")/.."

declare -A CELL TAG
LEJ="method=lejepa frame=in100_vits16 bs=128 num_classes=100 method.lr=0.001 method.warmup_ep=10 method.eta_min=1e-05 method.grad_clip=1.0"
CELL[lejepa]="$LEJ";                                                              TAG[lejepa]=""
CELL[lejepa+sacreg]="$LEJ +method.h_reg=sacreg +method.h_lamb=0.0146";            TAG[lejepa+sacreg]=sacreg
VIC="method=vicreg frame=in100_vits16 bs=256 num_classes=100"
CELL[vicreg]="$VIC";                                                              TAG[vicreg]=""
CELL[vicreg+sacreg]="$VIC +method.h_reg=sacreg +method.h_lamb=1.548 +method.h_tap=cls"; TAG[vicreg+sacreg]=sacreg
SIM="method=simclr frame=in100_vits16 bs=256 num_classes=100"
CELL[simclr]="$SIM";                                                              TAG[simclr]=""
CELL[simclr+sacreg]="$SIM +method.h_reg=sacreg +method.h_lamb=0.098";             TAG[simclr+sacreg]=sacreg
DIN="method=dino frame=in100_vits16 bs=128 num_classes=100 method.ema_base=0.996 method.local_size=96"
CELL[dino]="$DIN";                                                                TAG[dino]=""
CELL[dino+sacreg]="$DIN +method.h_reg=sacreg +method.h_lamb=0.01";                TAG[dino+sacreg]=sacreg
BYO="method=byol frame=in100_vits16 bs=256 num_classes=100 method.ema_base=0.996"
CELL[byol]="$BYO";                                                                TAG[byol]=""
CELL[byol+sacreg]="$BYO +method.h_reg=sacreg +method.h_lamb=0.0205";              TAG[byol+sacreg]=sacreg
VIS="method=visreg frame=in100_vits16 bs=128 num_classes=100"
CELL[visreg]="$VIS";                                                              TAG[visreg]=""
CELL[visreg+sacreg]="$VIS +method.h_reg=sacreg +method.h_lamb=0.0123";            TAG[visreg+sacreg]=sacreg
OURS="method=lambdajepa frame=in100_vits16 bs=128 num_classes=100 +method.aug=lejepa +method.V=4 \
method.expander_dim=256 method.w_inv=32.8 method.w_floor=157.8 method.z_floor_batch=view_mean \
method.z_d_slice=128 method.h_floor_batch=view_mean method.h_d_slice=128 method.queue_steps=3"
CELL[lambdajepa-zonly]="$OURS method.h_lamb=0.0";                                 TAG[lambdajepa-zonly]=zonly
CELL[lambdajepa]="$OURS method.h_lamb=1.894";                                     TAG[lambdajepa]=hz
ORDER="lejepa lejepa+sacreg vicreg vicreg+sacreg simclr simclr+sacreg dino dino+sacreg byol byol+sacreg visreg visreg+sacreg lambdajepa-zonly lambdajepa"

case "$1" in
  list)
    for c in $ORDER; do echo "$c: ${CELL[$c]}"; done ;;
  train)
    c=${2:?cell}; s=${3:-0}
    python experiments/train.py ${CELL[$c]} ${TAG[$c]:+tag=${TAG[$c]}} seed=$s ;;
  land)
    rid=${2:?run_id}
    python experiments/extract.py ckpt=outputs/${rid}_ep100.pt adapter=native run_id=$rid.extL 'h_layers=[3,6,9]' orbit_v=8
    python experiments/probe.py run_id=$rid.extL
    python experiments/depth_metrics.py $rid.extL_depth_metrics.csv $rid.extL=$rid.extL
    python experiments/thickness_append.py $rid.extL ;;
  *)
    sed -n '2,5p' "$0"; exit 1 ;;
esac
