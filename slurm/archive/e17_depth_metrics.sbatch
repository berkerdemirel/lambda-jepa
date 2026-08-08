#!/bin/bash
#SBATCH --job-name=e17-depthmetrics
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out
source /nfs/scistore19/locatgrp/bdemirel/ssl_project/slurm/_env.sh
cd /nfs/scistore19/locatgrp/bdemirel/ssl_project
python experiments/e20f_depth_metrics.py e17_depth_metrics.csv \
  lejepa_ctrl=in100.lejepa.s0.e17c.ext lejepa_sigreg=in100.lejepa.s0.hpull_sigreg.ext \
  lejepa_sigreg_inv=in100.lejepa.s0.hpull_sigreg_inv.ext \
  lejepa_sigreg_t=in100.lejepa.s0.hpull_sigreg_t.ext lejepa_sigreg3=in100.lejepa.s0.hpull_sigreg3.ext \
  simclr_ctrl=in100.simclr.s0.e17c.ext simclr_uniform=in100.simclr.s0.hpull_uniform.ext \
  simclr_uniform_align=in100.simclr.s0.hpull_uniform_align.ext \
  byol_ctrl=in100.byol.s0.e17c.ext byol_align=in100.byol.s0.hpull_align.ext \
  dino_ctrl=in100.dino.s0.e17c.ext dino_protoce=in100.dino.s0.hpull_protoce.ext \
  vicreg_ctrl=in100.vicreg.s0.e17c.ext vicreg_c015=in100.vicreg.s0.hpull_varcov.c015.ext
