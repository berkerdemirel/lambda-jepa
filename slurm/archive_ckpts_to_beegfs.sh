#!/bin/bash
#SBATCH --job-name=arch_ckpt
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out
# Part (b) of the 2026-09-08 disk rescue (part (a), the feature store, moved in job 65060659 and is
# now a symlink to BeeGFS). Moves every closed-arm checkpoint to BeeGFS and symlinks it back so the
# paths cited in the cards keep resolving. The two LIVE in1k runs are excluded — they keep writing
# to outputs/ untouched. Split from the features job because that one exited on an rsync delete
# error (the filesystem was too full to even unlink) before reaching this half.
P=/nfs/scistore19/locatgrp/bdemirel/ssl_project
B=/mnt/beegfs/locatgrp/shared/bdemirel/ckpt_archive
mkdir -p $B; cd $P/outputs || exit 1
df -h /nfs/scistore19/locatgrp | tail -1
ls *.pt 2>/dev/null | grep -v "e27v6s400\|e27v6b400" > /tmp/arch_ckpt.$$
echo "moving $(wc -l < /tmp/arch_ckpt.$$) checkpoints"
rsync -a --remove-source-files --files-from=/tmp/arch_ckpt.$$ --info=stats2 $P/outputs/ $B/
rc=$?; echo "rsync rc=$rc (23 = transferred, some source deletes failed)"
n=0; miss=0
while read -r f; do
  [ -f "$B/$f" ] || { echo "  NOT on BeeGFS, left alone: $f"; miss=$((miss+1)); continue; }
  [ -e "$P/outputs/$f" ] && [ ! -L "$P/outputs/$f" ] && rm -f "$P/outputs/$f"   # source delete that rsync could not do
  ln -sfn "$B/$f" "$P/outputs/$f" && n=$((n+1))
done < /tmp/arch_ckpt.$$
rm -f /tmp/arch_ckpt.$$
echo "symlinked back: $n | not transferred: $miss"
df -h /nfs/scistore19/locatgrp | tail -1
echo "ARCHIVE_CKPT_DONE"
