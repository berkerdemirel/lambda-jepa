#!/bin/bash
#SBATCH --job-name=arch_beegfs
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/bdemirel/ssl_project/outputs/%x_%j.out
# The group NFS hit 100% on 2026-09-08 and killed e27v6s400's segment mid-checkpoint (OSError 28).
# Berker's call: MOVE the heavy things to BeeGFS and symlink them back, so every path in every card
# still resolves and nothing is lost. Moves (a) the whole feature store, (b) every checkpoint EXCEPT
# the two live in1k runs' files, which keep writing to outputs/ untouched.
P=/nfs/scistore19/locatgrp/bdemirel/ssl_project
B=/mnt/beegfs/locatgrp/shared/bdemirel
mkdir -p $B/features $B/ckpt_archive
df -h /nfs/scistore19/locatgrp | tail -1

echo "== (a) feature store -> $B/features"
rsync -a --remove-source-files --info=stats2 $P/features/ $B/features/ || { echo "FEATURES RSYNC FAILED"; exit 1; }
find $P/features -type d -empty -delete
[ -d $P/features ] && { echo "features/ not empty after move — leaving it, no symlink"; ls -R $P/features | head; } || ln -s $B/features $P/features
echo "features -> $(readlink -f $P/features)"

echo "== (b) closed-arm checkpoints -> $B/ckpt_archive (live runs e27v6s400 / e27v6b400 excluded)"
cd $P/outputs || exit 1
ls *.pt 2>/dev/null | grep -v "e27v6s400\|e27v6b400" > /tmp/arch_list.$$ 
echo "moving $(wc -l < /tmp/arch_list.$$) files, $(du -ch $(cat /tmp/arch_list.$$) 2>/dev/null | tail -1 | cut -f1)"
rsync -a --remove-source-files --files-from=/tmp/arch_list.$$ --info=stats2 $P/outputs/ $B/ckpt_archive/ || { echo "CKPT RSYNC FAILED"; exit 1; }
n=0; while read -r f; do
  [ -e "$P/outputs/$f" ] && { echo "  still present, skipped: $f"; continue; }
  [ -f "$B/ckpt_archive/$f" ] && { ln -s "$B/ckpt_archive/$f" "$P/outputs/$f" && n=$((n+1)); }
done < /tmp/arch_list.$$
rm -f /tmp/arch_list.$$
echo "symlinked back: $n files"

echo "== after"; df -h /nfs/scistore19/locatgrp | tail -1; df -h /mnt/beegfs | tail -1
echo "ARCHIVE_DONE rc=0"
