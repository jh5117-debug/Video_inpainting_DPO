# Exp50 VOID Asset Download

Status: `VOID_ASSETS_BLOCKED`.

Milestone B started after Milestone A was committed and pushed. Pre-download disk and permission checks were run with bounded `du` timeouts. No download was attempted because the required NAS asset/output roots are not writable by user `hj` on PAI.

Blocked paths:

- `/mnt/nas/hj/H20_Video_inpainting_DPO/third_party/VOID`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/void`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/void`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp50_pai_void_adapter_feasibility`

`sudo -n true` failed because sudo requires a password, and root SSH with the available key is not permitted. Therefore Codex cannot safely create/chown the requested target directories in this session.

No VOID repo/model/base/sample data was downloaded, avoiding unapproved fallback writes.

## Required Assets Still Missing

- Official repo: `Netflix/void-model`
- VOID Pass1: `void_pass1.safetensors`
- VOID Pass2: `void_pass2.safetensors`
- Base model: `alibaba-pai/CogVideoX-Fun-V1.5-5b-InP`
- Official sample data / converted VOR-VOID data root

## Next Unblock

Create and grant `hj:hj` ownership for the requested EXP50 directories, or provide an approved fallback root for assets and outputs. After that, Milestone B can resume with actual downloads and checksums.

## Pre-download Audit

```text
# Milestone B pre-download disk/permission audit
dsw-753014-85f54df947-bkp7h
2026-06-30T09:40:48,067897233+08:00
research/exp50-pai-void-adapter-feasibility-20260630
8696dc711c6924b989ce27a9792474d1b94a3f3f
8696dc7 Add Exp50 VOID adapter readback
34844d7 Record Exp23 GPU2456 pair completion
56ee864 Fix Exp23 Stage2 aggregation CLI
c2be705 Record Exp23 first GPU2456 checkpoint
389c9b8 Record Exp23 GPU2-4-5-6 retry launch
f0293d4 Record Exp23 Phy launch blocked by GPU7 ghost
d9d7077 Wire Exp23 Phy two-stage runner
a2455b3 Record Exp23 process-title launch audit
## df -h
Filesystem                                                                                                Size  Used Avail Use% Mounted on
84830878f2f8c4e11f997b32d4e29036e51c6f9757c0a1bf051e79f401c7f7ad-rootfs                                   5.3T   28G  5.0T   1% /
tmpfs                                                                                                      64M     0   64M   0% /dev
tmpfs                                                                                                     931G     0  931G   0% /sys/fs/cgroup
/dev/vda                                                                                                   30G  2.0G   27G   7% /tmp
virtiofs-default                                                                                          7.0T  216G  6.4T   4% /etc/dsw/config
overlay                                                                                                    30G  2.0G   27G   7% /etc/dsw
tmpfs                                                                                                     1.8T   56K  1.8T   1% /dev/shm
rund:1FDr4OBr:cpfs-01000vwrt8a6usy68r6wu-000001.cn-shanghai.cpfs.aliyuncs.com:/pku/                        70T   68T  2.3T  97% /mnt/data/pku
tmpfs                                                                                                     931G  4.1M  931G   1% /etc/hosts
172.28.48.25:/                                                                                             10P   10T   10P   1% /mnt/nas
rund:jPaxcTjS:cpfs-01000vwrt8a6usy68r6wu-000001.cn-shanghai.cpfs.aliyuncs.com:/csgo-datasets-fullsubset/   70T   68T  2.3T  97% /mnt/data/csgo-datasets-fullsubset
ossfs2                                                                                                    512T     0  512T   0% /mnt/data/csgo-datasets-oss
rund:wGCr8iuM:cpfs-01000vwrt8a6usy68r6wu-000001.cn-shanghai.cpfs.aliyuncs.com:/csgo-datasets/              70T   68T  2.3T  97% /mnt/data/csgo-datasets
84830878f2f8c4e11f997b32d4e29036e51c6f9757c0a1bf051e79f401c7f7ad-rootfs                                   5.3T   28G  5.0T   1% /usr/lib/x86_64-linux-gnu/libcuda.so.560.35.05
tmpfs                                                                                                     931G  196K  931G   1% /proc/driver/nvidia/params
tmpfs                                                                                                     931G  4.0K  931G   1% /etc/nvidia/nvidia-application-profiles-rc.d
## du weights
22G	/mnt/nas/hj/H20_Video_inpainting_DPO/weights
## du third_party
151M	/mnt/nas/hj/H20_Video_inpainting_DPO/third_party
## du data
DU_TIMEOUT_OR_FAILED data
## write checks
NOT_WRITABLE /mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo
NOT_WRITABLE /mnt/nas/hj/H20_Video_inpainting_DPO/third_party
NOT_WRITABLE /mnt/nas/hj/H20_Video_inpainting_DPO/weights
NOT_WRITABLE /mnt/nas/hj/H20_Video_inpainting_DPO/data/external
WRITABLE /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch
WRITABLE /mnt/nas/hj/H20_Video_inpainting_DPO/runtime
## sudo/root checks
sudo: a password is required
SUDO_NOT_AVAILABLE

```
