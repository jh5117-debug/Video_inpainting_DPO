# PAI post-maintenance system audit

## identity
dsw-753014-85f54df947-bkp7h
2026-06-25T09:18:11,183067027+08:00
uid=1001(hj) gid=1001(hj) groups=1001(hj)
uid=1001(hj) gid=1001(hj) groups=1001(hj)
hj
0022
/home/hj

## key paths
### /mnt/nas/hj
/mnt/nas/hj
drwxr-xr-x 755 root root 0 0 4096 /mnt/nas/hj
Filesystem     Type  Size  Used Avail Use% Mounted on
172.28.48.25:/ nfs    10P  8.9T   10P   1% /mnt/nas
### /mnt/workspace/hj/nas_hj
/mnt/nas/hj
lrwxrwxrwx 777 root root 0 0 11 /mnt/workspace/hj/nas_hj
Filesystem     Type  Size  Used Avail Use% Mounted on
172.28.48.25:/ nfs    10P  8.9T   10P   1% /mnt/nas
### /home/hj
/home/hj
drwxr-x--- 750 hj hj 1001 1001 4096 /home/hj
Filesystem                                                              Type     Size  Used Avail Use% Mounted on
84830878f2f8c4e11f997b32d4e29036e51c6f9757c0a1bf051e79f401c7f7ad-rootfs overlay  5.3T  9.2G  5.0T   1% /
### /opt/conda
### /mnt/nas/hj/conda_envs
/mnt/nas/hj/conda_envs
drwxr-xr-x 755 root root 0 0 4096 /mnt/nas/hj/conda_envs
Filesystem     Type  Size  Used Avail Use% Mounted on
172.28.48.25:/ nfs    10P  8.9T   10P   1% /mnt/nas

## nvidia-smi
Thu Jun 25 09:18:11 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.95.05              Driver Version: 580.95.05      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA L20X                    On  |   00000000:03:00.0 Off |                    0 |
| N/A   29C    P0             75W /  700W |       0MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA L20X                    On  |   00000000:07:00.0 Off |                    0 |
| N/A   30C    P0             75W /  700W |       0MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA L20X                    On  |   00000000:0B:00.0 Off |                    0 |
| N/A   29C    P0             74W /  700W |       0MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA L20X                    On  |   00000000:0F:00.0 Off |                    0 |
| N/A   27C    P0             75W /  700W |       0MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   4  NVIDIA L20X                    On  |   00000000:14:00.0 Off |                    0 |
| N/A   29C    P0             79W /  700W |       0MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   5  NVIDIA L20X                    On  |   00000000:18:00.0 Off |                    0 |
| N/A   30C    P0             76W /  700W |       0MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   6  NVIDIA L20X                    On  |   00000000:1C:00.0 Off |                    0 |
| N/A   30C    P0             76W /  700W |       0MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   7  NVIDIA L20X                    On  |   00000000:20:00.0 Off |                    0 |
| N/A   28C    P0             75W /  700W |       0MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

## tmux
