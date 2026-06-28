# Exp39 H20 Transfer Blocked By Disk Margin

Date: 2026-06-28

Status: `H20_TRANSFER_BLOCKED_DISK_MARGIN_LT_20_PERCENT`

## Reason

The PAI MiniMax selected manifest references are small enough to transfer by
task threshold:

- referenced PAI paths: `758`
- existing referenced paths: `758`
- approximate referenced data size: `2.781 GiB`
- MiniMax model symlink target on PAI: about `2.6G`

However, H20 `/home/nvme01` was audited as:

```text
3.4T total, 3.1T used, 367G free, 90% used
```

That leaves roughly `10.8%` free space, below the required `20%` disk-margin
threshold. Therefore no PAI-to-H20 data migration was started.

## Protection

- PAI remained read-only.
- No PAI process was signaled.
- No PAI GPU was used.
- No H20 training was started.
- No partial data copy was launched.

## Required User Decision

Before copying selected MiniMax data to H20, either free enough H20
`/home/nvme01` space to exceed the 20% margin or explicitly authorize copying
despite the margin.
