# GPU budget

Cap: `$___` (the user fills it in; also in `CLAUDE.md` Status and `notes/scope.md`). Set a billing
alert before the first session. One row per session; destroy the instance at the end (a stopped
instance still bills its disk).

| Date (UTC) | Provider | Host / instance ID | GPU, driver | Start–end (UTC) | Hours | $/h | Cost | What was done |
|---|---|---|---|---|---|---|---|---|
| 2026-11-13 | vast.ai (on-demand) | instance 60562810, host 136.56.33.107 | RTX 4090 24 GB, driver 595.99.02 | first ssh 23:43Z, sessions 23:44–23:58Z, destroyed by the user after 23:59Z | ~0.33 (from the cost) | $0.428 | $0.14 (vast.ai, from the user) | D0 prep sessions 1 and 2 on one instance: host checks, NVIDIA example, seed 0 (4 bugs found, fixed on the Mac), then all 11 seeds pass |

**Spent:** $0.14
