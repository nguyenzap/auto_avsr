# Priority Table

| Metric | Preprocessing Priority | Why |
|--------|----------------------|-----|
| Total CPU Virtual Cores | 🔴 Critical | Parallel worker groups |
| Total System RAM | 🔴 Critical | Dataset caching, avoid disk I/O |
| GPU Name | 🟠 Important | Architecture compatibility (avoid Volta) |
| Number of GPUs | 🟠 Important | Parallel job groups |
| VRAM per GPU | 🟡 Moderate | Only needs ~500MB per detector |
| Total VRAM | 🟡 Moderate | Same as above |
| Total GPU TeraFLOPS | ⚪ Irrelevant | Not compute-bound |
| DLPerf Score | ⚪ Irrelevant | Training benchmark, wrong phase |



# GPU Choice
**8× RTX 3060**
- Compute: 96.3 TFLOPS
- VRAM: 12 GB (96 GB total)
- Max CUDA: 13.1
- PCIe: 3.0, 16x — 9.7 GB/s
- CPUs: 72
- RAM: 193 GB
- Storage: 2325 MB/s (NVMe)
- DLPerf: 80.7