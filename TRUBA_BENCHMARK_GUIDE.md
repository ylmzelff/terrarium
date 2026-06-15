# TRUBA Paper-Quality Benchmark Guide

Terrarium OT vs Plain AND benchmark'ını TRUBA'da çalıştırmak için.

## 1. Hazırlık (Local'de bir kez)

```bash
# Scripti kontrol et
python tests/test_paper_comparison.py --sizes 32 64 --runs 1  # Quick test
```

## 2. TRUBA'ya Yükleme

```bash
# TRUBA'ya erişim (SSH ile)
ssh truba.ulakbim.gov.tr

# Proje dizinine git
cd ~/projects/Terrarium  # veya senin path'in

# Scripti kopyala
cp tests/test_paper_comparison.py .
cp run_paper_benchmark_truba.sh .
```

## 3. TRUBA'da Çalıştırma

### Seçenek A: Batch Script ile (Önerilen)

```bash
# Script executable yap
chmod +x run_paper_benchmark_truba.sh

# SLURM'a gönder
sbatch run_paper_benchmark_truba.sh

# Status kontrol et
squeue --user=$USER

# Output izle
tail -f logs/paper_benchmark_*.log
```

### Seçenek B: Direct Python ile

```bash
# İnteractive node al (test için)
srun --pty --time=01:00:00 --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G bash

# Sonra çalıştır
module load python/3.10
source .venv/bin/activate
python tests/test_paper_comparison.py --sizes 8 16 32 56 112 224 240 448 480 960 --runs 3
```

## 4. Sonuçları Kontrol Etme

```bash
# Local'e indir
scp -r truba.ulakbim.gov.tr:~/projects/Terrarium/tests/results/ ./

# CSV dosyası aç
ls -1t tests/results/paper_comparison_*.csv | head -1

# İçeriğini göster
cat tests/results/paper_comparison_YYYYMMDD_HHMMSS.csv
```

## 5. Sorunlar

### ModuleNotFoundError: No module named 'src'

```bash
# TRUBA'da proje root'ta olduğundan emin ol
cd ~/projects/Terrarium
python tests/test_paper_comparison.py
```

### MCP Client Error

- MCP server'ın local'de çalışması gerekebilir
- Veya TRUBA'da MCP server başlat
- `.env` dosyasında `MCP_HTTP_URL` ayarla

### Memory/Timeout Error

`run_paper_benchmark_truba.sh` içinde values'ları artır:
```bash
#SBATCH --mem=64G          # 32G'den 64G'ye
#SBATCH --time=08:00:00    # 4 saat'ten 8 saat'e
```

## 6. Output Files

Benchmark tamamlandıktan sonra:
- `tests/results/paper_comparison_YYYYMMDD_HHMMSS.csv` ← Ana results
- `logs/paper_benchmark_*.log` ← Detaylı log
- `logs/paper_benchmark_*.err` ← Error log (varsa)

## CSV Columns

```
array_size,run_id,seed,ot_seconds,plain_seconds,overhead_seconds,overhead_percent
```

Örnek sonuç:
```
32,1,42,10.523,8.142,2.381,29.2
32,2,43,10.601,8.237,2.364,28.7
64,1,42,21.453,15.327,6.126,40.0
...
```

## Makale Kullanımı

Results CSV dosyasını bu Python script ile visualize et (benchmark sizes ile eşleşen):

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("tests/results/paper_comparison_*.csv")

# Group by size
summary = df.groupby("array_size").agg({
    "ot_seconds": ["mean", "std", "min", "max"],
    "plain_seconds": ["mean", "std", "min", "max"],
    "overhead_percent": ["mean", "std"]
})

print(summary)

# Plot - matching the benchmark visualization
sizes = sorted(df["array_size"].unique())
ot_means = [df[df["array_size"]==s]["ot_seconds"].mean() for s in sizes]
plain_means = [df[df["array_size"]==s]["plain_seconds"].mean() for s in sizes]

plt.figure(figsize=(12, 7))
plt.plot(sizes, ot_means, marker="s", linewidth=2, markersize=8, label="OT Protocol", color="#c53030")
plt.plot(sizes, plain_means, marker="o", linewidth=2, markersize=8, linestyle="--", label="Baseline (Plain AND)", color="#2c5282")
plt.xlabel("Number of Slots", fontsize=12)
plt.ylabel("Runtime (seconds)", fontsize=12)
plt.title("Runtime Comparison vs Slot Size (Average per Group)", fontsize=13)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("paper_results_comparison.pdf", dpi=300, bbox_inches='tight')
plt.show()
```

## TRUBA Komutları Özeti

```bash
# Script hazırlığı - güncellenmiş sizes
sbatch run_paper_benchmark_truba.sh

# Veya manuel olarak
python tests/test_paper_comparison.py --sizes 8 16 32 56 112 224 240 448 480 960 --runs 3

# Status check
squeue -u $USER

# Çıktı izle
tail -f logs/paper_benchmark_*.log

# Sonucu local'e indir
scp -r truba.ulakbim.gov.tr:~/Terrarium/tests/results .

# İş iptal et (gerekirse)
scancel <job_id>
```
