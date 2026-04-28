# 🔒 OT Protocol vs Plain Intersection - Benchmark Suite

Bu benchmark suite, privacy-preserving **Oblivious Transfer (OT)** protokolü ile normal **bitwise AND** işleminin performansını **ayrı ayrı** test edip karşılaştırır.

## 📋 Dosyalar

| Dosya | Açıklama |
|-------|----------|
| `benchmark_ot_timing.py` | Sadece OT protokol timing testi |
| `benchmark_plain_timing.py` | Sadece plain AND timing testi (baseline) |
| `compare_benchmarks.py` | İki test sonucunu karşılaştırır |
| `benchmark_ot_vs_plain.py` | Birleşik benchmark (legacy, optional) |
| `benchmark_ot_vs_plain.ipynb` | Google Colab notebook |

## 🚀 Kullanım

### 1️⃣ Testleri Ayrı Ayrı Çalıştır

```powershell
# Önce Plain test (hızlı, dependency yok)
python tests/benchmark_plain_timing.py

# Sonra OT test (yavaş, crypto module gerekir)
python tests/benchmark_ot_timing.py
```

**Neden ayrı?** OT testinde sürelerde dalgalanmalar olabiliyor, bu yüzden testleri farklı zamanlarda veya farklı koşullarda çalıştırabilirsiniz.

### 2️⃣ Sonuçları Karşılaştır

```powershell
python tests/compare_benchmarks.py
```

Bu script:
- `tests/results/ot_timing_results.json` okur
- `tests/results/plain_timing_results.json` okur
- Karşılaştırma tablosu yazdırır
- `comparison_results.csv` ve `comparison_plot.png` oluşturur

## ⚙️ Parametreler

### benchmark_ot_timing.py

```powershell
# Varsayılan (tüm array boyutları, 100 run)
python tests/benchmark_ot_timing.py

# Özel array boyutları
python tests/benchmark_ot_timing.py --sizes 8 16 32 56

# Daha az run (hızlı test)
python tests/benchmark_ot_timing.py --runs 50

# 256-bit OT (daha güvenli ama yavaş)
python tests/benchmark_ot_timing.py --bit-size 256

# Farklı pattern
python tests/benchmark_ot_timing.py --pattern random

# Tümü birlikte
python tests/benchmark_ot_timing.py --sizes 8 16 32 --runs 50 --pattern zeros --output my_ot_test
```

### benchmark_plain_timing.py

```powershell
# Varsayılan
python tests/benchmark_plain_timing.py

# Özel parametreler
python tests/benchmark_plain_timing.py --sizes 8 16 32 --runs 50 --pattern alternating --output my_plain_test
```

### compare_benchmarks.py

```powershell
# Varsayılan dosyalarla
python tests/compare_benchmarks.py

# Özel dosyalar
python tests/compare_benchmarks.py --ot-file my_ot_test.json --plain-file my_plain_test.json

# Grafik olmadan (sadece tablo)
python tests/compare_benchmarks.py --no-plot
```

## 📊 Array Patterns

Her iki test de 4 farklı pattern destekler:

| Pattern | Açıklama | Örnek (size=8) |
|---------|----------|----------------|
| `zeros` | Tüm slotlar busy (worst case intersection) | `[0,0,0,0,0,0,0,0]` |
| `ones` | Tüm slotlar available (best case intersection) | `[1,1,1,1,1,1,1,1]` |
| `alternating` | Dönüşümlü pattern (no intersection) | Agent A: `[1,0,1,0,1,0,1,0]`<br>Agent B: `[0,1,0,1,0,1,0,1]` |
| `random` | Rastgele binary array | `[1,0,1,1,0,0,1,0]` |

## 📈 Çıktı Dosyaları

Tüm sonuçlar `tests/results/` klasörüne kaydedilir:

```
tests/results/
├── ot_timing_results.json       # OT test sonuçları
├── ot_timing_results.csv
├── plain_timing_results.json    # Plain test sonuçları  
├── plain_timing_results.csv
├── comparison_results.csv       # Karşılaştırma tablosu
└── comparison_plot.png          # Grafikler
```

## 🎯 Örnek Workflow

```powershell
# 1. Virtual environment aktif
.venv\Scripts\Activate.ps1

# 2. Plain baseline test (hızlı)
python tests/benchmark_plain_timing.py --runs 100 --pattern zeros

# 3. OT test (yavaş, farklı zamanda çalıştırabilirsin)
python tests/benchmark_ot_timing.py --runs 100 --pattern zeros

# 4. Sonuçları karşılaştır
python tests/compare_benchmarks.py
```

## 🔬 Google Colab

Eğer Colab'da çalıştırmak istersen:

```python
# 1. Repository clone
!git clone https://github.com/ylmzelff/terrarium.git
%cd terrarium

# 2. Dependencies install
!pip install -q pybind11 matplotlib

# 3. OT module build
!cd crypto && python setup.py install

# 4. Plain test
!python tests/benchmark_plain_timing.py --runs 100

# 5. OT test
!python tests/benchmark_ot_timing.py --runs 100

# 6. Compare
!python tests/compare_benchmarks.py
```

Veya direkt `benchmark_ot_vs_plain.ipynb` notebook'u aç.

## 📋 Varsayılan Array Boyutları

Her iki test de aynı boyutları kullanır (senin belirttiğin boyutlar):

```
8, 16, 32, 56, 112, 224, 240, 448, 480, 960
```

## ⚡ Performans Notları

- **Plain AND**: ~0.001-0.1 ms (çok hızlı, ama privacy yok)
- **OT Protocol**: ~10-100 ms (100-1000x daha yavaş, ama privacy-preserving)
- **Timing dalgalanmaları**: OT testinde sistem yükü, CPU frequency scaling, ve crypto operations nedeniyle dalgalanmalar olabilir
- **Recommendation**: Her testi en az 100 run ile çalıştırın, average almak için

## 🛠️ Troubleshooting

### OT module bulunamıyor
```powershell
cd crypto
python setup.py install
cd ..
```

### Matplotlib yok
```powershell
pip install matplotlib
```

### Results klasörü yok
Otomatik oluşturulur, ama manuel de oluşturabilirsin:
```powershell
mkdir tests\results
```

## 📊 Örnek Çıktı

```
================================================================================
PLAIN INTERSECTION BENCHMARK (Baseline)
================================================================================
Array sizes:  [8, 16, 32, 56, 112, 224, 240, 448, 480, 960]
Runs per size: 100
Input:        Generated (zeros)
================================================================================

📊 Testing size: 8
   Using zeros pattern
   Running 100 iterations... 10 20 30 40 50 60 70 80 90 100 ✓
   ⏱️  Avg:  0.003 ms
   📈 Min:  0.002 ms
   📈 Max:  0.015 ms
   📊 StdDev: 0.002 ms

...

================================================================================
BENCHMARK COMPLETE
================================================================================
```

## 📝 Notes

- Ayrı test dosyaları kullanmanın avantajı: Her testi bağımsız çalıştırabilir, farklı zamanlarda test edebilir, sonuçları karşılaştırabilirsin
- OT test uzun sürebilir (960 array size için ~50 saniye), sabırlı ol
- Timing dalgalanmaları normal, bu yüzden std_dev (standart sapma) da kaydediliyor
