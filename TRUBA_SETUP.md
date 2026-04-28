# TRUBA HPC'de Terrarium Çalıştırma Kılavuzu

## 📋 Ön Hazırlık

### 1. Projeyi TRUBA'ya Yükle

```bash
# Lokal bilgisayardan TRUBA'ya transfer
scp -r Terrarium/ your_username@172.16.7.1:~/

# veya git clone (eğer repo public ise)
ssh your_username@172.16.7.1
git clone https://github.com/your-repo/Terrarium.git
cd Terrarium
```

### 2. Konfigurasyon Ayarları

Config dosyanızı düzenleyin (`examples/configs/meeting_scheduling.yaml`):

**Simulation Mode (GPU gerektirmez):**

```yaml
environment:
  use_real_calendars: false # Simulasyon modu

llm:
  provider: openai # veya anthropic
  model: "gpt-4o-mini"
  # API key .env dosyasında tanımlı olmalı
```

**Production Mode (GPU gerektirir - vLLM):**

```yaml
environment:
  use_real_calendars: false # veya true (Graph API için)

llm:
  provider: vllm
  model: "Qwen/Qwen2.5-7B-Instruct"
  device: "auto" # GPU otomatik kullanılır
```

### 3. API Keys (.env dosyası)

`.env` dosyası oluşturun:

```bash
# OpenAI API (eğer openai provider kullanıyorsanız)
OPENAI_API_KEY=sk-...

# Anthropic API (eğer anthropic provider kullanıyorsanız)
ANTHROPIC_API_KEY=sk-ant-...

# Microsoft Graph API (use_real_calendars: true için)
AZURE_CLIENT_ID=...
AZURE_TENANT_ID=...
```

## 🚀 Kurulum

### Otomatik Kurulum

```bash
cd Terrarium
chmod +x setup_truba.sh
./setup_truba.sh
```

### Manuel Kurulum

```bash
# 1. Modülleri yükle
module purge
module load centos7.9/comp/python/3.11
module load centos7.9/lib/gmp/6.2.1
module load centos7.9/comp/gcc/11

# 2. Virtual environment oluştur
python -m venv .venv
source .venv/bin/activate

# 3. Paketleri yükle
pip install --upgrade pip
pip install -r requirements.txt

# 4. Crypto modülünü build et
cd crypto
python setup.py install
cd ..
```

## 🎯 Çalıştırma

### Yöntem 1: Interaktif (Test İçin)

```bash
# 1. Modülleri yükle
module load centos7.9/comp/python/3.11
module load centos7.9/lib/gmp/6.2.1

# 2. Virtual environment'ı aktifleştir
source .venv/bin/activate

# 3. Simülasyonu çalıştır
python examples/base_main.py --config examples/configs/meeting_scheduling.yaml
```

### Yöntem 2: SLURM Job (Production İçin)

```bash
# 1. SLURM script'i düzenle
nano run_simulation.slurm
# -> #SBATCH --account=YOUR_ACCOUNT_ID satırını kendi hesabınızla değiştirin

# 2. Job'u gönder
sbatch run_simulation.slurm

# 3. Job durumunu kontrol et
squeue -u $USER

# 4. Log dosyalarını görüntüle
tail -f logs/slurm-JOBID.out  # JOBID'yi squeue'den alın
```

## 📊 Çıktılar

Simülasyon tamamlandığında şu dosyalar oluşur:

```
logs/
  MeetingSchedulingEnvironment/
    baseline_PROVIDER/
      seed_42/
        TIMESTAMP/
          agent_prompts.json       # LLM'e gönderilen promptlar
          agent_trajectories.json  # Agent aksiyon geçmişi
          tool_calls.json          # Tüm tool çağrıları
          blackboard_*.txt         # Blackboard event logları
```

## 🔧 Sorun Giderme

### Crypto Modülü Build Hatası

```bash
# GMP kurulu değilse:
module load centos7.9/lib/gmp/6.2.1

# Veya manuel kurulum:
cd crypto
python setup.py install --verbose
```

### GPU Bulunamadı Hatası (vLLM)

```bash
# CUDA modülünü yükle:
module load centos7.9/comp/cuda/11.8

# GPU'yu kontrol et:
nvidia-smi

# Eğer GPU yok ise config'i düzenle:
# llm.device: "cpu" veya provider: "openai"
```

### Out of Memory (OOM) Hatası

SLURM script'te RAM/GPU memory'yi artır:

```bash
#SBATCH --mem=64G          # 64 GB RAM
#SBATCH --gres=gpu:1       # veya gpu:2 (2 GPU)
```

## 📝 Notlar

- **Simulation mode**: API key gerektirmez, hızlıdır, test için idealdir
- **Production mode + Graph API**: Microsoft hesabı gerektirir, gerçek takvim verileri kullanır
- **vLLM provider**: GPU gerektirir (7B model için ~8-16 GB VRAM)
- **OpenAI/Anthropic provider**: API key gerektirir, TRUBA kredisine yansımaz

## 📞 Destek

TRUBA kullanıcı desteği: destek@truba.gov.tr  
Terrarium dökümanları: [README.md](README.md)
