# Terrarium - TRUBA Hızlı Başlangıç

## 🚀 Tek Komutla Çalıştır

### 1️⃣ İlk Kurulum (Sadece bir kez)

```bash
# TRUBA'ya bağlan
ssh your_username@172.16.7.1

# Projeyi upload et (lokalde çalıştır)
scp -r Terrarium/ your_username@172.16.7.1:~/

# TRUBA'da kurulumu yap
cd Terrarium
chmod +x setup_truba.sh run.sh
./setup_truba.sh
```

### 2️⃣ Çalıştır (Her seferinde)

**İnteraktif mod (test için):**

```bash
cd Terrarium
./run.sh
```

**SLURM job (production için):**

```bash
cd Terrarium
# İlk çalıştırmadan önce account ID'yi düzenle:
nano run_simulation.slurm  # -> #SBATCH --account=YOUR_ACCOUNT_ID

# Job'u gönder
sbatch run_simulation.slurm

# Logları izle
tail -f logs/slurm-*.out
```

## 📝 Config Dosyası

`examples/configs/meeting_scheduling.yaml` - Simulation parametreleri:

```yaml
simulation:
  max_iterations: 1
  max_conversation_steps: 3
  seed: 42

environment:
  name: MeetingSchedulingEnvironment
  use_real_calendars: false # true = gerçek takvim, false = simulasyon
  num_days: 5
  slots_per_day: 24

llm:
  provider: huggingface # GPU yok ise -> openai veya anthropic
  model: "Qwen/Qwen2.5-7B-Instruct"
  temperature: 0.0
```

## 🔧 Provider Seçenekleri

| Provider      | Gereksinim          | Kullanım                             |
| ------------- | ------------------- | ------------------------------------ |
| `huggingface` | GPU (VRAM: 8-16 GB) | Lokal model, API key gerektirmez     |
| `vllm`        | GPU (VRAM: 8-16 GB) | Hızlı inference, API key gerektirmez |
| `openai`      | API key (.env)      | GPT-4, GPT-4o, gpt-4o-mini           |
| `anthropic`   | API key (.env)      | Claude 3.5 Sonnet                    |

## 📊 Çıktılar

```
logs/MeetingSchedulingEnvironment/baseline_PROVIDER/seed_42/TIMESTAMP/
├── agent_prompts.json         # LLM promptları
├── agent_trajectories.json    # Agent aksiyonları
├── tool_calls.json             # Tool çağrıları
└── blackboard_*.txt            # Blackboard logları
```

## 💡 Örnekler

**GPU ile lokal model:**

```yaml
llm:
  provider: vllm
  model: "Qwen/Qwen2.5-7B-Instruct"
  device: "auto"
```

**API ile cloud model:**

```yaml
llm:
  provider: openai
  model: "gpt-4o-mini"
```

**Gerçek Teams takvimi:**

```yaml
environment:
  use_real_calendars: true
  graph_api:
    client_id: "..."
    tenant_id: "..."
```

## 📖 Detaylı Dokümantasyon

- [TRUBA_SETUP.md](TRUBA_SETUP.md) - Kapsamlı kurulum kılavuzu
- [README.md](README.md) - Proje dokümantasyonu
