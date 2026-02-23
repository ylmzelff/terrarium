# Microsoft Graph API Integration Guide

**Production-Ready Integration for Real Outlook/Teams Calendars**

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Azure Setup (Optional)](#azure-setup)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [Testing](#testing)
7. [Troubleshooting](#troubleshooting)
8. [FAQ](#faq)

---

## Overview

Terrarium now supports **dual-mode operation**:

- **Simulation Mode** (Default): Generates controlled test availability data
- **Production Mode** (Optional): Fetches real availability from Microsoft Outlook calendars

### Features

✅ **OAuth2 Authentication** via MSAL (Microsoft Authentication Library)  
✅ **Real Calendar Data** from Outlook/Exchange  
✅ **Teams Meeting Creation** with automatic join links  
✅ **Rate Limiting** (2 req/s - Graph API best practice)  
✅ **Timezone Support** with pytz  
✅ **Free for Students** (M365 Education accounts)  
✅ **No Code Changes** - Just config toggle!

---

## Quick Start

### 1. Install Dependencies (Optional)

```bash
# Only needed for production mode
pip install -e ".[graph]"
```

This installs:

- `msal` - Microsoft Authentication Library
- `pytz` - Timezone support

### 2. Toggle Mode in Config

**Simulation Mode (Default)**:

```yaml
# examples/configs/meeting_scheduling.yaml
environment:
  use_real_calendars: false # ✅ No setup needed!
```

**Production Mode**:

```yaml
environment:
  use_real_calendars: true # Requires Azure setup
  graph_api:
    # ... (see Configuration section)
```

That's it! The system automatically chooses the right mode.

---

## Azure Setup (Optional)

**Only needed if `use_real_calendars: true`**

### Option A: University Account (Recommended for Students)

If your university provides M365:

1. **Check Access**: Visit https://portal.azure.com
2. **Login**: Use your university email (e.g., `name@students.university.edu`)
3. **Request Permission**: Email your IT department:

```
Subject: Azure App Registration Request for Academic Project

Hi IT Team,

I'm working on a multi-agent AI system for my graduation project and need
to create an Azure App Registration to test Microsoft Graph API integration.

Required Permissions:
- Calendars.Read (delegated)
- Calendars.ReadWrite (delegated)
- OnlineMeetings.ReadWrite (delegated)

Purpose: Testing calendar availability and Teams meeting creation
Scope: Personal account only (academic project)

Can you assist with this?

Thank you!
[Your Name]
```

### Option B: M365 Developer Program (Free - 90 Days)

**Best for testing without university access**:

1. **Sign Up**: https://developer.microsoft.com/microsoft-365/dev-program
2. **Get Sandbox**: 90-day renewable E5 subscription
3. **Includes**:
   - 25 test users
   - Full Outlook + Teams access
   - All Graph API features

### Azure App Registration Steps

Once you have access to Azure Portal:

#### 1. Create App Registration

1. Visit https://portal.azure.com
2. Go to **Azure Active Directory** → **App registrations**
3. Click **New registration**
4. Fill in:
   - **Name**: `terrarium-meeting-scheduler`
   - **Supported account types**: "Single tenant" (your org only)
   - **Redirect URI**: `Web` → `http://localhost:8000/callback`
5. Click **Register**

#### 2. Add Permissions

1. Go to **API permissions**
2. Click **Add a permission** → **Microsoft Graph** → **Delegated permissions**
3. Search and add:
   - ✅ `Calendars.Read`
   - ✅ `Calendars.ReadWrite`
   - ✅ `OnlineMeetings.ReadWrite`
   - ✅ `User.Read`
4. Click **Grant admin consent** (if you have admin rights)

#### 3. Create Client Secret

1. Go to **Certificates & secrets**
2. Click **New client secret**
3. **Description**: `terrarium-prod-secret`
4. **Expires**: 24 months
5. Click **Add**
6. **⚠️ COPY THE SECRET VALUE NOW** (won't be shown again!)

#### 4. Get IDs

From the **Overview** page, copy:

- **Application (client) ID** - Your `AZURE_CLIENT_ID`
- **Directory (tenant) ID** - Your `AZURE_TENANT_ID`

#### 5. Configure Environment

```bash
# Copy example file
cp .env.example .env

# Edit .env and fill in your values
AZURE_CLIENT_ID=your-client-id-here
AZURE_CLIENT_SECRET=your-secret-here
AZURE_TENANT_ID=your-tenant-id-here
```

**⚠️ Security**: `.env` is in `.gitignore` - never commit secrets to git!

---

## Configuration

### Full Config Example

```yaml
# examples/configs/meeting_scheduling.yaml

environment:
  name: MeetingSchedulingEnvironment

  # ============================================
  # TOGGLE: Simulation vs Production
  # ============================================
  use_real_calendars: false # false=simulation, true=production

  # ============================================
  # Graph API Settings (Production Only)
  # ============================================
  graph_api:
    # Azure credentials (from environment variables)
    client_id: "${AZURE_CLIENT_ID}"
    client_secret: "${AZURE_CLIENT_SECRET}"
    tenant_id: "${AZURE_TENANT_ID}"

    # Timezone for calendar operations
    timezone: "Europe/Istanbul" # Adjust to your timezone

    # Map agent names → real email addresses
    agent_emails:
      AgentA: "alice@university.edu"
      AgentB: "bob@university.edu"
      AgentC: "charlie@university.edu"

  # ============================================
  # Simulation Settings (Simulation Only)
  # ============================================
  intersections: 3 # Guaranteed common slots
  availability_rate: 0.4 # 40% chance for extra slots
  num_days: 1
  slots_per_day: 6
```

### Environment Variables

Use `python-dotenv` or shell environment:

**PowerShell** (Windows):

```powershell
$env:AZURE_CLIENT_ID="your-client-id"
$env:AZURE_CLIENT_SECRET="your-secret"
$env:AZURE_TENANT_ID="your-tenant-id"
```

**Bash** (Linux/Mac):

```bash
export AZURE_CLIENT_ID="your-client-id"
export AZURE_CLIENT_SECRET="your-secret"
export AZURE_TENANT_ID="your-tenant-id"
```

**`.env` file** (Recommended):

```bash
# .env
AZURE_CLIENT_ID=abc123...
AZURE_CLIENT_SECRET=def456...
AZURE_TENANT_ID=ghi789...
```

---

## Usage

### Simulation Mode (Default)

```bash
# No setup needed - just run!
python examples/base_main.py --config examples/configs/meeting_scheduling.yaml
```

**Output**:

```
🔬 Simulation mode: Generating controlled test availability
  🎯 Guaranteed intersection slots: [2, 5, 8]
  AgentA: 4/12 available slots
  AgentB: 5/12 available slots
  ✅ Guaranteed: ALL 2 participants available at slots [2, 5, 8]
```

### Production Mode

```bash
# 1. Install dependencies
pip install -e ".[graph]"

# 2. Set environment variables (see Configuration section)
cp .env.example .env
# Edit .env with your Azure credentials

# 3. Update config
# Set use_real_calendars: true in meeting_scheduling.yaml

# 4. Run!
python examples/base_main.py --config examples/configs/meeting_scheduling.yaml
```

**Output**:

```
📅 Production mode: Fetching real availability from Microsoft Graph API
✅ Graph API client initialized
📅 Fetching availability for AgentA (alice@university.edu)...
  ✅ AgentA: 7/12 available slots (from Outlook calendar)
📅 Fetching availability for AgentB (bob@university.edu)...
  ✅ AgentB: 6/12 available slots (from Outlook calendar)
```

---

## Testing

### Test 1: Dependencies Installed

```bash
python -c "from llm_server.clients.graph_client import GraphAPIClient; print('✅ Graph client available')"
```

### Test 2: Environment Variables

```bash
python -c "import os; print('Client ID:', os.getenv('AZURE_CLIENT_ID', 'NOT SET'))"
```

### Test 3: Simulation Mode

```bash
# Default config already set to simulation
python examples/base_main.py --config examples/configs/meeting_scheduling.yaml
```

Should see: `🔬 Simulation mode: Generating controlled test availability`

### Test 4: Production Mode (requires Azure setup)

```yaml
# meeting_scheduling.yaml
environment:
  use_real_calendars: true
```

```bash
python examples/base_main.py --config examples/configs/meeting_scheduling.yaml
```

Should see: `📅 Production mode: Fetching real availability from Microsoft Graph API`

---

## Troubleshooting

### ❌ `ImportError: No module named 'msal'`

**Solution**: Install Graph API dependencies

```bash
pip install -e ".[graph]"
```

### ❌ `ValueError: Missing required environment variables`

**Solution**: Set Azure credentials

```bash
# Check if variables are set
echo $env:AZURE_CLIENT_ID   # PowerShell
echo $AZURE_CLIENT_ID       # Bash

# Set them in .env file (recommended)
cp .env.example .env
# Edit .env with your values
```

### ❌ `Authentication failed: AADSTS700016`

**Error Message**: "Application not found in the directory"

**Solution**: Your client_id is incorrect or app wasn't created in the right tenant

- Double-check client_id in Azure Portal
- Ensure you're using the right tenant

### ❌ `Authentication failed: AADSTS50076`

**Error Message**: "Due to a configuration change or because you moved to a new location..."

**Solution**: Your organization requires Multi-Factor Authentication (MFA)

- Contact IT to allow "public client flows" for your app
- Or use OAuth2 device code flow (TODO: implement in future version)

### ❌ `403 Forbidden: Insufficient privileges`

**Solution**: Missing Graph API permissions

1. Go to Azure Portal → Your App → API permissions
2. Ensure these are added:
   - Calendars.Read (delegated)
   - Calendars.ReadWrite (delegated)
   - OnlineMeetings.ReadWrite (delegated)
3. Click "Grant admin consent"

### ❌ `429 Too Many Requests`

**Solution**: Rate limit exceeded

- Graph API client has built-in rate limiting (2 req/s)
- If you still hit limits, reduce `num_agents` in config
- Microsoft limit: 2000 requests/minute per app

### ❌ `No email configured for agent`

**Error**: `❌ No email configured for agent: AgentA`

**Solution**: Add email mapping in config

```yaml
graph_api:
  agent_emails:
    AgentA: "alice@university.edu" # ← Add this
    AgentB: "bob@university.edu"
```

---

## FAQ

### Q: Is this free?

**A**: Yes, if you use:

- University M365 account (already included with tuition)
- M365 Developer Program (90 days free, renewable)

Azure App Registration and Graph API calls are always free.

### Q: Do I need to implement this for my project?

**A**: No! Simulation mode is perfect for:

- Academic projects
- Algorithm testing
- Portfolio demonstrations

Production mode is optional for:

- Real-world deployments
- Corporate use cases
- "Wow factor" in demos

### Q: Can I mix simulation and production?

**A**: Not directly, but you can:

1. Test with simulation (quick iterations)
2. Switch to production for final demo
3. Switch back to simulation for regression testing

Just toggle `use_real_calendars` in config.

### Q: What about Google Calendar?

**A**: Not supported yet, but the architecture is ready!

- Copy `graph_client.py` → `google_calendar_client.py`
- Use Google Calendar API v3
- Add `use_google_calendar` config option

### Q: Can I create Teams meetings automatically?

**A**: Yes! The `create_teams_meeting()` method is ready:

```python
client.create_teams_meeting(
    subject="Project Sync",
    start_datetime=datetime(2026, 2, 21, 10, 0),
    end_datetime=datetime(2026, 2, 21, 11, 0),
    attendees=["user1@example.com", "user2@example.com"]
)
# Returns: {"join_url": "https://teams.microsoft.com/l/meetup/..."}
```

TODO: Add this as an agent tool (future work).

### Q: Is this secure?

**A**: Yes, if you follow best practices:

- ✅ Use `.env` file (never commit to git)
- ✅ Rotate secrets every 6 months
- ✅ Use separate credentials for dev/prod
- ✅ OAuth2 with minimal scopes (principle of least privilege)
- ✅ MSAL handles token refresh automatically

### Q: How long does setup take?

**A**:

- Simulation mode: 0 minutes (already working!)
- Azure setup: 15-30 minutes (one-time)
- M365 Developer signup: 5 minutes

### Q: What about privacy?

**A**:

- OT protocol still works in production mode
- Graph API only fetches availability (free/busy)
- Meeting details (subject, attendees) are never exposed
- Each agent only sees their own calendar + final intersection

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│         MeetingSchedulingEnvironment                    │
│                                                         │
│  _generate_availability_for_meeting()                   │
│         │                                               │
│         ├─── use_real_calendars=false ──────────────┐  │
│         │         (Default)                          │  │
│         │                                            │  │
│         │    _generate_simulated_availability()     │  │
│         │    • Controlled test data                 │  │
│         │    • Guaranteed intersections             │  │
│         │    • Agent-specific random seeds          │  │
│         │                                            │  │
│         └─── use_real_calendars=true ──────────────┐│  │
│                   (Production)                      ││  │
│                                                     ││  │
│              _fetch_real_availability()            ││  │
│              • Azure OAuth2 authentication         ││  │
│              • Graph API getSchedule               ││  │
│              • Convert to slot format              ││  │
│              • Fallback to simulation on error     ││  │
│                                                     ││  │
│              ┌─────────────────────────────┐       ││  │
│              │   GraphAPIClient            │       ││  │
│              │   (graph_client.py)         │       ││  │
│              │                             │       ││  │
│              │  • authenticate()           │       ││  │
│              │  • get_availability()       │       ││  │
│              │  • create_teams_meeting()   │       ││  │
│              │  • Rate limiting (2 req/s)  │       ││  │
│              │  • Token refresh            │       ││  │
│              └─────────────────────────────┘       ││  │
│                         ↓                          ││  │
│              Microsoft Graph API v1.0              ││  │
│              https://graph.microsoft.com           ││  │
│                         ↓                          ││  │
│              Microsoft 365 (Outlook/Teams)         ││  │
└─────────────────────────────────────────────────────┘  │
```

---

## Next Steps

✅ **You're all set for simulation mode!** (No action needed)

**Want production mode?**

1. ☐ Install dependencies: `pip install -e ".[graph]"`
2. ☐ Sign up for M365 Developer Program (or ask IT for university access)
3. ☐ Create Azure App Registration (15 minutes)
4. ☐ Configure `.env` with credentials
5. ☐ Toggle `use_real_calendars: true` in config
6. ☐ Run and test!

**Questions?** Open an issue on GitHub or check the [main README](../README.md).

---

## Credits

- **Microsoft Graph API**: https://docs.microsoft.com/graph
- **MSAL Python**: https://github.com/AzureAD/microsoft-authentication-library-for-python
- **M365 Developer Program**: https://developer.microsoft.com/microsoft-365/dev-program

**Author**: Terrarium Development Team  
**Last Updated**: February 2026
