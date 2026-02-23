import os
import requests
from datetime import datetime, timedelta, timezone
from msal import PublicClientApplication
from dotenv import load_dotenv

load_dotenv()

def get_access_token():
    client_id = os.environ.get("AZURE_CLIENT_ID")
    tenant_id = os.environ.get("AZURE_TENANT_ID")

    if not client_id or not tenant_id:
        raise Exception("AZURE_CLIENT_ID veya AZURE_TENANT_ID eksik!")

    print(f"Client ID : {client_id}")
    print(f"Tenant ID : {tenant_id}")
    
    # 'common' → kişisel + kurumsal hesapları birlikte gösterir
    # Böylece takvimi olan doğru hesabı seçebilirsin
    authority_url = "https://login.microsoftonline.com/common"
    app = PublicClientApplication(client_id, authority=authority_url)

    for account in app.get_accounts():
        app.remove_account(account)

    scopes = ["Calendars.Read", "User.Read"]

    print("\n🔑 Device flow başlatılıyor...")
    flow = app.initiate_device_flow(scopes=scopes)
    
    if "user_code" not in flow:
        raise Exception(f"Device flow başlatılamadı: {flow}")

    print(f"\n{flow['message']}\n")
    result = app.acquire_token_by_device_flow(flow)
    
    if "access_token" not in result:
        print(f"Token yok! Hata: {result.get('error')}")
        print(f"Açıklama: {result.get('error_description')}")
        return None
    
    print(f"\n✅ Token alındı!")
    print(f"   Token type : {result.get('token_type')}")
    print(f"   Scope      : {result.get('scope')}")
    return result.get("access_token")

def test_graph(token):
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    # ADIM 1: /me endpoint — en temel test
    print("\n--- TEST 1: /me endpoint ---")
    r = requests.get("https://graph.microsoft.com/v1.0/me", headers=headers)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        me = r.json()
        print(f"✅ Kullanıcı: {me.get('displayName')} ({me.get('mail') or me.get('userPrincipalName')})")
    else:
        print(f"❌ Yanıt: {r.text}")
        print(f"   WWW-Authenticate: {r.headers.get('WWW-Authenticate','')}")
        return

    # ADIM 2: Basit events endpoint
    print("\n--- TEST 2: /me/events (son 5) ---")
    r2 = requests.get(
        "https://graph.microsoft.com/v1.0/me/events?$top=5&$select=subject,start,end,isOnlineMeeting",
        headers=headers
    )
    print(f"Status: {r2.status_code}")
    if r2.status_code == 200:
        evts = r2.json().get('value', [])
        print(f"✅ {len(evts)} etkinlik bulundu:")
        for e in evts:
            print(f"   - {e.get('subject')} | {e.get('start',{}).get('dateTime','')[:16]}")
    else:
        print(f"❌ Yanıt: {r2.text}")
        print(f"   WWW-Authenticate: {r2.headers.get('WWW-Authenticate','')}")
        return

    # ADIM 3: calendarView (bugün)
    print("\n--- TEST 3: /me/calendarView (bugün) ---")
    now = datetime.now(timezone.utc)
    s = now.strftime('%Y-%m-%dT00:00:00Z')
    e = (now + timedelta(days=1)).strftime('%Y-%m-%dT00:00:00Z')

    r3 = requests.get(
        f"https://graph.microsoft.com/v1.0/me/calendarView?startDateTime={s}&endDateTime={e}"
        f"&$select=subject,start,end,isOnlineMeeting,onlineMeeting"
        f"&$orderby=start/dateTime",
        headers={**headers, 'Prefer': 'outlook.timezone="Turkey Standard Time"'}
    )
    print(f"Status: {r3.status_code}")
    if r3.status_code == 200:
        evts = r3.json().get('value', [])
        print(f"✅ Bugün {len(evts)} etkinlik:")
        if not evts:
            print("   (boş — takvimde bugün etkinlik yok veya calendarView farklı sonuç verdi)")
        for e in evts:
            print(f"   - {e.get('subject')} | {e.get('start',{}).get('dateTime','')[:16]} | Teams: {e.get('isOnlineMeeting')}")
    else:
        print(f"❌ Hata: {r3.status_code}")
        print(f"   Yanıt: {r3.text}")
        print(f"   WWW-Authenticate: {r3.headers.get('WWW-Authenticate','')}")

if __name__ == "__main__":
    try:
        token = get_access_token()
        if token:
            test_graph(token)
    except Exception as ex:
        print(f"💥 Hata: {ex}")
