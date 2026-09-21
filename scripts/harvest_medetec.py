import sys, os, re, time, urllib.request, ssl
from email import policy
from email.parser import BytesParser
from urllib.parse import urljoin, urlparse

CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE
HDRS = {"User-Agent": "Mozilla/5.0"}

def harvest(mhtml_path, dest):
    os.makedirs(dest, exist_ok=True)
    msg = BytesParser(policy=policy.default).parse(open(mhtml_path, "rb"))
    base_url, htmls = None, []
    for part in msg.walk():
        loc = part.get("Content-Location", "")
        if part.get_content_type() == "text/html":
            try: htmls.append(part.get_content())
            except Exception: pass
            if not base_url and loc: base_url = loc
    urls = []
    for html in htmls:
        for m in re.finditer(r'(?:src|href)\s*=\s*["\']([^"\']+\.(?:jpe?g|png))["\']', html, re.I):
            u = urljoin(base_url or "", m.group(1).strip())
            if u.startswith("http") and u not in urls: urls.append(u)
    print("%s -> %d urls found" % (os.path.basename(mhtml_path), len(urls)))
    ok = skip = fail = 0
    for u in urls:
        name = re.sub(r"[^\w.\-]", "_", os.path.basename(urlparse(u).path)) or ("img_%d.jpg" % ok)
        outp = os.path.join(dest, name)
        if os.path.exists(outp): skip += 1; continue
        try:
            data = urllib.request.urlopen(urllib.request.Request(u, headers=HDRS), timeout=30, context=CTX).read()
            if len(data) < 8000: skip += 1; continue
            open(outp, "wb").write(data); ok += 1; time.sleep(0.3)
        except Exception as e:
            fail += 1; print("   fail:", u[:70], str(e)[:50])
    print("   downloaded %d, skipped %d, failed %d" % (ok, skip, fail))

if __name__ == "__main__":
    harvest(sys.argv[1], sys.argv[2])
