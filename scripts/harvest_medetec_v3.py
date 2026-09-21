"""
Medetec harvester v3: downloads FULL-SIZE images directly from
medetec.co.uk galleries (thumbnails/ -> images/ URL swap verified
against the live site structure on 21 Sep 2026).
Run:  python scripts/harvest_medetec_v3.py
"""
import os, re, time, io, urllib.request, ssl
from urllib.parse import urljoin, unquote, urlparse
from PIL import Image

CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE
HDRS = {"User-Agent": "Mozilla/5.0"}
ROOT = "data/raw/medetec"

GALLERIES = [
    ("https://www.medetec.co.uk/slide%20scans/foot-ulcers/index.html", "diabetic"),
    ("https://www.medetec.co.uk/slide%20scans/pressure-ulcer-images-a/index.html", "pressure"),
    ("https://www.medetec.co.uk/slide%20scans/pressure-ulcer-images-b/index.html", "pressure"),
    ("https://www.medetec.co.uk/slide%20scans/leg-ulcer-images/index.html", "_venous_inbox"),
    ("https://www.medetec.co.uk/slide%20scans/leg-ulcer-images-2/index.html", "_venous_inbox"),
]

def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HDRS), timeout=40, context=CTX).read()

def main():
    grand = 0
    for page_url, dest in GALLERIES:
        gname = unquote(urlparse(page_url).path).split("/")[-2]
        try:
            html = get(page_url).decode("utf-8", "ignore")
        except Exception as e:
            print(f"[FAIL] {gname}: {e}"); continue
        thumbs = re.findall(r'src=["\']([^"\']*thumbnails/[^"\']+\.jpe?g)["\']', html, re.I)
        seen, fulls = set(), []
        for t in thumbs:
            full = urljoin(page_url, t).replace("/thumbnails/", "/images/")
            if full not in seen and "introslide" not in full.lower():
                seen.add(full); fulls.append(full)
        d = os.path.join(ROOT, dest); os.makedirs(d, exist_ok=True)
        ok = fail = 0
        print(f"[{gname}] {len(fulls)} full-size images found -> {dest}")
        for i, u in enumerate(fulls, 1):
            name = os.path.basename(unquote(urlparse(u).path))
            outp = os.path.join(d, name)
            if os.path.exists(outp):
                ok += 1; continue
            try:
                data = get(u)
                img = Image.open(io.BytesIO(data)); w, h = img.size
                if min(w, h) < 200:
                    print(f"   [{i}] STILL small {w}x{h}: {name}"); fail += 1; continue
                open(outp, "wb").write(data); ok += 1
                print(f"   [{i}/{len(fulls)}] {name} {w}x{h}")
                time.sleep(0.2)
            except Exception as e:
                fail += 1; print(f"   [{i}] FAIL {name}: {str(e)[:60]}")
        print(f"[{gname}] done: {ok} ok, {fail} failed")
        grand += ok
    print(f"\nTOTAL full-size downloaded: {grand}")

if __name__ == "__main__":
    main()
