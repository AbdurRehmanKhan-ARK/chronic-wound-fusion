import sys, os, re
from email import policy
from email.parser import BytesParser

def extract(mhtml_path, dest):
    os.makedirs(dest, exist_ok=True)
    with open(mhtml_path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)
    count, skip, seen = 0, 0, set()
    for part in msg.walk():
        ct = part.get_content_type()
        if not ct.startswith("image/") or ct == "image/gif":
            continue
        data = part.get_payload(decode=True)
        if not data or len(data) < 10000:
            skip += 1
            continue
        loc = part.get("Content-Location", "") or part.get_filename() or ""
        base = os.path.basename(loc.split("?")[0].split("#")[0])
        if not re.search(r"\.(jpe?g|png|bmp|webp)$", base, re.I):
            base = "img_%04d.%s" % (count, ct.split("/")[1].split("+")[0])
        base = re.sub(r"[^\w.\-]", "_", base)
        if base in seen:
            root, ext = os.path.splitext(base)
            n = 1
            while ("%s_%d%s" % (root, n, ext)) in seen:
                n += 1
            base = "%s_%d%s" % (root, n, ext)
        seen.add(base)
        with open(os.path.join(dest, base), "wb") as out:
            out.write(data)
        count += 1
    print("%s -> %s | %d images saved, %d skipped" % (os.path.basename(mhtml_path), dest, count, skip))

if __name__ == "__main__":
    extract(sys.argv[1], sys.argv[2])
