from pathlib import Path
import re

path = Path('app.py')
text = path.read_text(encoding='utf-8')
pattern = r"raw = response\.choices\[0\]\.message\.content\.strip\(\).*?plan = json\.loads\(raw\)"
replacement = "raw = response.choices[0].message.content.strip()\n        if raw.startswith(\"`\"):\n            raw = raw[3:]\n            if raw.endswith(\"`\"):\n                raw = raw[:-3]\n            raw = raw.strip()\n            if raw.lower().startswith(\"json\"):\n                raw = raw[4:].strip()\n        plan = json.loads(raw)"
new_text, count = re.subn(pattern, replacement, text, flags=re.S)
if count == 0:
    raise SystemExit('pattern not replaced')
path.write_text(new_text, encoding='utf-8')
