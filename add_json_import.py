from pathlib import Path

path = Path('app.py')
text = path.read_text(encoding='utf-8')
old = "import os, re, io, math, datetime as dt\nfrom typing import Optional, Tuple, Dict, Any, List\n\nimport pandas as pd"
new = "import os, re, io, math, datetime as dt\nfrom typing import Optional, Tuple, Dict, Any, List\n\nimport json\n\nimport pandas as pd"
if old not in text:
    raise SystemExit('import block not found')
path.write_text(text.replace(old, new), encoding='utf-8')
