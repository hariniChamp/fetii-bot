from pathlib import Path

path = Path('app.py')
text = path.read_text(encoding='utf-8')
old = '''    def fallback_help(self) -> QueryResponse:\n        txt = (\n            "Try questions like:\n"\n            "- How many groups went to Moody Center last month?\n"\n            "- Top drop-off spots for ages 18-24 on Saturday nights\n"\n            "- When do large groups (6+) typically ride downtown?"\n        )\n        return QueryResponse(text=txt, kind="message")\n'''
new = '''    def fallback_help(self) -> QueryResponse:\n        txt = ("Try questions like:\n"
               "- How many groups went to Moody Center last month?\n"
               "- Top drop-off spots for ages 18-24 on Saturday nights\n"
               "- When do large groups (6+) typically ride downtown?")\n        return QueryResponse(text=txt, kind="message")\n'''
if old not in text:
    raise SystemExit('fallback block not found')
path.write_text(text.replace(old, new), encoding='utf-8')
