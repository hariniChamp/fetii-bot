from pathlib import Path

path = Path('app.py')
text = path.read_text(encoding='utf-8')
needle = '    def fallback_help(self) -> QueryResponse:'
start = text.find(needle)
if start == -1:
    raise SystemExit('fallback not found')
end = text.find('\n\n    def answer(', start)
if end == -1:
    raise SystemExit('answer marker not found')
replacement = '''    def fallback_help(self) -> QueryResponse:\n        txt = """Try questions like:\n- How many groups went to Moody Center last month?\n- Top drop-off spots for ages 18-24 on Saturday nights\n- When do large groups (6+) typically ride downtown?"""\n        return QueryResponse(text=txt, kind="message")\n\n'''
text = text[:start] + replacement + text[end:]
path.write_text(text, encoding='utf-8')
