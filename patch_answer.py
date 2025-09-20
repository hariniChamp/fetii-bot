from pathlib import Path

path = Path('app.py')
text = path.read_text(encoding='utf-8')
old = '''    def answer(self, question: str) -> QueryResponse:\n        cleaned = question.strip()\n        if not cleaned:\n            return self.fallback_help()\n        ai_resp = self.ai_sql(cleaned)\n        if ai_resp:\n            return ai_resp\n        for handler in self.handlers:\n            resp = handler(cleaned)\n            if resp:\n                return resp\n        return self.fallback_help()\n'''
new = '''    def answer(self, question: str) -> QueryResponse:\n        cleaned = question.strip()\n        if not cleaned:\n            return self.fallback_help()\n        ai_resp = self.ai_sql(cleaned)\n        if ai_resp and ai_resp.kind != "message":\n            return ai_resp\n        for handler in self.handlers:\n            resp = handler(cleaned)\n            if resp:\n                return resp\n        return ai_resp if ai_resp else self.fallback_help()\n'''
if old not in text:
    raise SystemExit('answer block not found')
path.write_text(text.replace(old, new), encoding='utf-8')
