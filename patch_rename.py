from pathlib import Path

path = Path('app.py')
text = path.read_text(encoding='utf-8')
old = 'def rename_first(df: pd.DataFrame, candidates: List[str], target: str) -> None:\n    for c in candidates:\n        if c in df.columns:\n            df.rename(columns={c: target}, inplace=True)\n            break\n'
new = 'def rename_first(df: pd.DataFrame, candidates: List[str], target: str) -> None:\n    lower_map = {str(col).lower(): col for col in df.columns}\n    for cand in candidates:\n        col = lower_map.get(str(cand).lower())\n        if col:\n            df.rename(columns={col: target}, inplace=True)\n            break\n'
if old not in text:
    raise SystemExit('rename_first block not found')
path.write_text(text.replace(old, new), encoding='utf-8')
