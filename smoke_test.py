import app

frames = app.build_master_frames(app.load_excel(None))
engine = app.QueryEngine(frames, "")
questions = [
    "How many groups went to Moody Center last month?",
    "Top drop-off spots for 18-24 on Saturday nights",
    "When do large groups (6+) ride downtown?",
    "Trips by day of week",
    "How many trips overall?",
    "How many groups went to the Castilian last month?",
]
for q in questions:
    res = engine.answer(q)
    print('\nQ:', q)
    print('Kind:', res.kind)
    print('Text:', res.text)
    if res.table is not None:
        print(res.table.head())
    if res.sample is not None:
        print(res.sample.head())
