import app

try:
    sheets = app.load_excel(None)
    frames = app.build_master_frames(sheets)
    trips = frames['trips']
    print('rows', len(trips))
    print(trips[['TripID','DropoffAddress','PartySize']].head())
except Exception as exc:
    import traceback
    traceback.print_exc()
