from pathlib import Path
import textwrap

path = Path('app.py')
text = path.read_text(encoding='utf-8')
start = text.find('class QueryEngine:')
end = text.find('# -------------------- UI --------------------')
if start == -1 or end == -1:
    raise SystemExit('Unable to locate QueryEngine block')

new_block = textwrap.dedent('''
class QueryEngine:
    def __init__(self, frames: Dict[str, pd.DataFrame], scraped_text: str):
        self.frames = frames
        self.trips = frames["trips"]
        self.scraped = scraped_text
        self.now = dt.datetime.now()
        self.handlers = [
            self.handle_count_to_place,
            self.handle_top_destinations,
            self.handle_large_group_hours,
            self.handle_trips_by_day,
            self.handle_trips_by_hour,
            self.handle_total_trips,
        ]

    def _apply_filters(self, question: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        df = self.trips.copy()
        meta: Dict[str, Any] = {"descriptors": []}

        day_token = parse_day_token(question)
        daypart = parse_daypart(question)
        df = apply_day_filters(df, day_token, daypart)
        meta["day_token"] = day_token
        meta["daypart"] = daypart
        if day_token:
            meta["descriptors"].append(day_token)
        if daypart:
            meta["descriptors"].append(daypart)

        age_range = parse_age_range(question)
        meta["age_range"] = age_range
        if age_range:
            df = filter_by_age_range(df, age_range)
            meta["descriptors"].append(f"ages {age_range[0]}-{age_range[1]}")

        threshold = parse_group_threshold(question)
        meta["threshold"] = threshold
        if threshold:
            df = filter_by_group_threshold(df, threshold)
            meta["descriptors"].append(f"{threshold}+ riders")

        meta["pre_timeframe_df"] = df.copy()

        timeframe = parse_timeframe(question, self.now)
        meta["timeframe"] = timeframe
        df, timeframe_label = apply_timeframe_filter(df, timeframe)
        meta["timeframe_label_requested"] = timeframe_label
        meta["timeframe_label"] = timeframe_label
        if timeframe_label:
            meta["descriptors"].append(timeframe_label)

        return df, meta

    def _detail_suffix(self, meta: Dict[str, Any]) -> str:
        desc = [d for d in meta.get("descriptors", []) if d]
        return f" ({', '.join(desc)})" if desc else ""

    def _fallback_timeframe(self, df: pd.DataFrame, base_df: pd.DataFrame, meta: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
        timeframe = meta.get("timeframe")
        if not timeframe or not df.empty:
            return df, ""
        if base_df.empty or "DateTime" not in base_df.columns:
            return df, ""
        dates = pd.to_datetime(base_df["DateTime"], errors="coerce").dropna()
        if dates.empty:
            return df, ""
        latest = dates.max()
        start, end = month_bounds(latest.year, latest.month)
        fallback_df = base_df[(base_df["DateTime"] >= start) & (base_df["DateTime"] < end)]
        if fallback_df.empty:
            return df, ""
        original_label = meta.get("timeframe_label")
        month_label = f"{latest.strftime('%B %Y')} (latest available)"
        descriptors = meta.get("descriptors", [])
        if original_label in descriptors:
            descriptors.remove(original_label)
        descriptors.append(month_label)
        meta["descriptors"] = descriptors
        meta["timeframe_label"] = month_label
        note = f" No data for {original_label}; showing {month_label} instead." if original_label else f" Showing {month_label}."
        return fallback_df, note

    def handle_count_to_place(self, question: str) -> Optional[QueryResponse]:
        match = re.search(
            r"how many (?:groups|rides|trips)[^?]*?(?:to|went to|headed to|drop(?:ped)? off at|at)\s+([A-Za-z0-9 '&\.-]+)",
            question,
            re.I,
        )
        if not match:
            return None
        place = match.group(1).strip()
        df, meta = self._apply_filters(question)
        if df.empty:
            return None

        dest_field = "DropoffAddress"
        simple_field = "DropoffSimple"
        label = "drop-off"
        lowered = question.lower()
        if "pickup" in lowered or "pick up" in lowered:
            dest_field = "PickupAddress"
            simple_field = "PickupSimple"
            label = "pickup"

        base_df = meta.get("pre_timeframe_df", df)
        base_mask = base_df[dest_field].astype(str).str.contains(place, case=False, na=False)
        if simple_field in base_df.columns:
            base_mask |= base_df[simple_field].astype(str).str.contains(place, case=False, na=False)
        if place.lower() == "downtown":
            base_filtered = filter_destination_keyword(base_df, "downtown", dest_field)
        else:
            base_filtered = base_df[base_mask]
        if base_filtered.empty:
            return None

        filtered = base_filtered
        if meta.get("timeframe"):
            filtered, note = self._fallback_timeframe(df[base_mask] if place.lower() != "downtown" else filter_destination_keyword(df, "downtown", dest_field), base_filtered, meta)
        else:
            note = ""
        if filtered.empty:
            return None

        count = int(len(filtered))
        text = f"{count:,} trips with {label} at {place}{self._detail_suffix(meta)}.{note}"
        text = polish_text(question, text, self.scraped)
        sample = format_sample(filtered, ["TripID", "DateTime", dest_field, "PickupAddress", "PartySize", "DominantAgeBucket"])
        return QueryResponse(text=text, kind="count", sample=sample)

    def handle_top_destinations(self, question: str) -> Optional[QueryResponse]:
        lowered = question.lower()
        if "top" not in lowered or not any(k in lowered for k in ["drop", "destination", "pickup", "spot"]):
            return None
        df, meta = self._apply_filters(question)
        if df.empty:
            return None

        dest_field = "DropoffSimple"
        label = "Drop-off"
        if "pickup" in lowered or "pick up" in lowered:
            dest_field = "PickupSimple"
            label = "Pickup"

        df_base = df.copy()
        if dest_field not in df_base.columns:
            return None

        counts = df_base[dest_field].fillna("(unknown)").astype(str).value_counts().head(10).reset_index()
        if counts.empty:
            base_df = meta.get("pre_timeframe_df", df_base)
            counts = base_df[dest_field].fillna("(unknown)").astype(str).value_counts().head(10).reset_index()
            note = " Showing latest available data." if not counts.empty else ""
        else:
            note = ""
        if counts.empty:
            return None
        counts.columns = [label, "Trips"]

        text = f"Top {label.lower()} destinations{self._detail_suffix(meta)}.{note}"
        text = polish_text(question, text, self.scraped)
        return QueryResponse(
            text=text,
            kind="table",
            table=counts,
            chart_type="bar",
            chart_x=label,
            chart_y="Trips",
            chart_sort=None,
            chart_title=text,
        )

    def handle_large_group_hours(self, question: str) -> Optional[QueryResponse]:
        lowered = question.lower()
        if not any(k in lowered for k in ["when do", "what time", "typical hours", "peak hours", "usually"]):
            return None
        df, meta = self._apply_filters(question)
        threshold = meta.get("threshold") or 6
        if meta.get("threshold") is None:
            df = filter_by_group_threshold(df, threshold)
            meta["descriptors"].append(f"{threshold}+ riders")
        if "downtown" in lowered:
            df = filter_destination_keyword(df, "downtown", "DropoffAddress")
        if df.empty or "Hour" not in df.columns:
            base_df = meta.get("pre_timeframe_df", df)
            df = filter_destination_keyword(base_df, "downtown", "DropoffAddress") if "downtown" in lowered else base_df
            if meta.get("timeframe"):
                df, note = self._fallback_timeframe(pd.DataFrame(), df, meta)
            else:
                note = ""
        else:
            note = ""
        if df.empty or "Hour" not in df.columns:
            return None

        counts = df["Hour"].dropna().astype(int).value_counts().sort_index()
        if counts.empty:
            return None
        hist = pd.DataFrame({"Hour": counts.index, "Trips": counts.values})
        top_hours = counts.sort_values(ascending=False).head(3).index.tolist()
        hour_label = ", ".join(f"{h:02d}:00" for h in top_hours)
        text = f"Peak hours for groups of {threshold}+ riders{self._detail_suffix(meta)}: {hour_label}.{note}"
        text = polish_text(question, text, self.scraped)
        return QueryResponse(
            text=text,
            kind="table",
            table=hist,
            chart_type="bar",
            chart_x="Hour",
            chart_y="Trips",
            chart_title="Trips by hour",
        )

    def handle_trips_by_day(self, question: str) -> Optional[QueryResponse]:
        lowered = question.lower()
        if not any(k in lowered for k in ["busiest day", "day of week", "rides by day", "which day"]):
            return None
        df, meta = self._apply_filters(question)
        if df.empty or "DOW" not in df.columns:
            base_df = meta.get("pre_timeframe_df", df)
            df, note = self._fallback_timeframe(pd.DataFrame(), base_df, meta)
            if df.empty or "DOW" not in df.columns:
                return None
        else:
            note = ""
        counts = df["DOW"].value_counts().reindex(DAY_NAMES).fillna(0).astype(int).reset_index()
        counts.columns = ["Day", "Trips"]
        text = f"Trips by day of week{self._detail_suffix(meta)}.{note}"
        text = polish_text(question, text, self.scraped)
        return QueryResponse(
            text=text,
            kind="table",
            table=counts,
            chart_type="bar",
            chart_x="Day",
            chart_y="Trips",
            chart_sort=DAY_NAMES,
            chart_title="Trips by day of week",
        )

    def handle_trips_by_hour(self, question: str) -> Optional[QueryResponse]:
        lowered = question.lower()
        if not any(k in lowered for k in ["by hour", "hourly", "hour of day", "time of day"]):
            return None
        df, meta = self._apply_filters(question)
        if df.empty or "Hour" not in df.columns:
            base_df = meta.get("pre_timeframe_df", df)
            df, note = self._fallback_timeframe(pd.DataFrame(), base_df, meta)
            if df.empty or "Hour" not in df.columns:
                return None
        else:
            note = ""
        counts = df["Hour"].dropna().astype(int).value_counts().sort_index()
        if counts.empty:
            return None
        table = pd.DataFrame({"Hour": counts.index, "Trips": counts.values})
        text = f"Trips by hour{self._detail_suffix(meta)}.{note}"
        text = polish_text(question, text, self.scraped)
        return QueryResponse(
            text=text,
            kind="table",
            table=table,
            chart_type="bar",
            chart_x="Hour",
            chart_y="Trips",
            chart_title="Trips by hour",
        )

    def handle_total_trips(self, question: str) -> Optional[QueryResponse]:
        lowered = question.lower()
        if "how many" not in lowered or not any(token in lowered for token in ["trip", "ride", "group"]):
            return None
        df, meta = self._apply_filters(question)
        if df.empty:
            base_df = meta.get("pre_timeframe_df", df)
            df, note = self._fallback_timeframe(df, base_df, meta)
            if df.empty:
                return None
        else:
            note = ""
        count = int(len(df))
        text = f"{count:,} trips match this question{self._detail_suffix(meta)}.{note}"
        text = polish_text(question, text, self.scraped)
        sample = format_sample(df, ["TripID", "DateTime", "PickupAddress", "DropoffAddress", "PartySize", "DominantAgeBucket"])
        return QueryResponse(text=text, kind="count", sample=sample)

    def ai_sql(self, question: str) -> Optional[QueryResponse]:
        if not (OPENAI_READY and DUCK_READY):
            return None
        sql = build_llm_sql(question, self.frames, self.now)
        if not sql:
            return None
        try:
            df = run_duckdb_query(self.frames, sql)
        except Exception as e:
            return QueryResponse(text=f"SQL error: {e}", kind="message", debug_sql=sql)
        if df.shape == (1, 1) and pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
            val = df.iloc[0, 0]
            if isinstance(val, (int, np.integer)) or (isinstance(val, float) and float(val).is_integer()):
                val = int(val)
            text = f"Answer: {val:,}" if isinstance(val, (int, np.integer)) else f"Answer: {val}"
            text = polish_text(question, text, self.scraped)
            return QueryResponse(text=text, kind="count", debug_sql=sql)
        chart_type, cx, cy, csort = infer_chart(df)
        text = polish_text(question, "Here’s what I found from the dataset.", self.scraped)
        return QueryResponse(text=text, kind="table", table=df, chart_type=chart_type, chart_x=cx, chart_y=cy, chart_sort=csort, debug_sql=sql)

    def fallback_help(self) -> QueryResponse:
        txt = """Try questions like:\n- How many groups went to Moody Center last month?\n- Top drop-off spots for ages 18-24 on Saturday nights\n- When do large groups (6+) typically ride downtown?"""
        return QueryResponse(text=txt, kind="message")

    def answer(self, question: str) -> QueryResponse:
        cleaned = question.strip()
        if not cleaned:
            return self.fallback_help()
        for handler in self.handlers:
            resp = handler(cleaned)
            if resp:
                return resp
        ai_resp = self.ai_sql(cleaned)
        if ai_resp:
            return ai_resp
        return self.fallback_help()
''')

path.write_text(text[:start] + new_block + text[end:], encoding='utf-8')
