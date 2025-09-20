from pathlib import Path

path = Path('app.py')
text = path.read_text(encoding='utf-8')
marker = '    def handle_total_trips'
idx = text.find(marker)
if idx == -1:
    raise SystemExit('handle_total_trips marker not found')
snippet = '''    def handle_llm_backstop(self, question: str) -> Optional[QueryResponse]:
        """Use OpenAI to draft a simple plan when heuristics are not enough."""
        if not OPENAI_READY or OpenAI is None:
            return None
        df = self.trips.copy()
        day_token = parse_day_token(question)
        daypart = parse_daypart(question)
        df = apply_day_filters(df, day_token, daypart)
        age_range = parse_age_range(question)
        if age_range:
            df = filter_by_age_range(df, age_range)
        threshold = parse_group_threshold(question)
        if threshold:
            df = filter_by_group_threshold(df, threshold)
        timeframe = parse_timeframe(question, self.now)
        df_time, timeframe_label = apply_timeframe(df, timeframe)
        timeframe_note = None
        working_df = df_time if not df_time.empty else df
        if df_time.empty and timeframe_label:
            timeframe_note = f"No trips match {timeframe_label}; using all available data instead."
        if working_df.empty:
            return None
        plan = generate_llm_plan(question, working_df)
        if not plan:
            return None
        filtered_df = apply_plan_filters(working_df, plan.get("filters"))
        if filtered_df.empty:
            return QueryResponse(
                text="The AI plan filtered out every row. Try rephrasing the question or loosening filters.",
                kind="message",
            )

        descriptors: List[str] = []
        if timeframe_label and not timeframe_note:
            descriptors.append(timeframe_label)
        if age_range:
            descriptors.append(f"ages {age_range[0]}-{age_range[1]}")
        if threshold:
            descriptors.append(f"{threshold}+ riders")
        if day_token:
            descriptors.append(day_token.lower())
        if daypart:
            descriptors.append(daypart)
        notes: List[str] = []
        if timeframe_note:
            notes.append(timeframe_note)

        operation = str(plan.get("operation", "")).lower()
        limit = int(plan.get("limit", 10) or 10)
        limit = max(1, min(30, limit))

        def append_details(base: str) -> str:
            detail = f" ({', '.join(descriptors)})" if descriptors else ""
            note_text = (" " + " ".join(notes)) if notes else ""
            return base + detail + note_text

        if operation in ("count", "count_rows", "count_trips"):
            count = int(len(filtered_df))
            text = append_details(f"{count:,} trips match this question")
            sample = format_sample(
                filtered_df,
                ["TripID", "DateTime", "PickupAddress", "DropoffAddress", "PartySize"],
                limit=25,
            )
            return QueryResponse(text=text + ".", kind="count", sample=sample)

        if operation in ("top_values", "top", "value_counts"):
            target_col = resolve_column(filtered_df, plan.get("target_column") or plan.get("group_column"))
            if not target_col:
                return QueryResponse(text="The AI plan referenced a column that does not exist.", kind="message")
            counts = (
                filtered_df[target_col]
                .fillna("(unknown)")
                .astype(str)
                .value_counts()
                .head(limit)
                .reset_index()
            )
            counts.columns = [target_col, "Trips"]
            text = append_details(f"Top {target_col} values")
            chart_sort = DAY_NAMES if target_col in ("DOW", "Day") else None
            return QueryResponse(
                text=text + ".",
                kind="table",
                table=counts,
                chart_type="bar",
                chart_x=target_col,
                chart_y="Trips",
                chart_sort=chart_sort,
                chart_title=text,
            )

        if operation in ("group_by", "distribution", "group_count"):
            group_col = resolve_column(filtered_df, plan.get("group_column") or plan.get("target_column"))
            if not group_col:
                return QueryResponse(text="The AI plan referenced a column that does not exist.", kind="message")
            grouped = (
                filtered_df[group_col]
                .fillna("(unknown)")
                .astype(str)
                .value_counts()
                .sort_index()
                .reset_index()
            )
            grouped.columns = [group_col, "Trips"]
            if group_col in ("DOW", "Day"):
                grouped[group_col] = pd.Categorical(grouped[group_col], categories=DAY_NAMES, ordered=True)
                grouped = grouped.sort_values(group_col)
            text = append_details(f"Trips by {group_col}")
            return QueryResponse(
                text=text + ".",
                kind="table",
                table=grouped.head(limit),
                chart_type="bar",
                chart_x=group_col,
                chart_y="Trips",
                chart_sort=DAY_NAMES if group_col in ("DOW", "Day") else None,
                chart_title=text,
            )

        return None

'''
text = text[:idx] + snippet + text[idx:]
path.write_text(text, encoding='utf-8')
