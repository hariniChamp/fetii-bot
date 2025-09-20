# Deployment Notes

1. **Create environment**
   `ash
   python -m venv .venv
   .venv\\Scripts\\activate
   pip install -r requirements.txt
   `
   > If equirements.txt is missing, install: streamlit pandas numpy altair openpyxl requests beautifulsoup4.

2. **Run locally**
   `ash
   streamlit run app.py
   `
   Upload FetiiAI_Data_Austin.xlsx when prompted or place it next to pp.py.

3. **Streamlit Cloud / Hugging Face Spaces**
   - Push this folder to GitHub.
   - Add OPENAI_API_KEY as a secret if you want polished answers.
   - Ensure FetiiAI_Data_Austin.xlsx is in the repo or stored in cloud storage.

4. **Optional extras**
   - Set ENABLE_SCRAPE=1 to prefetch fetii.com copy (requires equests + eautifulsoup4).
   - Use Streamlit st.secrets for API keys in hosted environments.
