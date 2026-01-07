# langgraph-sql-agent — README

## Summary

Small utility to download a database and run a SQL agent that queries it. This README is concise and focused on quick setup and usage.

## Files

- requirements.txt — Python dependencies. Install with pip.
- download_db.py — downloads and prepares the database file(s).
- sql_agent.py — runs the SQL agent against the prepared database.

## Prerequisites

- Python 3.8+
- Git (optional)
- Internet access to download remote DB (if applicable)

## Install

1. Create virtual environment:
   python -m venv .venv
   source .venv/bin/activate # or .venv\Scripts\activate on Windows
2. Install dependencies:
   pip install -r requirements.txt

## Usage

- Prepare/download database:
  python download_db.py

  - Optional flags: provide --url, --output, or other flags as implemented.

- Run SQL agent:
  python sql_agent.py
  - Typical options: --db /path/to/db, --query "SELECT ..." or interactive mode.

Replace flag names above with the actual options from each script (see scripts for exact CLI).

## Configuration

- Edit environment variables or a config file if scripts support them.
- Common config: DB path, remote DB URL, API keys for any LLM integrations.

## Troubleshooting

- Missing packages: rerun pip install -r requirements.txt
- DB not found: confirm download_db.py succeeded and path passed to sql_agent.py is correct.
- Permission issues: check file system permissions.

## Contributing

- Open a PR with minimal, focused changes.
- Add tests for new behavior where applicable.

## Environment (.env) and Gemini API key

### .env file

Create a .env file at the project root to hold secrets and per-environment values. Example contents:
GEMINI_API_KEY=your_api_key_here
GEMINI_SERVICE_ACCOUNT_FILE=/path/to/service-account.json # optional, prefer for server use
DB_PATH=./data/mydb.sqlite

Add .env to .gitignore to avoid committing secrets.

### Reading .env in Python

Install python-dotenv if not already in requirements:
pip install python-dotenv

Example snippet to load values (use in sql_agent.py or a shared config module):
from dotenv import load_dotenv
import os

load_dotenv() # reads .env in project root
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERVICE_ACCOUNT_FILE = os.getenv("GEMINI_SERVICE_ACCOUNT_FILE")

# If using a Google service account JSON file, point Google SDKs at it:

if SERVICE_ACCOUNT_FILE:
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_FILE

Use GEMINI_API_KEY (or the Google auth flow) when initializing your LLM client in code.

### Obtaining a new Gemini API key

1. Create a Google Cloud project (or use an existing one).
2. Enable the Generative AI / Vertex AI API for the project.
3. For server-to-server use (recommended):
   - Create a service account with the required roles (Vertex AI User or minimal roles your client needs).
   - Create and download a JSON key. Store the JSON securely and set GEMINI_SERVICE_ACCOUNT_FILE to its path in .env.
4. For quick testing (if supported):
   - Create an API key in the Cloud Console and set GEMINI_API_KEY in .env.
5. Optionally restrict the key (IP, referrer) and rotate regularly.

Refer to Google Cloud Console documentation for up-to-date UI steps and required IAM roles.

### Configuring the project to use the key

- Prefer service-account JSON for production; set GOOGLE_APPLICATION_CREDENTIALS via SERVICE_ACCOUNT_FILE as shown above.
- For direct API-key usage, read GEMINI_API_KEY from environment and pass it to your LLM client library per its configuration.
- Example: when initializing your client, do something like client = GeminiClient(api_key=GEMINI_API_KEY) or rely on Google libraries using GOOGLE_APPLICATION_CREDENTIALS.

### Security tips

- Never commit keys or service-account JSON to version control.
- Add .env and any key files to .gitignore.
- Use least-privilege IAM roles, restrict API keys, and rotate credentials periodically.
- Store production secrets in a secrets manager when possible (Secret Manager, Vault, etc.).
