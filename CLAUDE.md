# HN-RERANK PROJECT GUIDELINES

## Commands
- Lint: `ruff check .`
- Type check: `pyright`
- Run: `python main.py`
- Test single file: `python file.py` (each file has `__main__` block for testing)
- Deploy to Railway:
  ```bash
  railway init -n hn-rerank
  railway up -c
  railway domain
  fh_railway_link
  railway volume add -m /app/data
  ```

## Code Style
- Python 3.12 required
- Type hints required for all functions/classes
- No try/except blocks - let program crash
- No comments unless necessary
- No database - store data in memory
- Small .py files with `__main__` testing blocks

### Style Details
- One empty line between concept blocks
- Two empty lines between functions/classes
- Compact code preferred
- Use assertions liberally for validation
- Print debug with `[LEVEL] {message}` format
- Comprehensive type hints required (both mypy and pyright must pass)
- Choose names that minimize understanding effort