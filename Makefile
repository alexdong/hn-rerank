.PHONY: run deploy

run:
	uv run main.py

deploy:
	railway init -n hn-rerank
	railway up -c
	railway domain
	fh_railway_link
	railway volume add -m /app/data
