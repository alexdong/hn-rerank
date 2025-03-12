.PHONY: run deploy

run:
	uv run main.py

deploy:
	railway up -c
	railway domain
