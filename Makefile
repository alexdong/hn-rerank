.PHONY: run deploy

run:
	uv run main.py

deploy:
	railway up
	railway domain
