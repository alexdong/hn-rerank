.PHONY: run lint deploy

run:
	python main.py

lint:
	ruff check .
	pyright .

deploy:
	railway init -n hn-rerank
	railway up -c
	railway domain
	fh_railway_link
	railway volume add -m /app/data
