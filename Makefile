.PHONY: venv install data train cv predict all clean

PY ?= python3
VENV := .venv
PIP := $(VENV)/bin/pip
PYBIN := $(VENV)/bin/python

venv:
	$(PY) -m venv $(VENV)

install: venv
	$(PYBIN) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

data: install
	$(PYBIN) app.py generate --rows 6000 --output london_house_prices.csv --new-output new_listings.csv

train: data
	$(PYBIN) app.py train --data london_house_prices.csv --target price --model artifacts/model.joblib

cv: data
	$(PYBIN) app.py train --data london_house_prices.csv --target price --cv 5

predict: train
	$(PYBIN) app.py predict --model artifacts/model.joblib --input new_listings.csv --output predictions.csv

all: predict

clean:
	rm -rf $(VENV) artifacts london_house_prices.csv new_listings.csv predictions.csv
