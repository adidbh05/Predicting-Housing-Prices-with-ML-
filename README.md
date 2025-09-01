London House Price Prediction (Synthetic)

Commands
- Create venv + install: `make install`
- Generate data: `make data`
- Train model: `make train`
- Cross-validate: `make cv`
- Predict on new listings: `make predict`
- Full pipeline: `make all`

Artifacts
- Training data: `london_house_prices.csv`
- New rows: `new_listings.csv`
- Model: `artifacts/model.joblib`
- Metadata: `artifacts/metadata.json`
- Predictions: `predictions.csv`

Notes
- Data is synthetic but engineered to be plausible with borough effects, square footage, bedrooms/bathrooms, property type, tube distance, year built, and text-like amenities.
