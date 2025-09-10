### Reaction Predictor – Code Architecture Overview

This document explains how `reaction_predictor.py` is structured, how its parts interact, and what the main methods do. It is aimed at developers or technically curious readers.

### Purpose and design
- Provide local, offline reaction product prediction using a pre-downloaded Hugging Face T5 model.
- Provide retrosynthesis route planning using AiZynthFinder with local ONNX policies/templates and ZINC stock.
- Offer optional natural-language conveniences (name ↔ SMILES) via OpenAI (if `.env` supplies a token).
- Integrate with PubChem for lookup by SMILES or formula and for structure images.
- Persist intermediate and final results to JSON files in the project root, minimizing console verbosity.

### External dependencies and resources
- Transformers: `AutoTokenizer`, `AutoModelForSeq2SeqLM` for reaction inference (local only; `TRANSFORMERS_OFFLINE=1`).
- AiZynthFinder: route search, expansion, filter, and stock selection driven by `config.yml`.
- RDKit: SMILES parsing, sanitization, and molecular formula calculations.
- Requests: PubChem PUG-REST calls; image downloads.
- dotenv: loads `.env` for `OPENAI_API_TOKEN` (optional).
- OpenAI: optional name ↔ SMILES conversions.
- Local assets:
  - Model: `models/reactiont5v2-forward` (tokenizer + model files).
  - AiZynthFinder: `config.yml`, ONNX policy files, `zinc_stock.hdf5`, and templates.

### Initialization flow
- Resolves `SCRIPT_DIR` for robust relative paths and sets `TRANSFORMERS_OFFLINE=1`.
- Loads tokenizer and model from `models/reactiont5v2-forward` using `local_files_only=True`.
- Initializes AiZynthFinder using absolute `config.yml`; selects policies: `filter=uspto`, `stock=zinc`, `expansion=uspto`.
- Reads optional `OPENAI_API_TOKEN` from `.env` to initialize `openai_client`.

### High-level components
1) Reaction prediction (local T5)
- Method: `predict_reaction(reactant_smiles, reagent_smiles, whole_input)`
- Formats input as `REACTANT:<...>REAGENT:<...>`; runs model.generate; decodes to product SMILES.

2) Retrosynthesis planning (AiZynthFinder)
- Methods:
  - `find_pathways_to_smiles(target_smiles)` → builds routes, then `flatten_all_routes` to normalize.
  - `get_route_precursors(flat_route)` → final in-stock building blocks.
  - `smiles_list_similarity(a, b)` → Jaccard similarity for precursor matching.
  - `find_best_matching_route(flat_routes, input_smiles)` → picks routes whose precursors best match provided input(s), reconstructs reaction steps from the route structure, reverses for forward readability, cleans metadata, and renumbers.
  - `process_any_route(route_data)` → validates and reconstructs reaction steps.
  - `reconstruct_reaction_from_route(route, step)` → builds `reactant >> products` from graph.
  - `validate_reaction_smiles(...)` and parsing helpers to handle corrupted/atom-mapped strings.

3) Natural language conversion (optional)
- Methods:
  - `natural_language_to_smiles(description)`
  - `smiles_to_natural_language(smiles)`
  - `build_reaction_input_from_json(reaction_json)`
- Require `OPENAI_API_TOKEN`; results stored as JSON (also error JSONs if failures occur).

4) PubChem integration
- Methods:
  - `search_pubchem_by_smiles(smiles)`
  - `search_pubchem_by_formula(molecular_formula, max_results)`
  - `predict_reaction_from_formulae(reactant_formulae, reagent_formulae, ...)` combines PubChem lookup → reaction prediction → PubChem breakdown of product.
  - `_break_down_product_smiles(product_smiles)` simple splitter on dots.
- Produces structured JSON outputs and error JSONs.

5) Imaging and diagrams
- Methods:
  - `download_chemical_image(chemical_name, ...)` (by name)
  - `download_chemical_image_by_formula(chemical_formula, ...)` (by formula)
  - `create_reaction_diagram(reactants, reagents, products, ...)` → composes individual PNGs into a single diagram with + and → symbols.
- Saves `.png` files and JSON metadata.

6) Utilities
- `single_formula_to_smiles(formula)` → PubChem first; falls back to Cactus NCI.
- `extract_formulas_from_route(route)` → formulas for molecules and parsed reaction steps.
- `parse_reaction_smiles(...)` and `parse_reaction_manually(...)` → robust reaction parsing for route steps.
- `check_if_input_smiles_is_in_route(...)` → trims routes when the input is internal (currently not used by default).

### Data flow (textual)
- Forward prediction:
  - Inputs (reactants/reagents SMILES) → format string → tokenizer → model → predicted product SMILES → returned string.
- Retrosynthesis:
  - Target SMILES → AiZynthFinder `tree_search` → routes → flatten → for each route: reconstruct reaction steps from graph → reverse to forward order → clean → renumber → JSON.
- Formula-driven reaction:
  - Formulae → PubChem SMILES lookup → predict product → split product SMILES → per part: PubChem info → JSON summary.
- Imaging:
  - Names/formulae → PubChem PNG endpoints → files → if diagram: compose into one image.

### Files written (by convention)
- Core outputs:
  - Route lists: `routes_result_*.json` (when used) and per-route JSONs like `<product_smiles>_0.json`.
  - Prediction/translation: `nl_to_smiles_result.json`, `smiles_to_nl_result.json`, `reaction_input_build_result.json` (+ matching `*_error.json`).
  - PubChem: `pubchem_search_result.json`, `pubchem_formula_search_result.json` (+ error variants), `reaction_from_formulae_result.json`.
  - Imaging: molecule `.png` files and `reaction_diagram_result.json` (+ error variants).

### Error handling and logging
- Network operations guarded by try/except; failures produce `*_error.json` alongside console messages.
- OpenAI-dependent features degrade gracefully if the token is absent; other features remain available.
- Route validation attempts to sanitize or reconstruct when reaction SMILES are malformed.

### Key method reference (selected)
- Prediction:
  - `predict_reaction(reactant_smiles=None, reagent_smiles=None, whole_input=None) -> str`
- Retrosynthesis / routes:
  - `find_pathways_to_smiles(target_smiles) -> list[flat_route]`
  - `find_best_matching_route(flat_routes, input_smiles) -> (list[routes_with_scores], best_score)`
  - `process_any_route(route_data) -> dict[int, step_info]`
- Natural language:
  - `natural_language_to_smiles(description) -> Optional[str]`
  - `smiles_to_natural_language(smiles) -> Optional[str]`
  - `build_reaction_input_from_json(reaction_json) -> Optional[str]`
- PubChem:
  - `search_pubchem_by_smiles(smiles) -> Optional[dict]`
  - `search_pubchem_by_formula(molecular_formula, max_results=5) -> Optional[list[tuple]]`
  - `predict_reaction_from_formulae(...) -> Optional[dict]`
- Imaging:
  - `download_chemical_image(name, ...) -> Optional[dict]`
  - `download_chemical_image_by_formula(formula, ...) -> Optional[str]`
  - `create_reaction_diagram(...) -> Optional[str]`

### Extensibility notes
- New predictors: Wrap additional models under a similar `predict_reaction` interface; keep local/offline first.
- Alternative LLMs: Natural-language helpers can be abstracted behind a provider-agnostic interface if needed.
- Route filters/stocks: Adjust via AiZynthFinder configuration/policy selection.
- Parsing robustness: Extend manual parsing to cover more edge cases or use RDKit reaction objects if feasible.

### Developer quick start
- Activate venv:
```bash
cd "/Users/alexiskirke/Dropbox/Contracting/upwork_and_cc/chemistry_deploy"
source venv/bin/activate
```
- Optional: add `.env` with `OPENAI_API_TOKEN=...`.
- Run the sample script:
```bash
python reaction_predictor.py
```
- Inspect JSON/PNG outputs in the project root.
