from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import openai
import os
import json
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import requests
from urllib.parse import quote
from PIL import Image, ImageDraw, ImageFont
from aizynthfinder.aizynthfinder import AiZynthFinder
from collections import Counter
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import re
# get deepcopy
from copy import deepcopy
import pubchempy as pcp
import ssl
import urllib.request
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from fastapi.middleware.cors import CORSMiddleware


class RetrosynthesisRequest(BaseModel):
    target_smiles: str
    max_routes: Optional[int] = 10

class RetrosynthesisResponse(BaseModel):
    success: bool
    target_smiles: str
    total_routes_found: int
    routes_returned: int
    routes: List[dict]
    error: Optional[str] = None

# Create FastAPI app
app = FastAPI(
    title="AlphaReact Retrosynthesis API",
    description="API for chemical retrosynthesis analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "https://www.sunmukai.com"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

predictor = None

@app.on_event("startup")
async def startup_event():
    """Initialize the predictor when the app starts"""
    global predictor
    try:
        predictor = ReactionPredictor()
        print("✅ ReactionPredictor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize ReactionPredictor: {e}")
        raise

@app.get("/")
async def root():
    return {"message": "AlphaReact Retrosynthesis API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "predictor_ready": predictor is not None}

@app.post("/find_routes", response_model=RetrosynthesisResponse)
async def find_retrosynthesis_routes(request: RetrosynthesisRequest):
    """
    Find retrosynthetic routes for a given target molecule
    
    Args:
        request: Contains target_smiles and optional max_routes
    
    Returns:
        JSON response with retrosynthesis routes
    """
    if predictor is None:
        raise HTTPException(status_code=500, detail="Predictor not initialized")
    
    try:
        print(f"Finding retrosynthetic routes for: {request.target_smiles}")
        
        # Find retrosynthetic pathways
        flat_routes = predictor.find_pathways_to_smiles(request.target_smiles)
        
        if not flat_routes:
            return RetrosynthesisResponse(
                success=False,
                target_smiles=request.target_smiles,
                total_routes_found=0,
                routes_returned=0,
                routes=[],
                error="No retrosynthetic routes found. This could be because the molecule is too complex, AiZynthFinder models are not properly configured, or the SMILES string is invalid."
            )
        
        print(f"Found {len(flat_routes)} potential retrosynthetic routes")
        
        # Limit routes to max_routes
        routes_to_process = flat_routes[:request.max_routes]
        processed_routes = []
        
        # Process each route
        for idx, route in enumerate(routes_to_process):
            try:
                # Process the route to get readable format
                processed_route = predictor.process_any_route(route)
                
                # Create the final route structure
                final_route = {
                    "route_id": idx + 1,
                    "total_steps": len(processed_route),
                    "target_molecule": request.target_smiles,
                    "steps": []
                }
                
                # Convert processed route to API format
                for step_idx, step_data in processed_route.items():
                    step_info = {
                        "step": step_idx,
                        "data": step_data
                    }
                    
                    # Determine step type based on data content
                    if "reconstructed_reaction_smiles" in step_data:
                        step_info["type"] = "reaction"
                    elif "smiles" in step_data and "formula" in step_data:
                        step_info["type"] = "molecule"
                    else:
                        step_info["type"] = "unknown"
                    
                    final_route["steps"].append(step_info)
                
                processed_routes.append(final_route)
                
            except Exception as e:
                print(f"Error processing route {idx}: {e}")
                continue
        
        return RetrosynthesisResponse(
            success=True,
            target_smiles=request.target_smiles,
            total_routes_found=len(flat_routes),
            routes_returned=len(processed_routes),
            routes=processed_routes
        )
        
    except Exception as e:
        print(f"Error in find_retrosynthesis_routes: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

# Ensure paths work whether CWD is the project dir or its parent
SCRIPT_DIR = Path(__file__).resolve().parent
# Local path to the HF model (must be pre-downloaded)
# Check multiple possible locations for the model
MODEL_LOCATIONS = [
    SCRIPT_DIR / "models" / "reactiont5v2-forward",  # Original location
    Path(r"c:\Users\786sy\Downloads\reaction_large_files\models\reactiont5v2-forward"),  # Downloads location
    Path.home() / "Downloads" / "reaction_large_files" / "models" / "reactiont5v2-forward"  # Generic downloads
]

MODEL_DIR = None
for location in MODEL_LOCATIONS:
    if location.exists():
        MODEL_DIR = location
        break

if MODEL_DIR is None:
    # Fallback to original location for error message
    MODEL_DIR = SCRIPT_DIR / "models" / "reactiont5v2-forward"
# If config.yml is not in the current working directory but exists next to this file,
# switch CWD to the script directory so relative paths work consistently
try:
    if not (Path.cwd() / "config.yml").exists() and (SCRIPT_DIR / "config.yml").exists():
        os.chdir(SCRIPT_DIR)
except Exception:
    # If changing directory fails, continue; absolute paths below will still work
    pass

# Bypass SSL verification (NOT recommended for production)
#ssl._create_default_https_context = ssl._create_unverified_context

# Load environment variables from .env file located in the project directory
load_dotenv(dotenv_path=SCRIPT_DIR / ".env")

class ReactionPredictor:
    """
    A class to predict chemical reaction products, convert between natural language
    and SMILES notation, and build reaction inputs from structured data.

    This class uses a pre-trained T5 model ('sagawa/ReactionT5v2-forward') for
    reaction prediction and OpenAI's GPT-4.1 model for natural language processing
    tasks.

    To use the natural language features, you must have a .env file in the root
    directory containing your OpenAI API key:
        OPENAI_API_TOKEN='your-api-key'

    The methods that call the OpenAI API will also write their results to JSON files
    in the root directory (e.g., 'nl_to_smiles_result.json', 
    'smiles_to_nl_result.json', 'reaction_input_build_result.json').

    Example usage:
        # Initialize the predictor
        predictor = ReactionPredictor()

        # Predict a reaction from SMILES
        product_smiles = predictor.predict_reaction(
            reactant_smiles='CCO',
            reagent_smiles='[H]Cl'
        )
        print(f"Predicted product: {product_smiles}")

        # Convert natural language to SMILES
        ethanol_smiles = predictor.natural_language_to_smiles("ethanol")
        print(f"SMILES for ethanol: {ethanol_smiles}")

        # Convert SMILES to natural language
        description = predictor.smiles_to_natural_language("CC(=O)O")
        print(f"Description for CC(=O)O: {description}")

        # Build reaction input from a JSON object
        reaction_data = {
            "reactants": ["benzene", "nitric acid"],
            "reagents": ["sulfuric acid"]
        }
        reaction_input = predictor.build_reaction_input_from_json(reaction_data)
        print(f"Formatted reaction input: {reaction_input}")

        # Download chemical structure image by name
        image_result = predictor.download_chemical_image("aspirin")
        if image_result and image_result['success']:
            print(f"Downloaded chemical image: {image_result['output_filename']}")

        # Download chemical structure image by formula
        filepath = predictor.download_chemical_image_by_formula("C6H12O6")
        if filepath:
            print(f"Downloaded glucose structure to: {filepath}")

        # Create a visual reaction diagram
        diagram_path = predictor.create_reaction_diagram(
            reactants=["H2", "O2"], 
            products=["H2O"]
        )
        if diagram_path:
            print(f"Reaction diagram saved to: {diagram_path}")
    """
    def __init__(self):
        """
        Initializes the ReactionPredictor.

        Loads the 'sagawa/ReactionT5v2-forward' model and tokenizer for reaction
        prediction and initializes the OpenAI client for natural language tasks.
        It expects an 'OPENAI_API_TOKEN' environment variable, typically stored
        in a .env file.
        """
        # Force Transformers to use local files only (no network)
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if not MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Local model directory not found: {MODEL_DIR}. "
                f"Download the model to this path before running."
            )
        self.tokenizer = AutoTokenizer.from_pretrained("sagawa/ReactionT5v2-forward", cache_dir="./models")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("sagawa/ReactionT5v2-forward", cache_dir="./models")
        
        # Use an absolute path for the AiZynthFinder config to be robust to CWD
        self.config_path = str((SCRIPT_DIR / "config.yml").resolve())
        
        # Download missing model files if URLs are provided
        self._download_missing_files()
        
        # Try to initialize AiZynthFinder, but make it optional
        try:
            self.finder = AiZynthFinder(configfile=self.config_path)
            self.finder.filter_policy.select("uspto")  
            
            # Try to select zinc stock, but continue if it fails
            try:
                self.finder.stock.select("zinc")
                print("✅ Zinc stock database loaded successfully")
            except Exception as e:
                print(f"⚠️ Warning: Could not load zinc stock database: {e}")
                print("⚠️ Retrosynthesis will work but may not stop at commercially available compounds")
            
            self.finder.expansion_policy.select("uspto")
            self.aizynthfinder_available = True
            print("✅ AiZynthFinder initialized successfully")
            
        except Exception as e:
            print(f"⚠️  AiZynthFinder not available: {e}")
            print("   Only forward prediction will be available")
            self.finder = None
            self.aizynthfinder_available = False
        
        # Initialize OpenAI client for natural language to SMILES conversion
        openai_token = os.getenv('OPENAI_API_TOKEN')
        if openai_token:
            self.openai_client = openai.OpenAI(api_key=openai_token)
            self.openai_available = True
        else:
            self.openai_client = None
            self.openai_available = False
            print("Warning: OPENAI_API_TOKEN not found in environment. Natural language features will not be available.")
            print("You can still use SMILES-based methods like predict_reaction() and test_from_csv().")

    def _download_missing_files(self):
        """Download missing model files from URLs if provided in environment variables"""
        
        # Configuration with download URLs from environment
        downloads = {
            "zinc_stock.hdf5": os.getenv("ZINC_STOCK_URL"),
            "uspto_model.onnx": os.getenv("USPTO_MODEL_URL"),
            "uspto_ringbreaker_model.onnx": os.getenv("USPTO_RINGBREAKER_MODEL_URL"),
            "uspto_filter_model.onnx": os.getenv("USPTO_FILTER_MODEL_URL"),
            "uspto_templates.csv.gz": os.getenv("USPTO_TEMPLATES_URL"),
            "uspto_ringbreaker_templates.csv.gz": os.getenv("USPTO_RINGBREAKER_TEMPLATES_URL"),
        }
        
        for filename, url in downloads.items():
            if url and url.strip():
                local_path = Path(filename)
                if not local_path.exists():
                    try:
                        print(f"📥 Downloading {filename} from {url[:50]}...")
                        response = requests.get(url, stream=True, timeout=300)
                        response.raise_for_status()
                        
                        with open(local_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        
                        print(f"✅ Downloaded {filename}")
                    except Exception as e:
                        print(f"❌ Failed to download {filename}: {e}")
                else:
                    print(f"⏭️  {filename} already exists, skipping download")


    
    def reconstruct_reaction_from_route(self, route_data, reaction_step):
        """
        Reconstruct the actual reaction based on the route structure,
        ignoring the potentially corrupted reaction SMILES
        """
        # Find the parent molecule (what's being disconnected)
        parent_mol = next(s for s in route_data['steps'] 
                          if s['idx'] == reaction_step['parent_idx'])
        
        # Find the product molecules (what it's being disconnected into)
        products = [s for s in route_data['steps'] 
                    if s.get('parent_idx') == reaction_step['idx']]
        
        # Build the correct reaction SMILES
        parent_smiles = parent_mol['smiles']
        product_smiles = [p['smiles'] for p in products]
        
        reconstructed_reaction = f"{parent_smiles}>>{'.'.join(product_smiles)}"
        
        return {
            'reactant': parent_smiles,
            'products': product_smiles,
            'reaction_smiles': reconstructed_reaction
        }
    
    def process_route_with_reconstruction(self, route_data):
        """
        Process the entire route, reconstructing reactions from the route structure
        rather than trusting the reaction SMILES
        """
        results = {}
        
        for step in route_data['steps']:
            if step['type'] == 'mol':
                # Process molecule normally
                try:
                    mol = Chem.MolFromSmiles(step['smiles'])
                    if mol:
                        formula = Descriptors.rdMolDescriptors.CalcMolFormula(mol)
                        results[step['idx']] = {
                            'smiles': step['smiles'],
                            'formula': formula
                        }
                    else:
                        results[step['idx']] = {
                            'smiles': step['smiles'],
                            'formula': 'Invalid SMILES'
                        }
                except Exception as e:
                    results[step['idx']] = {
                        'smiles': step['smiles'],
                        'formula': f'Error: {str(e)}'
                    }
                    
            elif step['type'] == 'reaction':
                # Reconstruct reaction from route structure
                reconstructed = self.reconstruct_reaction_from_route(route_data, step)
                
                # Calculate formulas
                try:
                    reactant_mol = Chem.MolFromSmiles(reconstructed['reactant'])
                    reactant_formula = Descriptors.rdMolDescriptors.CalcMolFormula(reactant_mol)
                    
                    product_formulas = []
                    for p_smiles in reconstructed['products']:
                        p_mol = Chem.MolFromSmiles(p_smiles)
                        if p_mol:
                            product_formulas.append(Descriptors.rdMolDescriptors.CalcMolFormula(p_mol))
                        else:
                            product_formulas.append('Invalid SMILES')
                    
                    results[step['idx']] = {
                        'original_reaction_smiles': step['smiles'],
                        'reconstructed_reaction_smiles': reconstructed['reaction_smiles'],
                        'reactants': [reconstructed['reactant']],
                        'products': reconstructed['products'],
                        'reactant_formulas': [reactant_formula],
                        'product_formulas': product_formulas
                    }
                except Exception as e:
                    results[step['idx']] = {
                        'error': f'Failed to process reaction: {str(e)}',
                        'original_reaction_smiles': step['smiles']
                    }
        
        return results
    
    def validate_reaction_smiles(self, reaction_smiles, expected_reactant=None, expected_products=None):
        """
        Validate if a reaction SMILES makes chemical sense and matches expected molecules
        """
        try:
            # Parse the reaction SMILES
            parts = reaction_smiles.split('>>')
            if len(parts) != 2:
                return False, "Invalid reaction format"
            
            # Clean and parse reactants
            reactant_smiles = re.sub(r':\d+', '', parts[0])  # Remove atom mapping
            product_smiles_list = [re.sub(r':\d+', '', p) for p in parts[1].split('.')]
            
            # Try to parse molecules
            reactant_mols = []
            for r in reactant_smiles.split('.'):
                mol = Chem.MolFromSmiles(r)
                if mol:
                    reactant_mols.append(Chem.MolToSmiles(mol))
            
            product_mols = []
            for p in product_smiles_list:
                mol = Chem.MolFromSmiles(p)
                if mol:
                    product_mols.append(Chem.MolToSmiles(mol))
            
            # Check if we got valid molecules
            if not reactant_mols or not product_mols:
                return False, "Could not parse molecules from reaction SMILES"
            
            # If expected molecules provided, check if they match
            if expected_reactant:
                expected_mol = Chem.MolFromSmiles(expected_reactant)
                if expected_mol:
                    expected_canonical = Chem.MolToSmiles(expected_mol)
                    if expected_canonical not in reactant_mols:
                        return False, f"Reactant mismatch. Expected: {expected_canonical}, Got: {reactant_mols}"
            
            if expected_products:
                expected_product_canonicals = []
                for ep in expected_products:
                    mol = Chem.MolFromSmiles(ep)
                    if mol:
                        expected_product_canonicals.append(Chem.MolToSmiles(mol))
                
                if set(expected_product_canonicals) != set(product_mols):
                    return False, f"Product mismatch. Expected: {expected_product_canonicals}, Got: {product_mols}"
            
            return True, "Valid reaction"
            
        except Exception as e:
            return False, f"Error validating reaction: {str(e)}"
    
    def process_any_route(self, route_data):
        """
        Main function to process any retrosynthetic route
        """
        #print("=== Processing Route ===\n")
        
        # First, check if reaction SMILES are valid
        for step in route_data['steps']:
            if step['type'] == 'reaction':
                # Get expected molecules from route structure
                parent = next(s for s in route_data['steps'] 
                             if s['idx'] == step['parent_idx'])
                products = [s for s in route_data['steps'] 
                           if s.get('parent_idx') == step['idx']]
                
                is_valid, message = self.validate_reaction_smiles(
                    step['smiles'],
                    expected_reactant=parent['smiles'],
                    expected_products=[p['smiles'] for p in products]
                )
                
                """if not is_valid:
                    print(f"Reaction at step {step['idx']} appears corrupted: {message}")
                    print("Will reconstruct from route structure instead.\n")"""
        
        # Process the route
        results = self.process_route_with_reconstruction(route_data)
        
        DISPLAY_RESULTS = False
        if DISPLAY_RESULTS: 
        # Display results
            for idx, data in results.items():
                print(f"Step {idx}:")
                for key, value in data.items():
                    print(f"  {key}: {value}")
                print()
            
        return results


    # this is currently used to deal with smiles in the reaction route prediction, not the reaction prediction itself
    def parse_reaction_smiles(self, reaction_smiles):
        try:
            # Try to parse as a reaction first
            rxn = AllChem.ReactionFromSmarts(reaction_smiles)
            
            # Get reactants and products
            reactants = []
            products = []
            reactant_formulas = []
            product_formulas = []
            
            # Process reactants
            for mol in rxn.GetReactants():
                # Update property cache to calculate implicit valences
                mol.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL^Chem.SANITIZE_PROPERTIES)
                
                # Get SMILES and formula
                smiles = Chem.MolToSmiles(mol)
                formula = Descriptors.rdMolDescriptors.CalcMolFormula(mol)
                
                reactants.append(smiles)
                reactant_formulas.append(formula)
            
            # Process products
            for mol in rxn.GetProducts():
                # Update property cache
                mol.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL^Chem.SANITIZE_PROPERTIES)
                
                # Get SMILES and formula
                smiles = Chem.MolToSmiles(mol)
                formula = Descriptors.rdMolDescriptors.CalcMolFormula(mol)
                
                products.append(smiles)
                product_formulas.append(formula)
            
            return {
                'reactants': reactants,
                'products': products,
                'reactant_formulas': reactant_formulas,
                'product_formulas': product_formulas
            }
            
        except Exception as e:
            print(f"Error parsing reaction: {e}")
            # Fallback: try manual parsing
            return self.parse_reaction_manually(reaction_smiles)


    def parse_reaction_manually(self, reaction_smiles):
        """Fallback parser that handles the reaction string manually"""
        import re
        
        # Remove atom mapping
        cleaned = re.sub(r':\d+', '', reaction_smiles)
        
        # Split into reactants and products
        parts = cleaned.split('>>')
        if len(parts) != 2:
            raise ValueError("Invalid reaction SMILES format")
        
        reactant_smiles = parts[0].split('.')
        product_smiles = parts[1].split('.')
        
        reactants = []
        products = []
        reactant_formulas = []
        product_formulas = []
        
        # Process each reactant
        for smi in reactant_smiles:
            # Clean up SMILES
            smi = smi.strip()
            # Handle special cases like [cH3] -> C
            smi = re.sub(r'\[cH(\d*)\]', r'C', smi)
            smi = re.sub(r'\[CH(\d*)\]', r'C', smi)
            
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    formula = Descriptors.rdMolDescriptors.CalcMolFormula(mol)
                    reactants.append(smi)
                    reactant_formulas.append(formula)
                else:
                    # If RDKit can't parse, just store the SMILES
                    reactants.append(smi)
                    reactant_formulas.append("Unable to parse")
            except:
                reactants.append(smi)
                reactant_formulas.append("Unable to parse")
        
        # Process each product
        for smi in product_smiles:
            smi = smi.strip()
            smi = re.sub(r'\[cH(\d*)\]', r'C', smi)
            smi = re.sub(r'\[CH(\d*)\]', r'C', smi)
            
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    formula = Descriptors.rdMolDescriptors.CalcMolFormula(mol)
                    products.append(smi)
                    product_formulas.append(formula)
                else:
                    products.append(smi)
                    product_formulas.append("Unable to parse")
            except:
                products.append(smi)
                product_formulas.append("Unable to parse")
        
        return {
            'reactants': reactants,
            'products': products,
            'reactant_formulas': reactant_formulas,
            'product_formulas': product_formulas
        }

    # Alternative: Simple formula extractor for your specific case
    # currently used to extract formulas from the reaction route prediction
    # not the reaction prediction 
    def extract_formulas_from_route(self, route_data):
        """Extract formulas for all molecules in the route"""
        results = {}
        
        for step in route_data['steps']:
            if step['type'] == 'mol':
                try:
                    mol = Chem.MolFromSmiles(step['smiles'])
                    if mol:
                        formula = Descriptors.rdMolDescriptors.CalcMolFormula(mol)
                        results[step['idx']] = {
                            'smiles': step['smiles'],
                            'formula': formula
                        }
                except:
                    results[step['idx']] = {
                        'smiles': step['smiles'],
                        'formula': 'Error parsing'
                    }
            elif step['type'] == 'reaction':
                # Parse the reaction
                result = self.parse_reaction_smiles(step['smiles'])
                results[step['idx']] = result
        
        return results
    
    # this takes the recursive retrosynthesis route (reaction route) dictionaries and flattens them into a list of non-recursive (flattened) route dicts.
    def flatten_all_routes(self,routes_list):
        """
        Takes a list of recursive retrosynthesis route dictionaries,
        returns a list of non-recursive (flattened) route dicts.
        """
        def flatten_route(route_dict):
            steps = []
            idx_counter = [0]
            def traverse(node, parent_idx=None, depth=0):
                idx = idx_counter[0]
                idx_counter[0] += 1
                step = {
                    "idx": idx,
                    "type": node.get("type"),
                    "smiles": node.get("smiles"),
                    "parent_idx": parent_idx,
                    "depth": depth,
                }
                if node.get("type") == "mol":
                    step["in_stock"] = node.get("in_stock", False)
                steps.append(step)
                for child in node.get("children", []):
                    traverse(child, parent_idx=idx, depth=depth+1)
            traverse(route_dict)
            return {"target_smiles": route_dict.get("smiles"), "steps": steps}
        # Apply to all routes in the list
        return [flatten_route(r) for r in routes_list]

    def find_pathways_to_smiles_from_formula(self, target_formula):
        input_smiles = self.find_smiles_from_formula(target_formula)
        return self.find_pathways_to_smiles(input_smiles)
    
   

    

    def find_pathways_to_smiles(self, target_smiles):
        if not self.aizynthfinder_available:
            return []
        self.finder.target_smiles = target_smiles
        self.finder.tree_search()
        self.finder.build_routes()
        return self.flatten_all_routes(self.finder.routes.dicts)
    
    def get_route_precursors(self, flat_route):
        """Extract final in-stock precursor SMILES from a flattened route."""
        steps = flat_route['steps']
        idx2step = {step['idx']: step for step in steps}
        # Find all parent idx for reactions
        reaction_parent_idxs = {step['parent_idx'] for step in steps if step['type'] == 'reaction'}
        precursors = []
        for step in steps:
            # Precursor: in stock, type=mol, not the parent of a reaction
            if step['type'] == 'mol' and step.get('in_stock', False):
                if step['idx'] not in reaction_parent_idxs:
                    precursors.append(step['smiles'])
        return precursors

    def smiles_list_similarity(self, list1, list2):
        """Jaccard similarity (for sets of SMILES, order-independent)"""
        set1, set2 = set(list1), set(list2)
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union

    def find_best_matching_route(self,flat_routes, input_smiles):
        """
        Given a list of flattened retrosynthesis routes and an input set (or single) SMILES,
        returns (best_route_index, best_similarity_score, best_route_dict).
        """
        # Always use a list for multi-reactant support
        input_smiles_list = [input_smiles] if isinstance(input_smiles, str) else list(input_smiles)

        best_route = None
        best_score = -1
        best_route_index = -1
        routes_with_scores = []
        for idx, flat_route in enumerate(flat_routes):
            #print(f"{json.dumps(flat_route, indent=2)}")
            #input("continue?")
            precursors = self.get_route_precursors(flat_route)
            score = self.smiles_list_similarity(precursors, input_smiles_list)
            #N = len(flat_route['steps']) - 1
            #print(f"{flat_route=}")
            #input("continue?")
            #flat_route = {i: flat_route[N - i] for i in range(len(flat_route))}
            routes_with_scores.append({
                "route": flat_route,
                "score": score
            })
            if score > best_score:
                best_score = score
                #best_route = flat_route
                #best_route_index = idx
        # delete all items in the dictionary which don't have the best score
        routes_with_scores = [{"route": i['route'], "score": i['score']} for i in routes_with_scores if i['score'] == best_score]
        # get the first item in the dictionary
        #best_route = routes_with_scores[0]['route']
        #best_route_index = 0
        for idx, route in enumerate(routes_with_scores):
            new_route = self.process_any_route(route['route'])
            # now go through route and swap any substring ">>" in values with "<<"
            # also swap any substring in a key of "reactant" with "product" and vice versa.
            # (because it's retrosynthetic)
            #print(f"{new_route=}")
            #input("continue?")
            #make rebuilt_route a deepcopy of new_route
            rebuilt_route = deepcopy(new_route)
            for steps in new_route.items():
                step_idx = steps[0]
                step_dict = steps[1]
                for step_dict_items in step_dict.items():
                    key = step_dict_items[0]
                    value = step_dict_items[1]
                    if ">>" in value:
                        reaction = value.split(">>")
                        rebuilt_route[step_idx][key] = reaction[1] + ">>" + reaction[0]
                    if "reactant" in key:
                        new_key = key.replace("reactant", "product")
                        rebuilt_route[step_idx][new_key] = value
                        # delete the old key
                        del rebuilt_route[step_idx][key]
                    elif "product" in key:
                        new_key = key.replace("product", "reactant")
                        rebuilt_route[step_idx][new_key] = value
                        # delete the old key
                        del rebuilt_route[step_idx][key]
            new_route = rebuilt_route
            REVERSE = True
            if REVERSE:
                # reverse keys of the route so it starts with input and ends with product
                N = len(new_route) - 1
                new_route = {i: new_route[N - i] for i in range(len(new_route))}
            # now go through the route and if it is a step which only has the keys "smiles" and "formula", delete the step
            new_route_copy = deepcopy(new_route)
            for step_idx, step_dict in new_route_copy.items():
                if len(step_dict) == 2 and "smiles" in step_dict and "formula" in step_dict:
                    del new_route[step_idx]
                # if "original_reaction_smiles" in step_dict, delete that item in the dictionary
                if "original_reaction_smiles" in step_dict:
                    # delete it from original new_route
                    del new_route[step_idx]["original_reaction_smiles"]
            # renumber the steps to start from 0
            new_route = {renum: new_route[i] for renum, i in enumerate(new_route.keys())}
            routes_with_scores[idx]['route'] = new_route

        # reverse the step numbering in the route and reorder the dictionary keys
        
        #best_route = self.check_if_input_smiles_is_in_route(best_route, input_smiles)
        #return best_route_index, best_score, best_route
        
        return routes_with_scores, best_score

    def check_if_input_smiles_is_in_route(self, flat_route, input_smiles):
        """
        manually checks if the input SMILES is in the route but is not a precursor. In that case, delete all steps before the input SMILES. And renumber the steps to start from 0.
        """
        
        for idx, step in enumerate(flat_route.items()):
            if step[1]['smiles'] == input_smiles and idx != 0:
                    print(f"{flat_route=}")
                    print(f"{idx=}")
                    input("continue?")
                    # delete all steps before the input SMILES
                    for i in range(idx):
                        del flat_route[i]
                    # renumber the steps to start from 0
                    N = len(flat_route) - 1
                    flat_route = {i: flat_route[N - i] for i in range(len(flat_route))}
                    print(f"NOW {flat_route=}")
                    input("continue?")
                    return flat_route
        return flat_route

    def predict_reaction(self, reactant_smiles=None, reagent_smiles=None, whole_input=None):
        """
        Predicts the product of a chemical reaction.

        This method can take SMILES strings for reactants and reagents or a
        pre-formatted input string. Reactants and reagents can be passed as
        individual strings or as lists of strings, which will be concatenated
        with dots.

        Args:
            reactant_smiles (str or list, optional): SMILES string(s) for the reactant(s).
                Can be a single string or a list of strings. Required if 'whole_input' 
                is not provided.
            reagent_smiles (str or list, optional): SMILES string(s) for the reagent(s).
                Can be a single string or a list of strings. Can be None or an empty 
                string if no reagent is used. Required if 'whole_input' is not provided.
            whole_input (str, optional): A complete, pre-formatted input string
                for the model. If provided, 'reactant_smiles' and 'reagent_smiles'
                are ignored. Example format: 'REACTANT:CCOREAGENT:[H]Cl'.

        Returns:
            str: The SMILES string of the predicted product.

        Raises:
            ValueError: If 'whole_input' is None and 'reactant_smiles' is not
                provided.
        """
        # Use whole_input if provided, otherwise format from reactant and reagent
        if whole_input is not None:
            input_string = whole_input
        else:
            # Validate that reactant_smiles is provided when whole_input is not
            if reactant_smiles is None:
                raise ValueError("Either whole_input must be provided, or reactant_smiles must be provided")
            
            # Convert lists to concatenated strings if needed
            if isinstance(reactant_smiles, list):
                if len(reactant_smiles) == 0:
                    raise ValueError("reactant_smiles list cannot be empty")
                reactant_smiles = ".".join(reactant_smiles)
            
            if isinstance(reagent_smiles, list):
                if len(reagent_smiles) == 0:
                    reagent_smiles = ""  # Empty list becomes empty string
                else:
                    reagent_smiles = ".".join(reagent_smiles)
            
            # Format the input string according to the model's expected format
            if reagent_smiles is None or reagent_smiles == "":
                input_string = f'REACTANT:{reactant_smiles}REAGENT:'
            else:
                input_string = f'REACTANT:{reactant_smiles}REAGENT:{reagent_smiles}'
        
        # Tokenize the input
        inp = self.tokenizer(input_string, return_tensors='pt')
        
        # Generate prediction
        output = self.model.generate(**inp, num_beams=1, num_return_sequences=1, return_dict_in_generate=True, output_scores=True)
        
        # Decode and clean the output
        predicted_smiles = self.tokenizer.decode(output['sequences'][0], skip_special_tokens=True).replace(' ', '').rstrip('.')
        
        return predicted_smiles
    
    def natural_language_to_smiles(self, description):
        """
        Converts a natural language description of a chemical compound to a SMILES string.

        This method uses the OpenAI GPT-4.1 model. The result, or any error,
        is saved to a JSON file ('nl_to_smiles_result.json' or
        'nl_to_smiles_error.json').

        Args:
            description (str): Natural language description of a chemical compound
                (e.g., "ethanol", "benzene").

        Returns:
            str or None: The SMILES string representation of the compound, or None
                if an error occurs.
        """
        if not self.openai_available:
            error_msg = "OpenAI API token not available. Cannot convert natural language to SMILES."
            print(error_msg)
            error_result = {"error": error_msg, "input_description": description, "smiles": None}
            
            # Write error to JSON file as per user preference
            output_filename = "nl_to_smiles_error.json"
            with open(output_filename, 'w') as f:
                json.dump(error_result, f, indent=2)
            
            return None

        prompt = f"""You are a chemistry expert. Convert the following natural language description of a chemical compound into SMILES notation.

Description: {description}

Please respond with only the SMILES string, no additional text or formatting.

Examples:
Input: "ethanol"
Output: CCO

Input: "benzene"
Output: c1ccccc1

Input: "acetic acid"
Output: CC(=O)O"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are a chemistry expert specializing in converting natural language descriptions to SMILES notation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            # Get the SMILES string directly
            smiles_result = response.choices[0].message.content.strip()
            
            # Write result to JSON file as per user preference
            result_data = {"input_description": description, "smiles": smiles_result}
            output_filename = "nl_to_smiles_result.json"
            with open(output_filename, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            return smiles_result
            
        except Exception as e:
            error_result = {"error": str(e), "input_description": description, "smiles": None}
            
            # Write error to JSON file as well
            output_filename = "nl_to_smiles_error.json"
            with open(output_filename, 'w') as f:
                json.dump(error_result, f, indent=2)
                
            return None
    
    def smiles_to_natural_language(self, smiles):
        """
        Converts a SMILES string to a natural language description.

        This method uses the OpenAI GPT-4.1 model. The result, or any error,
        is saved to a JSON file ('smiles_to_nl_result.json' or
        'smiles_to_nl_error.json').

        Args:
            smiles (str): The SMILES string representation of a chemical compound.

        Returns:
            str or None: A natural language description of the compound, or None
                if an error occurs.
        """
        if not self.openai_available:
            error_msg = "OpenAI API token not available. Cannot convert SMILES to natural language."
            print(error_msg)
            error_result = {"error": error_msg, "input_smiles": smiles, "description": None}
            
            # Write error to JSON file as per user preference
            output_filename = "smiles_to_nl_error.json"
            with open(output_filename, 'w') as f:
                json.dump(error_result, f, indent=2)
            
            return None

        prompt = f"""You are a chemistry expert. Convert the following SMILES notation into a natural language description of the chemical compound.

SMILES: {smiles}

Please respond with only the common chemical name or a clear description of the compound, no additional text or formatting.

Examples:
Input: CCO
Output: ethanol

Input: c1ccccc1
Output: benzene

Input: CC(=O)O
Output: acetic acid

Input: ClC1=CC=CC(Br)=C1
Output: 1-bromo-3-chlorobenzene"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are a chemistry expert specializing in converting SMILES notation to natural language descriptions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            # Get the natural language description directly
            description_result = response.choices[0].message.content.strip()
            
            # Write result to JSON file as per user preference
            result_data = {"input_smiles": smiles, "description": description_result}
            output_filename = "smiles_to_nl_result.json"
            with open(output_filename, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            return description_result
            
        except Exception as e:
            error_result = {"error": str(e), "input_smiles": smiles, "description": None}
            
            # Write error to JSON file as well
            output_filename = "smiles_to_nl_error.json"
            with open(output_filename, 'w') as f:
                json.dump(error_result, f, indent=2)
                
            return None
    
    def build_reaction_input_from_json(self, reaction_json):
        """
        Builds a reaction input string from a JSON object with natural language descriptions.

        This method takes a dictionary containing lists of reactants and reagents
        described in natural language, converts each to its SMILES representation,
        and then formats them into a single input string suitable for the
        `predict_reaction` method.

        The result, or any error, is saved to a JSON file
        ('reaction_input_build_result.json' or 'reaction_input_build_error.json').

        Args:
            reaction_json (dict): A dictionary with 'reactants' and 'reagents' keys.
                The values should be lists of strings, where each string is a
                natural language name for a compound.
                Example:
                {
                    "reactants": ["benzene", "nitric acid"],
                    "reagents": ["sulfuric acid"]
                }

        Returns:
            str or None: The formatted reaction input string for the prediction
                model, or None if an error occurs.
        """
        if not self.openai_available:
            error_msg = "OpenAI API token not available. Cannot build reaction input from natural language JSON."
            print(error_msg)
            error_result = {
                "error": error_msg,
                "input_json": reaction_json,
                "final_input_string": None
            }
            
            # Write error to JSON file as per user preference
            output_filename = "reaction_input_build_error.json"
            with open(output_filename, 'w') as f:
                json.dump(error_result, f, indent=2)
            
            return None

        try:
            # Extract reactants and reagents lists
            reactants_nl = reaction_json.get('reactants', [])
            reagents_nl = reaction_json.get('reagents', [])
            
            # Convert reactants to SMILES
            reactant_smiles_list = []
            for reactant in reactants_nl:
                smiles = self.natural_language_to_smiles(reactant)
                if smiles:
                    reactant_smiles_list.append(smiles)
            
            # Convert reagents to SMILES
            reagent_smiles_list = []
            for reagent in reagents_nl:
                smiles = self.natural_language_to_smiles(reagent)
                if smiles:
                    reagent_smiles_list.append(smiles)
            
            # Concatenate reactants with "."
            concatenated_reactants = ".".join(reactant_smiles_list)
            
            # Concatenate reagents with "."
            concatenated_reagents = ".".join(reagent_smiles_list)
            
            # Build the final input string
            if concatenated_reagents:
                input_string = f"REACTANT:{concatenated_reactants}REAGENT:{concatenated_reagents}"
            else:
                input_string = f"REACTANT:{concatenated_reactants}REAGENT:"
            
            # Write result to JSON file as per user preference
            result_data = {
                "input_json": reaction_json,
                "reactant_smiles": reactant_smiles_list,
                "reagent_smiles": reagent_smiles_list,
                "final_input_string": input_string
            }
            output_filename = "reaction_input_build_result.json"
            with open(output_filename, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            return input_string
            
        except Exception as e:
            error_result = {
                "error": str(e),
                "input_json": reaction_json,
                "final_input_string": None
            }
            
            # Write error to JSON file as well
            output_filename = "reaction_input_build_error.json"
            with open(output_filename, 'w') as f:
                json.dump(error_result, f, indent=2)
                
            return None

    def test_from_csv(self, csv_filepath, output_filepath="test_results.csv"):
        """
        Tests reaction predictions from a CSV file and saves the results.

        The input CSV file must contain an 'input' column with the full reaction
        string and a 'PRODUCT' column with the expected SMILES result.

        The method generates an output CSV file that includes the original data
        plus two new columns: 'predicted_product' and 'test_passed' (a boolean
        indicating if the prediction matched the expected product). A summary is
        also printed to the console.

        Args:
            csv_filepath (str): The path to the input CSV file.
            output_filepath (str, optional): The path to save the output CSV file.
                Defaults to "test_results.csv".
        """
        try:
            df = pd.read_csv(csv_filepath)
        except FileNotFoundError:
            print(f"Error: The file {csv_filepath} was not found.")
            return

        predictions = []
        results = []

        for index, row in df.iterrows():
            input_string = row['input']
            expected_product = row['PRODUCT']

            predicted_product = self.predict_reaction(whole_input=input_string)
            predictions.append(predicted_product)

            test_passed = (predicted_product == expected_product)
            results.append(test_passed)

        df['predicted_product'] = predictions
        df['test_passed'] = results

        # Save the results to a new CSV file
        try:
            df.to_csv(output_filepath, index=False)
            print(f"Results saved to {output_filepath}")
        except Exception as e:
            print(f"Error saving results to {output_filepath}: {e}")


        passed_tests = sum(results)
        total_tests = len(df)
        failed_tests_count = total_tests - passed_tests

        print("--- CSV Test Summary ---")
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests_count}")

        if failed_tests_count > 0:
            print("\n--- Failed Tests Details ---")
            for index, row in df[df['test_passed'] == False].iterrows():
                print(f"Row {index + 2}:")
                print(f"  Input:    {row['input']}")
                print(f"  Expected: {row['PRODUCT']}")
                print(f"  Predicted: {row['predicted_product']}\n")

    def test_from_natural_language_csv(self, csv_filepath, output_filepath="natural_language_test_results.csv"):
        """
        Tests reaction predictions from a CSV file containing natural language descriptions.

        The input CSV file must contain an 'input' column with natural language
        descriptions of reactions and a 'PRODUCT' column with natural language
        descriptions of expected products.

        This method converts natural language descriptions to SMILES for prediction,
        then converts the predicted SMILES back to natural language descriptions.
        The output CSV includes the original data plus 'predicted_product_nl' and
        'test_passed' columns.

        Args:
            csv_filepath (str): The path to the input CSV file.
            output_filepath (str, optional): The path to save the output CSV file.
                Defaults to "natural_language_test_results.csv".
        """
        if not self.openai_available:
            print("Error: OpenAI API token not available. Cannot run natural language CSV tests.")
            print("Please set the OPENAI_API_TOKEN environment variable to use this feature.")
            return

        try:
            # Read CSV with proper handling of commas within text fields
            df = pd.read_csv(csv_filepath, quoting=1)  # QUOTE_ALL mode
        except FileNotFoundError:
            print(f"Error: The file {csv_filepath} was not found.")
            return
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            print("Trying alternative CSV reading method...")
            try:
                # Alternative method: read with different quoting
                df = pd.read_csv(csv_filepath, quoting=3)  # QUOTE_NONE mode
            except Exception as e2:
                print(f"Alternative method also failed: {e2}")
                return

        predictions_nl = []
        results = []

        for index, row in df.iterrows():
            print(f"Processing row {index + 1}/{len(df)}...")
            input_description = row['input']
            expected_product_nl = row['PRODUCT']

            # Convert natural language input to SMILES for prediction
            print(f"  Converting input to SMILES...")
            input_smiles = self.natural_language_to_smiles(input_description)
            
            if input_smiles:
                # Predict the product in SMILES format
                print(f"  Predicting reaction product...")
                predicted_product_smiles = self.predict_reaction(whole_input=input_smiles)
                
                # Convert predicted SMILES back to natural language
                print(f"  Converting prediction to natural language...")
                predicted_product_nl = self.smiles_to_natural_language(predicted_product_smiles)
                
                if predicted_product_nl:
                    predictions_nl.append(predicted_product_nl)
                    # Simple string comparison for test result
                    test_passed = (predicted_product_nl.lower().strip() == expected_product_nl.lower().strip())
                    results.append(test_passed)
                    print(f"  Row {index + 1} completed successfully")
                else:
                    predictions_nl.append("Conversion failed")
                    results.append(False)
                    print(f"  Row {index + 1} failed: prediction conversion failed")
            else:
                predictions_nl.append("Input conversion failed")
                results.append(False)
                print(f"  Row {index + 1} failed: input conversion failed")

        # Create a new DataFrame with only the required columns in the specified order
        output_df = pd.DataFrame({
            'input': df['input'],
            'expected_output': df['PRODUCT'],
            'actual_output': predictions_nl
        })

        # Save the results to a new CSV file
        try:
            output_df.to_csv(output_filepath, index=False)
            print(f"Results saved to {output_filepath}")
        except Exception as e:
            print(f"Error saving results to {output_filepath}: {e}")

        passed_tests = sum(results)
        total_tests = len(df)
        failed_tests_count = total_tests - passed_tests

        print("--- Natural Language CSV Test Summary ---")
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests_count}")

        if failed_tests_count > 0:
            print("\n--- Failed Tests Details ---")
            for i, (passed, input_desc, expected, actual) in enumerate(zip(results, df['input'], df['PRODUCT'], predictions_nl)):
                if not passed:
                    print(f"Row {i + 2}:")
                    print(f"  Input:    {input_desc}")
                    print(f"  Expected: {expected}")
                    print(f"  Predicted: {actual}\n")

    def search_pubchem_by_smiles(self, smiles):
        """
        Searches PubChem for a chemical compound using its SMILES string.

        This method uses the PubChem REST API to find chemical information
        including names, molecular formulas, molecular weights, and other
        identifiers. The result is saved to a JSON file.

        Args:
            smiles (str): The SMILES string representation of a chemical compound.

        Returns:
            dict or None: A dictionary containing the chemical information from PubChem,
                or None if an error occurs or no results are found.

        Example:
            result = predictor.search_pubchem_by_smiles("CCO")
            if result:
                print(f"Compound name: {result.get('IUPACName', 'N/A')}")
                print(f"Molecular formula: {result.get('MolecularFormula', 'N/A')}")
        """
        try:
            # URL encode the SMILES string for safe API calls
            encoded_smiles = quote(smiles)
            
            # PubChem API endpoint for SMILES search - try different endpoints (2025 format)
            urls_to_try = [
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded_smiles}/cids/JSON",
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded_smiles}/property/IUPACName,MolecularFormula,MolecularWeight,SMILES/JSON",
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded_smiles}/JSON"
            ]
            
            # Make the API request with proper headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json'
            }
            
            data = None
            for i, url in enumerate(urls_to_try):
                try:
                    print(f"Trying PubChem API endpoint {i+1}: {url}")
                    response = requests.get(url, headers=headers, timeout=30)
                    print(f"Response status: {response.status_code}")
                    if response.status_code == 200 and response.text.strip():
                        data = response.json()
                        print(f"Successfully got data from endpoint {i+1}")
                        break
                    else:
                        print(f"Endpoint {i+1} failed - status: {response.status_code}, content preview: {response.text[:100]}...")
                except Exception as e:
                    print(f"Endpoint {i+1} exception: {str(e)}")
                    continue
            
            if data is None:
                error_msg = "All PubChem API endpoints failed to return valid data"
                print(error_msg)
                error_result = {"error": error_msg, "smiles": smiles, "result": None}
                
                # Write error to JSON file
                output_filename = "pubchem_search_error.json"
                with open(output_filename, 'w') as f:
                    json.dump(error_result, f, indent=2)
                
                return None
            
            # Extract the first compound result
            if 'PC_Compounds' in data and len(data['PC_Compounds']) > 0:
                    compound = data['PC_Compounds'][0]
                    
                    # Extract basic information
                    result = {
                        'smiles': smiles,
                        'pubchem_cid': compound.get('id', {}).get('id', {}).get('cid', 'N/A'),
                        'molecular_formula': 'N/A',
                        'molecular_weight': 'N/A',
                        'iupac_name': 'N/A',
                        'common_names': [],
                        'synonyms': []
                    }
                    
                    # Extract molecular formula and weight
                    if 'props' in compound:
                        for prop in compound['props']:
                            if 'urn' in prop and 'label' in prop['urn']:
                                label = prop['urn']['label']
                                if label == 'Molecular Formula':
                                    result['molecular_formula'] = prop.get('value', {}).get('sval', 'N/A')
                                elif label == 'Molecular Weight':
                                    result['molecular_weight'] = prop.get('value', {}).get('sval', 'N/A')
                                elif label == 'IUPAC Name':
                                    result['iupac_name'] = prop.get('value', {}).get('sval', 'N/A')
                                elif label == 'Title':
                                    result['common_names'].append(prop.get('value', {}).get('sval', ''))
                    
                    # Write result to JSON file as per user preference
                    output_filename = "pubchem_search_result.json"
                    with open(output_filename, 'w') as f:
                        json.dump(result, f, indent=2)
                    
                    return result
            else:
                error_msg = f"No compounds found for SMILES: {smiles}"
                print(error_msg)
                error_result = {"error": error_msg, "smiles": smiles, "result": None}
                
                # Write error to JSON file
                output_filename = "pubchem_search_error.json"
                with open(output_filename, 'w') as f:
                    json.dump(error_result, f, indent=2)
                
                return None

        except requests.exceptions.RequestException as e:
            error_msg = f"Network error during PubChem search: {str(e)}"
            print(error_msg)
            error_result = {"error": error_msg, "smiles": smiles, "result": None}
            
            # Write error to JSON file
            output_filename = "pubchem_search_error.json"
            with open(output_filename, 'w') as f:
                json.dump(error_result, f, indent=2)
            
            return None
        except Exception as e:
            error_msg = f"Unexpected error during PubChem search: {str(e)}"
            print(error_msg)
            error_result = {"error": error_msg, "smiles": smiles, "result": None}
            
            # Write error to JSON file
            output_filename = "pubchem_search_error.json"
            with open(output_filename, 'w') as f:
                json.dump(error_result, f, indent=2)
            
            return None


    def search_pubchem_by_formula(self, molecular_formula, max_results=5):
        """
        Searches PubChem for compounds with a given molecular formula and returns
        the closest N matches as tuples.

        This method uses the PubChem REST API to find compounds matching the
        molecular formula. The results are sorted by relevance and returned
        as tuples in the format (molecular_formula, name, smiles).

        Args:
            molecular_formula (str): The molecular formula to search for (e.g., "C2H6O").
            max_results (int, optional): Maximum number of results to return. Defaults to 5.

        Returns:
            list or None: A list of tuples, each containing (molecular_formula, name, smiles).
                Returns None if an error occurs.

        Example:
            results = predictor.search_pubchem_by_formula("C2H6O", max_results=3)
            if results:
                for formula, name, smiles in results:
                    print(f"Formula: {formula}, Name: {name}, SMILES: {smiles}")
        """
        try:
            # URL encode the molecular formula for safe API calls
            encoded_formula = quote(molecular_formula)
            
            # Make the API request with proper headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json'
            }
            
            # Step 1: Get the CID for the compound name
            cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_formula}/cids/JSON"
            
            #print(f"Step 1: Getting CID for {molecular_formula}")
            response = requests.get(cid_url, headers=headers, timeout=30)
            #print(f"CID response status: {response.status_code}")
            
            if response.status_code == 200:
                cid_data = response.json()
                if 'IdentifierList' in cid_data and 'CID' in cid_data['IdentifierList']:
                    cids = cid_data['IdentifierList']['CID']
                    #print(f"Found CIDs: {cids}")
                    
                    results = []
                    for cid in cids[:max_results]:  # Limit to max_results
                        # Step 2: Get properties for each CID
                        #print(f"Step 2: Getting properties for CID {cid}")
                        prop_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
                        
                        prop_response = requests.get(prop_url, headers=headers, timeout=30)
                        #print(f"Properties response status: {prop_response.status_code}")
                        
                        if prop_response.status_code == 200:
                            prop_data = prop_response.json()
                            if 'PropertyTable' in prop_data and 'Properties' in prop_data['PropertyTable']:
                                for prop in prop_data['PropertyTable']['Properties']:
                                    smiles = prop.get('ConnectivitySMILES', 'N/A')
                                    name = 'N/A'  # We're only getting SMILES, not name
                                    formula = molecular_formula
                                    
                                    # Create tuple in the format (molecular_formula, name, smiles)
                                    results.append((formula, name, smiles))
                                    #  print(f"Found compound: {formula}, {name}, {smiles}")
                    
                    # Write results to JSON file as per user preference
                    output_filename = "pubchem_formula_search_result.json"
                    with open(output_filename, 'w') as f:
                        json.dump({
                            'search_formula': molecular_formula,
                            'max_results_requested': max_results,
                            'results_found': len(results),
                            'compounds': [{'formula': formula, 'name': name, 'smiles': smiles} for formula, name, smiles in results]
                        }, f, indent=2)
                    
                    return results
                else:
                    print("No CIDs found in response")
            else:
                pass
                #print(f"CID request failed with status: {response.status_code}")
            
            # If we get here, the two-step approach failed
            error_msg = "Two-step PubChem search failed"
            #print(error_msg)
            error_result = {
                "error": error_msg, 
                "molecular_formula": molecular_formula, 
                "max_results": max_results,
                "results": None
            }
            
            # Write error to JSON file
            output_filename = "pubchem_formula_search_error.json"
            with open(output_filename, 'w') as f:
                json.dump(error_result, f, indent=2)
            
            return None
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error during PubChem formula search: {str(e)}"
            print(error_msg)
            error_result = {
                "error": error_msg, 
                "molecular_formula": molecular_formula, 
                "max_results": max_results,
                "results": None
            }
            
            # Write error to JSON file
            output_filename = "pubchem_formula_search_error.json"
            with open(output_filename, 'w') as f:
                json.dump(error_result, f, indent=2)
            
            return None
        except Exception as e:
            error_msg = f"Unexpected error during PubChem formula search: {str(e)}"
            print(error_msg)
            error_result = {
                "error": error_msg, 
                "molecular_formula": molecular_formula, 
                "max_results": max_results,
                "results": None
            }
            
            # Write error to JSON file
            output_filename = "pubchem_formula_search_error.json"
            with open(output_filename, 'w') as f:
                json.dump(error_result, f, indent=2)
            
                return None


    def predict_reaction_from_formulae(self, reactant_formulae=None, reagent_formulae=None, max_results_per_formula=3):
        """
        Predicts a chemical reaction from molecular formulae using PubChem lookups.
        
        This method takes molecular formulae for reactants and reagents, looks up their
        SMILES representations using PubChem, predicts the reaction product, then breaks
        down the product to get molecular formulae and names.
        
        Args:
            reactant_formulae (list, optional): List of molecular formulae for reactants.
            reagent_formulae (list, optional): List of molecular formulae for reagents.
            max_results_per_formula (int, optional): Maximum PubChem results per formula. Defaults to 3.
            
        Returns:
            dict or None: A dictionary containing the reaction details, or None if an error occurs.
                Format: {
                    'reactant_formulae': [...],
                    'reagent_formulae': [...],
                    'reactant_smiles': [...],
                    'reagent_smiles': [...],
                    'predicted_product_smiles': '...',
                    'product_breakdown': [
                        {'formula': '...', 'name': '...', 'smiles': '...'},
                        ...
                    ]
                }
        """
        try:
            # Initialize results
            reactant_smiles_list = []
            reagent_smiles_list = []
            reaction_details = {
                'reactant_formulae': reactant_formulae or [],
                'reagent_formulae': reagent_formulae or [],
                'reactant_smiles': [],
                'reagent_smiles': [],
                'predicted_product_smiles': '',
                'product_breakdown': []
            }
            
            print("=== Reaction Prediction from Molecular Formulae ===")
            
            # Step 1: Convert reactant formulae to SMILES
            if reactant_formulae:
                print(f"Converting {len(reactant_formulae)} reactant formulae to SMILES...")
                for formula in reactant_formulae:
                    print(f"  Looking up formula: {formula}")
                    formula_results = self.search_pubchem_by_formula(formula, max_results_per_formula)
                    
                    if formula_results and len(formula_results) > 0:
                        # Use the first (most relevant) result
                        _, _, smiles = formula_results[0]
                        reactant_smiles_list.append(smiles)
                        reaction_details['reactant_smiles'].append(smiles)
                        print(f"    Found SMILES: {smiles}")
                    else:
                        print(f"    No SMILES found for {formula}")
                        reactant_smiles_list.append('N/A')
                        reaction_details['reactant_smiles'].append('N/A')
            
            # Step 2: Convert reagent formulae to SMILES
            if reagent_formulae:
                print(f"Converting {len(reagent_formulae)} reagent formulae to SMILES...")
                for formula in reagent_formulae:
                    print(f"  Looking up formula: {formula}")
                    formula_results = self.search_pubchem_by_formula(formula, max_results_per_formula)
                    
                    if formula_results and len(formula_results) > 0:
                        # Use the first (most relevant) result
                        _, _, smiles = formula_results[0]
                        reagent_smiles_list.append(smiles)
                        reaction_details['reagent_smiles'].append(smiles)
                        print(f"    Found SMILES: {smiles}")
                    else:
                        print(f"    No SMILES found for {formula}")
                        reagent_smiles_list.append('N/A')
                        reaction_details['reagent_smiles'].append('N/A')
            
            # Step 3: Predict the reaction
            print("Predicting reaction product...")
            if reactant_smiles_list and any(smiles != 'N/A' for smiles in reactant_smiles_list):
                # Filter out 'N/A' values
                valid_reactants = [s for s in reactant_smiles_list if s != 'N/A']
                valid_reagents = [s for s in reagent_smiles_list if s != 'N/A'] if reagent_smiles_list else None
                
                if valid_reagents:
                    predicted_product = self.predict_reaction(reactant_smiles=valid_reactants, reagent_smiles=valid_reagents)
                else:
                    predicted_product = self.predict_reaction(reactant_smiles=valid_reactants)
                
                reaction_details['predicted_product_smiles'] = predicted_product
                print(f"  Predicted product SMILES: {predicted_product}")
                
                # Step 4: Break down the product and get molecular formulae/names
                print("Breaking down product into molecular formulae...")
                if predicted_product and predicted_product != 'N/A':
                    # Try to identify individual compounds in the product
                    # This is a simplified approach - in practice, you might need more sophisticated parsing
                    product_compounds = self._break_down_product_smiles(predicted_product)
                    
                    for compound_smiles in product_compounds:
                        print(f"  Analyzing compound: {compound_smiles}")
                        
                        # Search PubChem for this SMILES to get formula and name
                        pubchem_result = self.search_pubchem_by_smiles(compound_smiles)
                        
                        if pubchem_result:
                            compound_info = {
                                'formula': pubchem_result.get('molecular_formula', 'Unknown'),
                                'name': pubchem_result.get('iupac_name', 'Unknown'),
                                'smiles': compound_smiles
                            }
                        else:
                            compound_info = {
                                'formula': 'Unknown',
                                'name': 'Unknown',
                                'smiles': compound_smiles
                            }
                        
                        reaction_details['product_breakdown'].append(compound_info)
                        print(f"    Found: {compound_info['formula']} ({compound_info['name']})")
                else:
                    print("  No valid product predicted")
            else:
                print("  No valid reactant SMILES found")
            
            # Write results to JSON file as per user preference
            output_filename = "reaction_from_formulae_result.json"
            with open(output_filename, 'w') as f:
                json.dump(reaction_details, f, indent=2)
            
            return reaction_details
            
        except Exception as e:
            error_msg = f"Error in predict_reaction_from_formulae: {str(e)}"
            print(error_msg)
            error_result = {
                "error": error_msg,
                "reactant_formulae": reactant_formulae,
                "reagent_formulae": reagent_formulae
            }
            
            # Write error to JSON file
            output_filename = "reaction_from_formulae_error.json"
            with open(output_filename, 'w') as f:
                json.dump(error_result, f, indent=2)
            
            return None
    
    def _break_down_product_smiles(self, product_smiles):
        """
        Helper method to break down a product SMILES into individual compounds.
        This is a simplified approach - in practice, you might need more sophisticated parsing.
        
        Args:
            product_smiles (str): The SMILES string of the predicted product.
            
        Returns:
            list: List of individual compound SMILES strings.
        """
        # Simple approach: split by dots (which separate different molecules)
        compounds = product_smiles.split('.')
        
        # Filter out empty strings and clean up
        compounds = [comp.strip() for comp in compounds if comp.strip()]
        
        # If no dots found, treat the whole thing as one compound
        if not compounds:
            compounds = [product_smiles]
        
        return compounds

    def download_chemical_image(self, chemical_name, image_size="large", output_filename=None):
        """
        Downloads a chemical structure image from PubChem using the chemical name.

        This method uses PubChem's PUG-REST API to download a PNG image of the
        chemical structure. The result metadata is saved to a JSON file.

        Args:
            chemical_name (str): The name of the chemical compound to download.
            image_size (str, optional): Size of the image. Can be "small", "large", 
                or specific dimensions like "300x300". Defaults to "large".
            output_filename (str, optional): Name of the output PNG file. If not provided,
                will use the chemical name with .png extension.

        Returns:
            dict or None: A dictionary containing download metadata, or None if an error occurs.
                Format: {
                    'chemical_name': '...',
                    'output_filename': '...',
                    'image_size': '...',
                    'file_size_bytes': ...,
                    'success': True/False
                }

        Example:
            result = predictor.download_chemical_image("aspirin")
            if result and result['success']:
                print(f"Downloaded {result['chemical_name']} to {result['output_filename']}")
        """
        try:
            # URL encode the chemical name for safe API calls
            encoded_name = quote(chemical_name)
            
            # Generate output filename if not provided
            if output_filename is None:
                # Clean the chemical name for use as filename
                safe_name = "".join(c for c in chemical_name if c.isalnum() or c in (' ', '-', '_')).strip()
                safe_name = safe_name.replace(' ', '_')
                output_filename = f"{safe_name}.png"
            
            # Construct the PubChem API URL
            if image_size in ["small", "large"]:
                url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/PNG?image_size={image_size}"
            else:
                # Assume it's specific dimensions like "300x300"
                url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/PNG?image_size={image_size}"
            
            print(f"Downloading chemical structure image for: {chemical_name}")
            print(f"API URL: {url}")
            print(f"Output file: {output_filename}")
            
            # Make the API request with proper headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'image/png'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200 and response.content:
                # Save the image to file
                with open(output_filename, 'wb') as f:
                    f.write(response.content)
                
                file_size = len(response.content)
                print(f"Successfully downloaded image: {file_size} bytes")
                
                # Prepare success result
                result = {
                    'chemical_name': chemical_name,
                    'output_filename': output_filename,
                    'image_size': image_size,
                    'file_size_bytes': file_size,
                    'success': True,
                    'url_used': url
                }
                
                # Write result to JSON file as per user preference
                output_json_filename = "chemical_image_download_result.json"
                with open(output_json_filename, 'w') as f:
                    json.dump(result, f, indent=2)
                
                return result
                
            else:
                error_msg = f"Failed to download image: HTTP {response.status_code}"
                if response.text:
                    error_msg += f", Response: {response.text[:200]}..."
                
                print(error_msg)
                
                error_result = {
                    'chemical_name': chemical_name,
                    'output_filename': output_filename,
                    'image_size': image_size,
                    'file_size_bytes': 0,
                    'success': False,
                    'error': error_msg,
                    'url_used': url
                }
                
                # Write error to JSON file
                output_json_filename = "chemical_image_download_error.json"
                with open(output_json_filename, 'w') as f:
                    json.dump(error_result, f, indent=2)
                
                return None
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error during image download: {str(e)}"
            print(error_msg)
            error_result = {
                'chemical_name': chemical_name,
                'output_filename': output_filename or f"{chemical_name}.png",
                'image_size': image_size,
                'file_size_bytes': 0,
                'success': False,
                'error': error_msg
            }
            
            # Write error to JSON file
            output_json_filename = "chemical_image_download_error.json"
            with open(output_json_filename, 'w') as f:
                json.dump(error_result, f, indent=2)
            
            return None
        except Exception as e:
            error_msg = f"Unexpected error during image download: {str(e)}"
            print(error_msg)
            error_result = {
                'chemical_name': chemical_name,
                'output_filename': output_filename or f"{chemical_name}.png",
                'image_size': image_size,
                'file_size_bytes': 0,
                'success': False,
                'error': error_msg
            }
            
            # Write error to JSON file
            output_json_filename = "chemical_image_download_error.json"
            with open(output_json_filename, 'w') as f:
                json.dump(error_result, f, indent=2)
            
            return None

    def download_chemical_image_by_formula(self, chemical_formula, image_size="large", output_filename=None):
        """
        Downloads a chemical structure image from PubChem using the chemical formula.

        This method uses PubChem's PUG-REST API to first find the compound ID (CID)
        for the given chemical formula, then downloads a PNG image of the chemical
        structure. The result metadata is saved to a JSON file.

        Args:
            chemical_formula (str): The chemical formula of the compound (e.g., "C6H12O6", "H2O").
                Must be case-sensitive and without spaces.
            image_size (str, optional): Size of the image. Can be "small", "large", 
                or specific dimensions like "300x300". Defaults to "large".
            output_filename (str, optional): Name of the output PNG file. If not provided,
                will use the chemical formula with .png extension.

        Returns:
            str or None: The file path of the downloaded image if successful, or None if an error occurs.
                Metadata is also written to JSON file.

        Example:
            filepath = predictor.download_chemical_image_by_formula("C6H12O6")
            if filepath:
                print(f"Downloaded glucose structure to {filepath}")
        """
        try:
            # URL encode the chemical formula for safe API calls
            encoded_formula = quote(chemical_formula)
            
            # Generate output filename if not provided
            if output_filename is None:
                # Clean the formula for use as filename
                safe_formula = "".join(c for c in chemical_formula if c.isalnum())
                output_filename = f"{safe_formula}.png"
            
            print(f"Downloading chemical structure image for formula: {chemical_formula}")
            
            # Step 1: Get CID(s) for the chemical formula
            cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastformula/{encoded_formula}/cids/TXT"
            
            print(f"Step 1: Getting CID for formula {chemical_formula}")
            print(f"CID API URL: {cid_url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/plain'
            }
            
            cid_response = requests.get(cid_url, headers=headers, timeout=30)
            print(f"CID response status: {cid_response.status_code}")
            
            if cid_response.status_code != 200:
                error_msg = f"Failed to get CID for formula {chemical_formula}: HTTP {cid_response.status_code}"
                print(error_msg)
                
                error_result = {
                    'chemical_formula': chemical_formula,
                    'output_filename': output_filename,
                    'image_size': image_size,
                    'success': False,
                    'error': error_msg,
                    'step_failed': 'cid_lookup'
                }
                
                # Write error to JSON file
                output_json_filename = "chemical_formula_image_download_error.json"
                with open(output_json_filename, 'w') as f:
                    json.dump(error_result, f, indent=2)
                
                return None
            
            # Parse CIDs from response (text format, one CID per line)
            cid_text = cid_response.text.strip()
            if not cid_text:
                error_msg = f"No CIDs found for formula: {chemical_formula}"
                print(error_msg)
                
                error_result = {
                    'chemical_formula': chemical_formula,
                    'output_filename': output_filename,
                    'image_size': image_size,
                    'success': False,
                    'error': error_msg,
                    'step_failed': 'no_cids_found'
                }
                
                # Write error to JSON file
                output_json_filename = "chemical_formula_image_download_error.json"
                with open(output_json_filename, 'w') as f:
                    json.dump(error_result, f, indent=2)
                
                return None
            
            # Get the first CID (most relevant)
            cids = cid_text.split('\n')
            first_cid = cids[0].strip()
            print(f"Found {len(cids)} CID(s). Using first CID: {first_cid}")
            
            # Step 2: Download the image using the CID
            if image_size in ["small", "large"]:
                image_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{first_cid}/PNG?image_size={image_size}"
            else:
                # Assume it's specific dimensions like "300x300"
                image_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{first_cid}/PNG?image_size={image_size}"
            
            print(f"Step 2: Downloading image using CID {first_cid}")
            print(f"Image API URL: {image_url}")
            print(f"Output file: {output_filename}")
            
            # Make the image request
            image_headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'image/png'
            }
            
            image_response = requests.get(image_url, headers=image_headers, timeout=30)
            print(f"Image response status: {image_response.status_code}")
            
            if image_response.status_code == 200 and image_response.content:
                # Save the image to file
                with open(output_filename, 'wb') as f:
                    f.write(image_response.content)
                
                file_size = len(image_response.content)
                print(f"Successfully downloaded image: {file_size} bytes")
                
                # Prepare success result
                result = {
                    'chemical_formula': chemical_formula,
                    'cid_used': first_cid,
                    'total_cids_found': len(cids),
                    'output_filename': output_filename,
                    'image_size': image_size,
                    'file_size_bytes': file_size,
                    'success': True,
                    'cid_url_used': cid_url,
                    'image_url_used': image_url
                }
                
                # Write result to JSON file as per user preference
                output_json_filename = "chemical_formula_image_download_result.json"
                with open(output_json_filename, 'w') as f:
                    json.dump(result, f, indent=2)
                
                return output_filename  # Return the file path
                
            else:
                error_msg = f"Failed to download image for CID {first_cid}: HTTP {image_response.status_code}"
                if image_response.text:
                    error_msg += f", Response: {image_response.text[:200]}..."
                
                print(error_msg)
                
                error_result = {
                    'chemical_formula': chemical_formula,
                    'cid_used': first_cid,
                    'output_filename': output_filename,
                    'image_size': image_size,
                    'success': False,
                    'error': error_msg,
                    'step_failed': 'image_download',
                    'cid_url_used': cid_url,
                    'image_url_used': image_url
                }
                
                # Write error to JSON file
                output_json_filename = "chemical_formula_image_download_error.json"
                with open(output_json_filename, 'w') as f:
                    json.dump(error_result, f, indent=2)
                
                return None
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error during formula image download: {str(e)}"
            print(error_msg)
            error_result = {
                'chemical_formula': chemical_formula,
                'output_filename': output_filename or f"{chemical_formula}.png",
                'image_size': image_size,
                'success': False,
                'error': error_msg,
                'step_failed': 'network_error'
            }
            
            # Write error to JSON file
            output_json_filename = "chemical_formula_image_download_error.json"
            with open(output_json_filename, 'w') as f:
                json.dump(error_result, f, indent=2)
            
            return None
        except Exception as e:
            error_msg = f"Unexpected error during formula image download: {str(e)}"
            print(error_msg)
            error_result = {
                'chemical_formula': chemical_formula,
                'output_filename': output_filename or f"{chemical_formula}.png",
                'image_size': image_size,
                'success': False,
                'error': error_msg,
                'step_failed': 'unexpected_error'
            }
            
            # Write error to JSON file
            output_json_filename = "chemical_formula_image_download_error.json"
            with open(output_json_filename, 'w') as f:
                json.dump(error_result, f, indent=2)
            
            return None

    def create_reaction_diagram(self, reactants, reagents=None, products=None, image_size="large", output_filename=None):
        """
        Creates a visual reaction diagram by downloading and combining chemical structure images.

        This method downloads structure images for all reactants, reagents, and products using
        their chemical formulas, then combines them into a single image showing the complete
        reaction with "+" symbols between compounds and "→" for the reaction arrow.

        Args:
            reactants (list): List of chemical formulas for reactants (e.g., ["H2", "O2"]).
            reagents (list, optional): List of chemical formulas for reagents. Can be None or empty.
            products (list, optional): List of chemical formulas for products (e.g., ["H2O"]).
            image_size (str, optional): Size for individual chemical images. Defaults to "large".
            output_filename (str, optional): Name of the output combined image file. If not provided,
                will generate a descriptive filename.

        Returns:
            str or None: The file path of the created reaction diagram if successful, or None if an error occurs.
                Metadata is also written to JSON file.

        Example:
            filepath = predictor.create_reaction_diagram(
                reactants=["H2", "O2"], 
                products=["H2O"]
            )
            if filepath:
                print(f"Reaction diagram saved to: {filepath}")
        """
        try:
            # Validate inputs
            if not reactants:
                raise ValueError("At least one reactant must be provided")
            
            if not products:
                products = []
            
            if reagents is None:
                reagents = []
            
            print(f"Creating reaction diagram:")
            print(f"  Reactants: {reactants}")
            print(f"  Reagents: {reagents}")
            print(f"  Products: {products}")
            
            # Generate output filename if not provided
            if output_filename is None:
                reactant_str = "_".join([formula.replace("(", "").replace(")", "") for formula in reactants])
                product_str = "_".join([formula.replace("(", "").replace(")", "") for formula in products]) if products else "unknown"
                output_filename = f"reaction_{reactant_str}_to_{product_str}.png"
            
            # Download all individual images
            downloaded_images = {}
            failed_downloads = []
            
            # Download reactant images
            for formula in reactants:
                print(f"Downloading reactant: {formula}")
                filepath = self.download_chemical_image_by_formula(formula, image_size)
                if filepath:
                    downloaded_images[f"reactant_{formula}"] = filepath
                else:
                    failed_downloads.append(f"reactant_{formula}")
            
            # Download reagent images
            for formula in reagents:
                print(f"Downloading reagent: {formula}")
                filepath = self.download_chemical_image_by_formula(formula, image_size)
                if filepath:
                    downloaded_images[f"reagent_{formula}"] = filepath
                else:
                    failed_downloads.append(f"reagent_{formula}")
            
            # Download product images
            for formula in products:
                print(f"Downloading product: {formula}")
                filepath = self.download_chemical_image_by_formula(formula, image_size)
                if filepath:
                    downloaded_images[f"product_{formula}"] = filepath
                else:
                    failed_downloads.append(f"product_{formula}")
            
            if len(downloaded_images) == 0:
                error_msg = "No images could be downloaded for any compounds"
                print(error_msg)
                
                error_result = {
                    'reactants': reactants,
                    'reagents': reagents,
                    'products': products,
                    'output_filename': output_filename,
                    'success': False,
                    'error': error_msg,
                    'failed_downloads': failed_downloads
                }
                
                # Write error to JSON file
                output_json_filename = "reaction_diagram_error.json"
                with open(output_json_filename, 'w') as f:
                    json.dump(error_result, f, indent=2)
                
                return None
            
            print(f"Successfully downloaded {len(downloaded_images)} images")
            if failed_downloads:
                print(f"Failed to download: {failed_downloads}")
            
            # Load and process images
            images = {}
            max_height = 0
            
            for key, filepath in downloaded_images.items():
                try:
                    img = Image.open(filepath)
                    # Check if image is valid and not empty
                    if img.width > 0 and img.height > 0:
                        images[key] = img
                        max_height = max(max_height, img.height)
                        print(f"Successfully loaded image: {key} ({img.width}x{img.height})")
                    else:
                        print(f"Skipping empty image: {filepath}")
                        failed_downloads.append(key)
                except Exception as e:
                    print(f"Failed to load image {filepath}: {e}")
                    failed_downloads.append(key)
            
            if len(images) == 0:
                error_msg = "No valid images could be loaded"
                print(error_msg)
                
                error_result = {
                    'reactants': reactants,
                    'reagents': reagents,
                    'products': products,
                    'output_filename': output_filename,
                    'success': False,
                    'error': error_msg,
                    'failed_downloads': failed_downloads
                }
                
                # Write error to JSON file
                output_json_filename = "reaction_diagram_error.json"
                with open(output_json_filename, 'w') as f:
                    json.dump(error_result, f, indent=2)
                
                return None
            
            # Calculate dimensions for combined image
            margin = 20
            symbol_width = 60
            arrow_width = 100
            
            # Organize images by type, only including successfully loaded images
            reactant_images = []
            reagent_images = []
            product_images = []
            
            # Filter to only include non-empty, successfully loaded images
            for key, img in images.items():
                if img is not None and img.width > 0 and img.height > 0:
                    if key.startswith("reactant_"):
                        reactant_images.append(img)
                    elif key.startswith("reagent_"):
                        reagent_images.append(img)
                    elif key.startswith("product_"):
                        product_images.append(img)
            
            print(f"Valid images found: {len(reactant_images)} reactants, {len(reagent_images)} reagents, {len(product_images)} products")
            
            # Calculate total width
            total_width = margin * 2  # Start and end margins
            
            # Reactants width
            if reactant_images:
                total_width += sum(img.width for img in reactant_images)
                total_width += symbol_width * (len(reactant_images) - 1)  # "+" symbols between reactants
            
            # Add reagents if any
            if reagent_images:
                if reactant_images:
                    total_width += symbol_width  # "+" before reagents
                total_width += sum(img.width for img in reagent_images)
                total_width += symbol_width * (len(reagent_images) - 1)  # "+" symbols between reagents
            
            # Arrow (only add if we have both reactants/reagents AND products)
            if product_images and (reactant_images or reagent_images):
                total_width += arrow_width  # "→" arrow
            
            # Products width
            if product_images:
                total_width += sum(img.width for img in product_images)
                total_width += symbol_width * (len(product_images) - 1)  # "+" symbols between products
            
            # Create the combined image with light grey background to match PubChem chemical structure images
            combined_height = max_height + margin * 2
            background_color = (248, 248, 248)  # Light grey to match actual PubChem image backgrounds
            combined_image = Image.new('RGB', (total_width, combined_height), background_color)
            draw = ImageDraw.Draw(combined_image)
            
            # Try to use a default font, fall back to default if not available
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 36)
            except:
                try:
                    font = ImageFont.truetype("arial.ttf", 36)
                except:
                    font = ImageFont.load_default()
            
            # Start positioning images
            x_offset = margin
            y_center = combined_height // 2
            
            # Place reactant images
            for i, img in enumerate(reactant_images):
                y_offset = y_center - img.height // 2
                combined_image.paste(img, (x_offset, y_offset))
                x_offset += img.width
                
                # Add "+" symbol if not the last reactant or if there are reagents
                if i < len(reactant_images) - 1 or reagent_images:
                    text_x = x_offset + symbol_width // 4
                    text_y = y_center - 18
                    draw.text((text_x, text_y), "+", fill='black', font=font)
                    x_offset += symbol_width
            
            # Place reagent images
            for i, img in enumerate(reagent_images):
                y_offset = y_center - img.height // 2
                combined_image.paste(img, (x_offset, y_offset))
                x_offset += img.width
                
                # Add "+" symbol if not the last reagent
                if i < len(reagent_images) - 1:
                    text_x = x_offset + symbol_width // 4
                    text_y = y_center - 18
                    draw.text((text_x, text_y), "+", fill='black', font=font)
                    x_offset += symbol_width
            
            # Add reaction arrow if there are products and we had reactants/reagents
            if product_images and (reactant_images or reagent_images):
                arrow_x = x_offset + arrow_width // 4
                arrow_y = y_center - 18
                draw.text((arrow_x, arrow_y), "→", fill='black', font=font)  # Proper arrow character
                x_offset += arrow_width
                
                # Place product images
                for i, img in enumerate(product_images):
                    y_offset = y_center - img.height // 2
                    combined_image.paste(img, (x_offset, y_offset))
                    x_offset += img.width
                    
                    # Add "+" symbol if not the last product
                    if i < len(product_images) - 1:
                        text_x = x_offset + symbol_width // 4
                        text_y = y_center - 18
                        draw.text((text_x, text_y), "+", fill='black', font=font)
                        x_offset += symbol_width
            
            # Save the combined image
            combined_image.save(output_filename)
            file_size = os.path.getsize(output_filename)
            
            print(f"Successfully created reaction diagram: {output_filename} ({file_size} bytes)")
            
            # Prepare success result
            result = {
                'reactants': reactants,
                'reagents': reagents,
                'products': products,
                'output_filename': output_filename,
                'image_size': image_size,
                'file_size_bytes': file_size,
                'success': True,
                'images_downloaded': len(downloaded_images),
                'failed_downloads': failed_downloads,
                'downloaded_files': list(downloaded_images.values())
            }
            
            # Write result to JSON file as per user preference
            output_json_filename = "reaction_diagram_result.json"
            with open(output_json_filename, 'w') as f:
                json.dump(result, f, indent=2)
            
            return output_filename  # Return the file path
            
        except Exception as e:
            error_msg = f"Unexpected error during reaction diagram creation: {str(e)}"
            print(error_msg)
            error_result = {
                'reactants': reactants,
                'reagents': reagents or [],
                'products': products or [],
                'output_filename': output_filename or "reaction_diagram.png",
                'success': False,
                'error': error_msg
            }
            
            # Write error to JSON file
            output_json_filename = "reaction_diagram_error.json"
            with open(output_json_filename, 'w') as f:
                json.dump(error_result, f, indent=2)
            
            return None

    def single_formula_to_smiles(self, formula):
        smiles = self.search_pubchem_by_formula(formula, 1)
        if smiles:
            return smiles[0][2]
        else:
            # Fallback to Cactus NCI service
            try:
                url = f"http://cactus.nci.nih.gov/chemical/structure/{formula}/smiles"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    cactus_smiles = response.text.strip()
                    if cactus_smiles and cactus_smiles != "Not found":
                        print(f"Found SMILES via Cactus NCI fallback for formula: {formula}")
                        return cactus_smiles
            except Exception as e:
                print(f"Error calling Cactus NCI service for formula {formula}: {e}")
            
            print(f"No matching molecule found for output formula: {formula}")
            return None

if __name__ == "__main__":
    import argparse
    import os
    
    # Get environment variables (for Railway deployment)
    default_host = os.getenv("HOST", "0.0.0.0")
    default_port = int(os.getenv("PORT", "8000"))
    
    parser = argparse.ArgumentParser(description="AlphaReact Retrosynthesis API")
    parser.add_argument("--mode", choices=["api", "test"], default="api", 
                       help="Run mode: 'api' for FastAPI server, 'test' for original test")
    parser.add_argument("--host", default=default_host, help="Host to bind the server")
    parser.add_argument("--port", type=int, default=default_port, help="Port to bind the server")
    parser.add_argument("--target-smiles", default="O=C(Cc1ccccc1)NC1C(=O)N2C1SC(C2C(=O)O)(C)C",
                       help="Target SMILES for test mode")
    
    args = parser.parse_args()
    
    # Clear screen (Windows compatible) - only in development
    if os.getenv("ENVIRONMENT") != "production":
        import platform
        if platform.system() == "Windows":
            os.system('cls')
        else:
            os.system('clear')
    
    if args.mode == "api":
        print("🚀 Starting AlphaReact Retrosynthesis API...")
        print(f"📡 Server will be available at: http://{args.host}:{args.port}")
        print("📚 API Documentation: http://localhost:8000/docs")
        print("🔍 Health Check: http://localhost:8000/health")
        print("\n" + "="*50)
        
        uvicorn.run(app, host=args.host, port=args.port)
        
    elif args.mode == "test":
        # Original test functionality
        predictor = ReactionPredictor()
        target_smiles = args.target_smiles
        
        print(f"Finding retrosynthetic routes for: {target_smiles}")
        print("-" * 70)
        
        # Find retrosynthetic pathways for the target molecule
        flat_routes = predictor.find_pathways_to_smiles(target_smiles)
        
        if not flat_routes:
            print("No retrosynthetic routes found for this molecule.")
            print("This could be because:")
            print("1. The molecule is too complex for the available models")
            print("2. AiZynthFinder models are not properly configured")
            print("3. The SMILES string might be invalid")
            import sys
            sys.exit()
        
        print(f"Found {len(flat_routes)} potential retrosynthetic routes")
        
        # Process and save each route
        for idx, route in enumerate(flat_routes):
            # Process the route to get readable format
            processed_route = predictor.process_any_route(route)
            
            # Create the final route structure
            final_route = {
                "route": processed_route,
                "target_molecule": target_smiles
            }
            
            # Save to JSON file
            filename = f"{target_smiles.replace('/', '_').replace('=', '_').replace('(', '_').replace(')', '_')}_{idx}.json"
            # Clean filename further for Windows compatibility
            filename = filename.replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
            
            with open(filename, "w") as f:
                json.dump(final_route, f, indent=2)
            
            print(f"\nRoute {idx + 1}:")
            print(f"Saved to: {filename}")
            print(json.dumps(final_route, indent=2))
            print("-" * 50)
        
        import sys
        sys.exit()