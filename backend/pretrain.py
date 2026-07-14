import sys
import os
import pandas as pd

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from attractions_recc import get_recc

def main():
    print("Starting pre-training script for VoyageAI RBM Recommender...")
    
    # Define default category ratings
    default_cat_rating = {
        'private_&_custom_tours': 5.0,
        'tours_&_sightseeing': 5.0,
        'cultural_&_theme_tours': 5.0,
        'sightseeing_tickets_&_passes': 5.0,
        'multi-day_&_extended_tours': 5.0
    }
    
    # Load attractions database
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    att_json_path = os.path.join(base_dir, 'data', 'etl', 'attractions.json')
    
    if not os.path.exists(att_json_path):
        print(f"Error: attractions.json not found at {att_json_path}")
        sys.exit(1)
        
    att_df = pd.read_json(att_json_path, orient='records')
    
    print("Training RBM model on the dataset...")
    # This will trigger training and export the model files to models/rbm_models/
    filename, user, rbm_att = get_recc(att_df, default_cat_rating)
    print(f"Pre-training complete! Model saved as models/rbm_models/{filename}")

if __name__ == '__main__':
    main()
