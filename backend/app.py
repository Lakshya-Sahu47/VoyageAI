import sys
import os

# Set current working directory to project root (one level up from this script)
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime as dt
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from attractions_recc import get_recc, filter_df, find_closest, top_recc
from hotel_recc import amenities_rating, model_train, get_hotel_recc, load_spark_json, get_image

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend'))
CORS(app)

# Load data on startup
att_df = pd.read_json('data/etl/attractions.json', orient='records')
del_dup = load_spark_json('data/etl/del_dup')
newh_df = load_spark_json('data/etl/newh_df')

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/downloads/<path:filename>')
def serve_downloads(filename):
    downloads_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'downloads')
    return send_from_directory(downloads_dir, filename)

@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json or {}
        
        # Inputs
        username = data.get('username', 'Guest')
        province = data.get('province', 'Ontario').lower().replace(' ', '_')
        budget_low = float(data.get('budget_low', 0))
        budget_high = float(data.get('budget_high', 1000))
        
        start_date_str = data.get('start_date', '2026-07-15')
        end_date_str = data.get('end_date', '2026-07-18')
        
        start_date = dt.datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = dt.datetime.strptime(end_date_str, '%Y-%m-%d').date()
        days = (end_date - start_date).days + 1
        
        # Category ratings (dict)
        category_ratings = data.get('category_ratings', {})
        # Convert ratings to float
        cat_rating = {k: float(v) for k, v in category_ratings.items()}
        
        # Preferred amenities (list of strings)
        amenities_pref = data.get('amenities_pref', [])
        
        # 1. ATTRACTIONS RECOMMENDATION
        print("Running attractions recommendation model...")
        filename, user, rbm_att = get_recc(att_df, cat_rating)
        with_url = filter_df(filename, user, budget_low, budget_high, province, att_df)
        
        # Build travel plan
        final = {
            'timeofday': [],
            'image': [],
            'name': [],
            'location': [],
            'price': [],
            'rating': [],
            'category': []
        }
        
        # Total slots needed (4 slots per day: Morning 1, Morning 2, Evening 1, Evening 2)
        total_slots = days * 4
        for i in range(days):
            final['timeofday'].append('Morning')
            final['timeofday'].append('Morning')
            final['timeofday'].append('Evening')
            final['timeofday'].append('Evening')
            
        for i in range(total_slots):
            if i % 4 == 0:
                final = top_recc(with_url, final)
            else:
                if len(final['location']) > 0:
                    final = find_closest(with_url, final['location'][-1], final['timeofday'][i], final)
                else:
                    final = top_recc(with_url, final)
                    
        # Format attraction schedule into a day-by-day JSON structure
        attractions_list = []
        for i in range(days):
            day_schedule = {
                'day': i + 1,
                'morning': [],
                'evening': []
            }
            # Morning items (indices i*4 and i*4 + 1)
            for j in range(2):
                idx = i * 4 + j
                if idx < len(final['name']):
                    day_schedule['morning'].append({
                        'name': final['name'][idx],
                        'category': final['category'][idx],
                        'price': final['price'][idx],
                        'rating': final['rating'][idx],
                        'location': final['location'][idx],
                        'image': final['image'][idx]
                    })
            # Evening items (indices i*4 + 2 and i*4 + 3)
            for j in range(2):
                idx = i * 4 + 2 + j
                if idx < len(final['name']):
                    day_schedule['evening'].append({
                        'name': final['name'][idx],
                        'category': final['category'][idx],
                        'price': final['price'][idx],
                        'rating': final['rating'][idx],
                        'location': final['location'][idx],
                        'image': final['image'][idx]
                    })
            attractions_list.append(day_schedule)
            
        # 2. HOTELS RECOMMENDATION
        print("Running hotel recommendation model...")
        usr_rating = amenities_rating(amenities_pref, newh_df, del_dup)
        
        hotels_list = []
        if len(usr_rating) > 0:
            best_rank, best_error, errors, usrid_s2 = model_train(usr_rating)
            u_tempdf = get_hotel_recc(usrid_s2)
            
            # Merge recommendations with hotel details and filter by province/destination location
            hotel_df = del_dup.merge(u_tempdf, on="id", how="inner")
            hotel_df['address'] = hotel_df['address'].str.lower()
            
            # Filter by matching province/location
            # Let's search if address contains the user's destination (province/city)
            hotel_sugg = hotel_df[hotel_df['address'].str.contains(province.replace('_', ' '), na=False)]
            # If no matches in that specific province, fallback to all recommended hotels
            if len(hotel_sugg) == 0:
                hotel_sugg = hotel_df
                
            recc_hotels = hotel_sugg.dropna(subset=['address', 'hotel_name', 'hotel_rating', 'price']).head(5)
            
            for _, row in recc_hotels.iterrows():
                hotels_list.append({
                    'name': row['hotel_name'],
                    'rating': float(row['hotel_rating']) if pd.notnull(row['hotel_rating']) else 0.0,
                    'price': row['price'],
                    'experience': row['hotel_experience'] if pd.notnull(row['hotel_experience']) else 'Good',
                    'address': row['address'],
                    'location': row['location'],
                    'amenities': row['amenities'] if 'amenities' in row and row['amenities'] is not None else '',
                    'image': get_image(row['hotel_name'])
                })
                
        return jsonify({
            'success': True,
            'username': username,
            'province': province.replace('_', ' ').title(),
            'days': days,
            'attractions': attractions_list,
            'hotels': hotels_list
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
