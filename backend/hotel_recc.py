import pandas as pd
import numpy as np
import math, re, glob, os
import pickle

class SimpleMF(object):
    '''
    Simple Matrix Factorization model using Stochastic Gradient Descent (SGD)
    '''
    def __init__(self, rank=8, lr=0.01, reg=0.1, n_epochs=50):
        self.rank = rank
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.users = []
        self.items = []
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.P = None
        self.Q = None

    def fit(self, df):
        self.users = df['user_id'].unique().tolist()
        self.items = df['att_id'].unique().tolist()
        self.user_to_idx = {u: i for i, u in enumerate(self.users)}
        self.item_to_idx = {item: i for i, item in enumerate(self.items)}
        
        n_users = len(self.users)
        n_items = len(self.items)
        
        self.P = np.random.normal(0, 0.1, (n_users, self.rank)).astype(np.float32)
        self.Q = np.random.normal(0, 0.1, (n_items, self.rank)).astype(np.float32)
        
        ratings = []
        for _, row in df.iterrows():
            u_idx = self.user_to_idx[row['user_id']]
            i_idx = self.item_to_idx[row['att_id']]
            ratings.append((u_idx, i_idx, float(row['user_rating'])))
            
        for epoch in range(self.n_epochs):
            for u, i, r in ratings:
                pred = np.dot(self.P[u], self.Q[i])
                err = r - pred
                # SGD update
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * self.P[u] - self.reg * self.Q[i])
                
    def predict(self, user_id, item_id):
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            return 3.0 # Default rating for unseen user/item
        u = self.user_to_idx[user_id]
        i = self.item_to_idx[item_id]
        return float(np.dot(self.P[u], self.Q[i]))

def get_rating(x):
    val = x / 5
    if x >= 0 and x <= val:
        return 1
    elif x > val and x <= 2*val:
        return 2
    elif x > 2*val and x <= 3*val:
        return 3
    elif x > 3*val and x <= 4*val:
        return 4
    else:
        return 5

def load_spark_json(directory):
    files = glob.glob(directory + '/*.json')
    if len(files) == 0:
        if os.path.exists(directory):
            return pd.read_json(directory, lines=True)
        return pd.DataFrame()
    return pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)

def amenities_rating(amenities_pref, newh_df, del_dup):
    '''
    Finds number of amenities present in hotels that user likes
    '''
    pa_df = pd.DataFrame(amenities_pref, columns=["amenities_pref"])
    newa_df = newh_df.merge(pa_df, left_on='amenities', right_on='amenities_pref', how='inner')
    ameni_len = newa_df.groupby('id')['amenities'].count().reset_index()
    ameni_len = ameni_len.rename(columns={'amenities': 'ameni_len'})
    
    ameni_df = del_dup.merge(ameni_len, on='id', how='inner')
    ameni_df = ameni_df.sort_values(by='ameni_len', ascending=False)
    
    find_rating = lambda a: get_rating(a)
    usr_rating = ameni_df.copy()
    usr_rating['rating'] = usr_rating['ameni_len'].apply(find_rating)
    return usr_rating

def model_train(usr_rating):
    '''
    Matrix Factorization collaborative filtering training using NumPy SimpleMF
    '''
    # Read ratings data from data/etl/u_id_df (previously project/u_id_df or etl/u_id_df)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    u_id_df = load_spark_json(os.path.join(base_dir, 'data/etl/u_id_df'))
    uid_count = u_id_df['user_id'].nunique()

    temp_usr = usr_rating[['id', 'rating']].rename(columns={'rating': 'user_rating'})
    usrid_df = temp_usr.merge(u_id_df[['id', 'att_id']].drop_duplicates(), on='id', how='inner')
    usrid_df['user_id'] = uid_count

    usrid_final_df = usrid_df[['user_id', 'att_id', 'user_rating']]
    org_df = u_id_df[['user_id', 'att_id', 'user_rating']]

    usrid_s1 = usrid_final_df.sample(frac=0.1, random_state=42)
    usrid_s2 = usrid_final_df.drop(usrid_s1.index)

    comb_df = pd.concat([org_df, usrid_s1], ignore_index=True)
    
    train_df = comb_df.sample(frac=0.8, random_state=42)
    val_df = comb_df.drop(train_df.index)

    ranks = [4, 8, 12]
    best_model = None
    best_rank = None
    best_error = float('inf')
    errors = []
    
    for r in ranks:
        model = SimpleMF(rank=r, lr=0.05, reg=0.02, n_epochs=100)
        model.fit(train_df)
        
        preds = []
        targets = []
        for _, row in val_df.iterrows():
            pred = model.predict(row['user_id'], row['att_id'])
            preds.append(pred)
            targets.append(row['user_rating'])
            
        rmse = np.sqrt(np.mean((np.array(preds) - np.array(targets)) ** 2)) if len(preds) > 0 else 0.0
        errors.append(rmse)
        
        if rmse < best_error:
            best_error = rmse
            best_rank = r
            best_model = model
            
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_file_dir = os.path.join(base_dir, 'models', 'mf_models', 'model_file')
    os.makedirs(model_file_dir, exist_ok=True)
    with open(os.path.join(model_file_dir, 'mf_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
        
    return best_rank, best_error, errors, usrid_s2

def get_hotel_recc(usrid_s2):
    '''
    Load best SimpleMF model and make recommendations
    '''
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_file_dir = os.path.join(base_dir, 'models', 'mf_models', 'model_file')
    with open(os.path.join(model_file_dir, 'mf_model.pkl'), 'rb') as f:
        model = pickle.load(f)

    user_ids = usrid_s2['user_id'].unique()
    u_id_df = load_spark_json(os.path.join(base_dir, 'data/etl/u_id_df'))
    all_hotel_ids = u_id_df['att_id'].unique()

    recs_list = []
    for uid in user_ids:
        user_preds = []
        for att_id in all_hotel_ids:
            pred = model.predict(uid, att_id)
            user_preds.append((att_id, pred))
        user_preds.sort(key=lambda x: x[1], reverse=True)
        for att_id, rating in user_preds[:50]:
            recs_list.append({'user_id': uid, 'att_id': att_id, 'rating': rating})
            
    recs_df = pd.DataFrame(recs_list)
    u_tempdf = recs_df.merge(u_id_df[['id', 'att_id']].drop_duplicates(), on='att_id', how='inner')
    u_tempdf = u_tempdf[['id']]
    
    return u_tempdf

def get_image(name):
    # Try to find a matching local image folder first
    name_clean = re.sub(' ', '_', name)
    safe_name = name_clean.lower()
    for filename in glob.glob("downloads/{name}/*jpg".format(name=safe_name)) + \
                    glob.glob("downloads/{name}/*png".format(name=safe_name)) + \
                    glob.glob("downloads/{name}/*jpeg".format(name=safe_name)):
        return filename
    for filename in glob.glob("downloads/{name}/*jpg".format(name=name_clean)) + \
                    glob.glob("downloads/{name}/*png".format(name=name_clean)) + \
                    glob.glob("downloads/{name}/*jpeg".format(name=name_clean)):
        return filename
    
    # Fallback to noimage.jpg or any jpg in downloads
    if os.path.exists("downloads/noimage.jpg"):
        return "downloads/noimage.jpg"
    for filename in glob.glob("downloads/*jpg"):
        return filename
    return None
        
def get_hotel_output(days, final):
    import ipywidgets as w
    from IPython.display import display, IFrame

    fields = ['NAME', 'PRICE', 'RATING', 'EXPERIENCE','LOCATION', 'ADDRESS', "AMENITIES"]
    recommendations = ['Recommendation']

    box_layout = w.Layout(justify_content='space-between',
                        display='flex',
                        flex_flow='row', 
                        align_items='stretch',
                       )
    column_layout = w.Layout(justify_content='space-between',
                        width='75%',
                        display='flex',
                        flex_flow='column', 
                       )
    tab = []
    for i in range(len(final['name'])):
        img_path = final['image'][i]
        if img_path and os.path.exists(img_path):
            with open(img_path, "rb") as f:
                image = f.read()
        else:
            image = b""
        name = final['name'][i]
        price = final['price'][i]
        rating = final['rating'][i]
        experience = final['experience'][i]
        loc = final['location'][i]
        address = final['address'][i]
        
        tab.append(w.VBox(children=
                        [
                         w.Image(value=image, format='jpg', width=300, height=400),
                         w.HTML(description=fields[0], value=f"<b><font color='black'>{name}</b>", disabled=True),
                         w.HTML(description=fields[1], value=f"<b><font color='black'>{price}</b>", disabled=True),
                         w.HTML(description=fields[2], value=f"<b><font color='black'>{rating}</b>", disabled=True), 
                         w.HTML(description=fields[3], value=f"<b><font color='black'>{experience}</b>", disabled=True), 
                         w.HTML(description=fields[4], value=f"<b><font color='black'>{loc}</b>", disabled=True),
                         w.HTML(description=fields[5], value=f"<b><font color='black'>{address}</b>", disabled=True)
                        ], layout=column_layout))

    tab_recc = w.Tab(children=tab)
    for i in range(len(tab_recc.children)):
        tab_recc.set_title(i, str('Hotel '+ str(i+1)))
    return tab_recc
