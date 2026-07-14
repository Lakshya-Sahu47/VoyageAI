import pandas as pd
import numpy as np
from utils import Util
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
from IPython.display import display

class RBM(object):
    '''
    Class definition for a simple RBM using NumPy
    '''
    def __init__(self, alpha, H, num_vis):
        self.alpha = alpha
        self.num_hid = H
        self.num_vis = num_vis
        self.errors = []
        self.energy_train = []
        self.energy_valid = []

    def training(self, train, valid, user, epochs, batchsize, free_energy, verbose, filename):
        '''
        Function where RBM training takes place
        '''
        prv_w = np.random.normal(loc=0, scale=0.01, size=[self.num_vis, self.num_hid]).astype(np.float32)
        prv_vb = np.zeros([self.num_vis], np.float32)
        prv_hb = np.zeros([self.num_hid], np.float32)

        print("Running NumPy-based RBM session")
        print("Training RBM with {0} epochs and batch size: {1}".format(epochs, batchsize))
        print("Starting the training process")
        util = Util()
        
        train_arr = np.array(train, dtype=np.float32)

        for i in range(epochs):
            for start, end in zip(range(0, len(train), batchsize), range(batchsize, len(train), batchsize)):
                batch = train_arr[start:end]
                
                # Phase 1: Input Processing (Visible -> Hidden)
                h0_prob = 1.0 / (1.0 + np.exp(-(np.dot(batch, prv_w) + prv_hb)))
                h0 = (h0_prob > np.random.uniform(size=h0_prob.shape)).astype(np.float32)
                
                # Phase 2: Reconstruction (Hidden -> Visible -> Hidden)
                v1_prob = 1.0 / (1.0 + np.exp(-(np.dot(h0, prv_w.T) + prv_vb)))
                v1 = (v1_prob > np.random.uniform(size=v1_prob.shape)).astype(np.float32)
                h1 = 1.0 / (1.0 + np.exp(-(np.dot(v1, prv_w) + prv_hb)))
                
                # Contrastive Divergence gradient
                CD = (np.dot(batch.T, h0) - np.dot(v1.T, h1)) / float(batch.shape[0])
                
                # Update weights and biases
                prv_w += self.alpha * CD
                prv_vb += self.alpha * np.mean(batch - v1, axis=0)
                prv_hb += self.alpha * np.mean(h0 - h1, axis=0)

            # Error calculation for this epoch (Mean Squared Error)
            h0_prob_all = 1.0 / (1.0 + np.exp(-(np.dot(train_arr, prv_w) + prv_hb)))
            h0_all = (h0_prob_all > np.random.uniform(size=h0_prob_all.shape)).astype(np.float32)
            v1_prob_all = 1.0 / (1.0 + np.exp(-(np.dot(h0_all, prv_w.T) + prv_vb)))
            v1_all = (v1_prob_all > np.random.uniform(size=v1_prob_all.shape)).astype(np.float32)
            
            err = train_arr - v1_all
            err_sum = np.mean(err * err)
            self.errors.append(err_sum)

            if valid:
                etrain = np.mean(util.free_energy(train_arr, prv_w, prv_vb, prv_hb))
                self.energy_train.append(etrain)
                valid_arr = np.array(valid, dtype=np.float32)
                evalid = np.mean(util.free_energy(valid_arr, prv_w, prv_vb, prv_hb))
                self.energy_valid.append(evalid)

            if verbose:
                print("Error after {0} epochs is: {1}".format(i+1, self.errors[i]))
            elif i % 10 == 9:
                print("Error after {0} epochs is: {1}".format(i+1, self.errors[i]))

        if not os.path.exists('models/rbm_models'):
            os.makedirs('models/rbm_models', exist_ok=True)
        filename = 'models/rbm_models/'+filename
        if not os.path.exists(filename):
            os.mkdir(filename)
        np.save(filename+'/w.npy', prv_w)
        np.save(filename+'/vb.npy', prv_vb)
        np.save(filename+'/hb.npy', prv_hb)
        
        if free_energy:
            print("Exporting free energy plot")
            self.export_free_energy_plot(filename)
        print("Exporting errors vs epochs plot")
        self.export_errors_plot(filename)
        
        # Feeding in the User and Reconstructing the input
        inputUser = np.array([train[user]], dtype=np.float32)
        feed = 1.0 / (1.0 + np.exp(-(np.dot(inputUser, prv_w) + prv_hb)))
        rec = 1.0 / (1.0 + np.exp(-(np.dot(feed, prv_w.T) + prv_vb)))
        return rec, prv_w, prv_vb, prv_hb

    def load_predict(self, filename, train, user):
        prv_w = np.load('models/rbm_models/'+filename+'/w.npy')
        prv_vb = np.load('models/rbm_models/'+filename+'/vb.npy')
        prv_hb = np.load('models/rbm_models/'+filename+'/hb.npy')
        
        print("Model restored from " + filename)
        
        inputUser = np.array([train[user]], dtype=np.float32)
        feed = 1.0 / (1.0 + np.exp(-(np.dot(inputUser, prv_w) + prv_hb)))
        rec = 1.0 / (1.0 + np.exp(-(np.dot(feed, prv_w.T) + prv_vb)))
        
        return rec, prv_w, prv_vb, prv_hb
        
    def calculate_scores(self, ratings, attractions, rec, user, rbm_att):
        '''
        Function to obtain recommendation scores for a user
        using the trained weights and rbm_att mapping
        '''
        # Find all attraction IDs the user has visited
        visited_places = ratings[ratings['user_id'] == user]['attraction_id'].unique().tolist()

        # Get details of visited places
        places_names = []
        places_categories = []
        places_prices = []
        for place in visited_places:
            names = attractions[attractions['attraction_id'] == place]['name'].tolist()
            cats = attractions[attractions['attraction_id'] == place]['category'].tolist()
            prices = attractions[attractions['attraction_id'] == place]['price'].tolist()
            places_names.append(names[0] if names else "Unknown")
            places_categories.append(cats[0] if cats else "Unknown")
            places_prices.append(prices[0] if prices else 0.0)

        seen_places = pd.DataFrame({
            'att_id': visited_places,
            'att_name': places_names,
            'att_cat': places_categories,
            'att_price': places_prices
        })

        # rec[0] contains recommendation scores for unique attractions
        unique_mapping = rbm_att[['attraction_id', 'rbm_att_id']].drop_duplicates()
        
        scores_df = pd.DataFrame({
            'rbm_att_id': np.arange(len(rec[0])),
            'Recommendation Score': rec[0]
        })
        
        attraction_scores = unique_mapping.merge(scores_df, on='rbm_att_id')
        
        # Filter out visited places
        unseen_scores = attraction_scores[~attraction_scores['attraction_id'].isin(visited_places)]
        
        # Merge with attraction details
        unseen_places = unseen_scores.merge(attractions[['attraction_id', 'name', 'category', 'price']], on='attraction_id')
        unseen_places = unseen_places.rename(columns={
            'attraction_id': 'att_id',
            'name': 'att_name',
            'category': 'att_cat',
            'price': 'att_price',
            'Recommendation Score': 'score'
        })
        
        unseen_places = unseen_places[['att_id', 'att_name', 'att_cat', 'att_price', 'score']]
        unseen_places = unseen_places.groupby(['att_id', 'att_name', 'att_cat', 'att_price'], as_index=False)['score'].max()
        
        # Display the head of the unseen scores for jupyter notebook visualization
        grouped_unseen = unseen_places[['att_id', 'score']].rename(columns={'att_id': 'attraction_id', 'score': 'Recommendation Score'})
        display(grouped_unseen.head())
        
        return unseen_places, seen_places

    def export(self, unseen, seen, filename, user):
        '''
        Function to export the final result for a user into csv format
        '''
        sorted_result = unseen.sort_values(by='score', ascending=False)
        
        if len(sorted_result) > 0:
            x = sorted_result[['score']].values.astype(float)
            min_max_scaler = preprocessing.MinMaxScaler((0, 5))
            x_scaled = min_max_scaler.fit_transform(x)
            sorted_result['score'] = x_scaled
        
        if not os.path.exists(filename):
            os.makedirs(filename, exist_ok=True)
            
        seen.to_csv(filename+'/user'+user+'_seen.csv')
        sorted_result.to_csv(filename+'/user'+user+'_unseen.csv')

    def export_errors_plot(self, filename):
        plt.figure()
        plt.plot(self.errors)
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.savefig(filename+"/error.png")
        plt.close()

    def export_free_energy_plot(self, filename):
        fig, ax = plt.subplots()
        ax.plot(self.energy_train, label='train')
        ax.plot(self.energy_valid, label='valid')
        leg = ax.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Free Energy")
        plt.savefig(filename+"/free_energy.png")
        plt.close()
