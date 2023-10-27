from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pandas as pd

class AccessoriesRecommendation:
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        self.vectorizer = TfidfVectorizer()
        self.train_vectors = self.vectorizer.fit_transform(self.train_y)
        
    def recommend_phone(self, phone):
        phone = phone.lower().replace("_", " ").replace("-", " ")
        phone_tokens = re.findall(r'\w+', phone)
        most_similar_index = -1
        max_similarity = 0
        
        for idx, train_phone in enumerate(self.train_x):
            train_tokens = re.findall(r'\w+', train_phone)
            similarity = cosine_similarity(self.vectorizer.transform([phone]), self.vectorizer.transform([train_phone]))[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_index = idx
        
        if most_similar_index == -1:
            return []
        
        most_similar_accessories = self.train_y[most_similar_index]
        similarities = cosine_similarity(self.train_vectors[most_similar_index], self.train_vectors).flatten()
        similar_indices = similarities.argsort()[::1]
        relevant_accessories = [self.train_y[i] for i in similar_indices if self.train_y[i] == most_similar_accessories][:4]
        return relevant_accessories
    



