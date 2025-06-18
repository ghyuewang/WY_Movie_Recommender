import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import os  # 导入os模块用于目录操作


# 基于电影的类型、概述、导演、主演和关键词等信息构建相似度矩阵（特征工程）
def build_similarity_matrix(df):
    """基于多种特征构建电影相似度矩阵"""
    # 处理缺失值
    df['overview'] = df['overview'].fillna('')
    df['genres'] = df['genres'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
    df['director'] = df['director'].fillna('')
    df['top_cast'] = df['top_cast'].fillna('')
    df['keywords'] = df['keywords'].fillna('')

    # 特征组合
    df['combined_features'] = (
            df['genres'] + ' ' +
            df['overview'] + ' ' +
            df['director'] + ' ' +
            df['top_cast'] + ' ' +
            df['keywords']
    )

    # 计算TF-IDF矩阵
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

    # 计算余弦相似度
    content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # 考虑评分相似度（归一化评分）
    scaler = MinMaxScaler()
    ratings = df['vote_average'].values.reshape(-1, 1)
    normalized_ratings = scaler.fit_transform(ratings)
    rating_similarity = np.dot(normalized_ratings, normalized_ratings.T)

    # 组合相似度矩阵（内容权重0.8，评分权重0.2）
    similarity_matrix = 0.8 * content_similarity + 0.2 * rating_similarity

    return similarity_matrix

def save_model(similarity_matrix, df):
    """保存模型和电影数据"""
    # 创建保存模型的目录（如果不存在）
    os.makedirs('models', exist_ok=True)

    model_data = {
        'similarity_matrix': similarity_matrix,
        'movie_ids': df['tmdb_id'].tolist(),
        'movie_titles': df['title'].tolist()
    }

    joblib.dump(model_data, 'models/recommendation_model.pkl')

    # 保存电影元数据，包含year列
    df[['tmdb_id', 'title', 'overview', 'genres', 'director', 'top_cast',
        'poster_path', 'vote_average', 'release_date', 'runtime', 'year','production_countries']].to_csv(
        'models/movie_metadata.csv', index=False
    )

def load_model():
    """加载模型和电影数据"""
    try:
        model_data = joblib.load('models/recommendation_model.pkl')
        movie_metadata = pd.read_csv('models/movie_metadata.csv')

        # 将genres列转回列表
        if 'genres' in movie_metadata.columns:
            movie_metadata['genres'] = movie_metadata['genres'].apply(
                lambda x: x.split('|') if isinstance(x, str) else []
            )

        return model_data, movie_metadata
    except FileNotFoundError:
        print("模型文件不存在，请先构建模型！")
        return None, None

def get_recommendations(movie_title, model_data, movie_metadata, top_n=10):
    """获取电影推荐"""
    if model_data is None or movie_metadata is None:
        return []

    similarity_matrix = model_data['similarity_matrix']
    movie_titles = model_data['movie_titles']

    # 查找电影索引
    if movie_title not in movie_titles:
        return []

    movie_index = movie_titles.index(movie_title)

    # 获取相似度得分
    similar_scores = list(enumerate(similarity_matrix[movie_index]))

    # 排序并获取前top_n个
    similar_scores = sorted(similar_scores, key=lambda x: x[1], reverse=True)
    similar_scores = similar_scores[1:top_n + 1]  # 跳过自己

    # 获取推荐电影ID和分数
    recommended_indices = [score[0] for score in similar_scores]
    recommended_scores = [score[1] for score in similar_scores]

    # 获取推荐电影信息
    recommended_movies = movie_metadata.iloc[recommended_indices].copy()
    recommended_movies['similarity_score'] = recommended_scores

    return recommended_movies
