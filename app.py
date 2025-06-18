import sys
from pathlib import Path  # 定位当前文件和项目根目录

# 获取当前文件（app.py）的绝对路径
current_file = Path(__file__)
# 定位到项目根目录（假设app.py在src/web下，根目录是上级的上级）
root_dir = current_file.parent.parent.parent

# 将根目录加入Python模块搜索路径
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import streamlit as st  # 构建 Web 应用
import pandas as pd  # 用于数据处理
import numpy as np  # 用于数据处理
from src.models.recommendation_model import load_model, get_recommendations  # 自定义函数


# 评分展示函数
def display_rating(rating):
    """将10分制评分转换为五星制并显示评分+星星"""
    if pd.isna(rating):
        return "暂无评分"

    # 计算五星制（四舍五入到最接近的0.5）
    stars = round(rating / 2, 1)
    full_stars = int(stars)
    has_half_star = (stars % 1) >= 0.5
    empty_stars = 5 - full_stars - (1 if has_half_star else 0)

    # 构建星星HTML（使用½字符表示半颗星）
    star_html = (
            f"<span class='star-icon'>★</span>" * full_stars +
            (f"<span class='star-icon'>½</span>" if has_half_star else "") +
            f"<span class='empty-star-icon'>☆</span>" * empty_stars
    )

    # 返回评分数字 + 星星（保持原始评分的10分制）
    return f"<span class='rating-number'>{rating:.2f}</span> <span class='star-rating'>{star_html}</span>"


# 顶部导航栏设置 渲染顶部导航栏，显示系统名称和数据库中的电影总数。
def render_top_navigation(total_movies):
    st.markdown(f"""
    <div class="top-nav">
        <div class="logo-container" style="justify-content: center; flex-direction: column; align-items: center;">
            <span class="logo-text" style="font-size: 2.5rem; text-align: center;">基于内容的电影推荐系统</span>
            <span class="nav-stats" style="font-size: 1.3rem; margin-top: 0.5rem;">（当前数据库电影总数: {total_movies}🎬）</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# 页面配置
st.set_page_config(
    page_title="个性化电影推荐系统",
    page_icon="🎬",
    layout="wide"
)

# 自定义样式
st.markdown("""
<style>
    .top-nav {
        background-color: #1e1e1e;
        padding: 1.5rem;
        margin: -1.5rem -1.5rem 1.5rem -1.5rem;
    }
    .logo-container {
        display: flex;
        justify-content: center; 
        align-items: center;
        flex-direction: column; /* 垂直排列 */
    }
    .logo-text {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .nav-stats {
        color: #F5C518; /* 改为黄色更醒目 */
        font-size: 1.3rem;
        text-align: right; /* 文本右对齐 */
    }
    .sort-container {
        margin: 1.5rem 0;
    }
    .random-button {
        margin-top: 1.5rem;
    }
    .rating-number {
        font-size: 1.2rem;  /* 评分数字字体大小，可调整 */
        vertical-align: middle; /* 与星星垂直居中 */
        margin-right: 5px; /* 数字与星星的间距 */
    }
    .star-icon {
        color: #F5C518;
        font-size: 1.2rem;  /* 星星大小 */
        margin: 0 2px;
        vertical-align: middle;
    }
    .empty-star-icon {
        color: #E0E0E0;
        font-size: 1.2rem;      /* 空白星星大小 */
        margin: 0 2px;
        vertical-align: middle;
    }
    .star-rating {
        white-space: nowrap;
        margin-left: 8px; /* 星星与评分数字的间距 */
    }
    .movie-title {
        font-size: 1.2rem;
        font-weight: bold;
    }
    .movie-info {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# 加载数据  ，实现数据的缓存，避免重复加载
@st.cache_data
def load_data():
    model_data, movie_metadata = load_model()
    return model_data, movie_metadata


model_data, movie_metadata = load_data()

# 初始化session_state 保存用户选择的排序方式
if 'sort_option' not in st.session_state:
    st.session_state.sort_option = "相似度"
    st.session_state.selected_movie = ""

total_movies = len(movie_metadata) if movie_metadata is not None else 0

# 顶部导航栏
render_top_navigation(total_movies)

# 初始化session_state
if 'sort_option' not in st.session_state:
    st.session_state.sort_option = "相似度"
if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = ""

# 侧边栏 - 电影选择（优化索引计算）
current_index = 0
if st.session_state.selected_movie and movie_metadata is not None:
    try:
        current_index = movie_metadata['title'].tolist().index(
            st.session_state.selected_movie
        )
    except ValueError:
        st.warning(f"电影 '{st.session_state.selected_movie}' 不在当前数据集中")
        st.session_state.selected_movie = ""

selected_movie = st.sidebar.selectbox(
    "选择一部电影获取推荐",
    movie_metadata['title'].tolist() if movie_metadata is not None else [],
    index=current_index,
    key="movie_selectbox"
)


# 随机推荐按钮（使用回调函数）
def random_recommendation():
    if movie_metadata is None or movie_metadata.empty:
        st.warning("电影数据为空，无法随机推荐")
    else:
        # st.write(f"数据有效，共有 {len(movie_metadata)} 部电影")
        with st.spinner("随机推荐中..."):
            random_movie = movie_metadata.sample(1).iloc[0]
            st.session_state.selected_movie = random_movie['title']
            st.session_state.sort_option = "相似度"
            # st.write(f"已选择: {st.session_state.selected_movie}")


# 绑定回调函数
if st.sidebar.button("🎲 随机推荐", on_click=random_recommendation, use_container_width=True, key="random_button"):
    pass

# 侧边栏 - 排序模块
st.sidebar.markdown("#### 排序方式")
sort_option = st.sidebar.radio(
    "",
    ["相似度", "评分降序", "评分升序", "时间降序", "时间升序"],
    index=["相似度", "评分降序", "评分升序", "时间降序", "时间升序"].index(
        st.session_state.sort_option
    )
)
st.session_state.sort_option = sort_option


def format_cast(cast):
    """处理主演信息，处理缺失值并格式化显示"""
    if pd.notna(cast):
        if isinstance(cast, str):
            return cast.replace('|', ', ')
        else:
            return '未知'
    return '未知'


# 主内容区
if selected_movie and model_data and movie_metadata is not None:
    # 获取推荐
    recommendations = get_recommendations(
        selected_movie,
        model_data,
        movie_metadata,
        top_n=50  # 设置值，但实际数量取决于模型和数据
    )

    # 应用排序
    if sort_option == "评分降序":
        recommendations = recommendations.sort_values('vote_average', ascending=False)
    elif sort_option == "评分升序":
        recommendations = recommendations.sort_values('vote_average', ascending=True)
    elif sort_option == "时间降序":
        recommendations = recommendations.sort_values('year', ascending=False)
    elif sort_option == "时间升序":
        recommendations = recommendations.sort_values('year', ascending=True)

    if not recommendations.empty:
        # 显示选中的电影
        st.subheader(f"你选择的电影: {selected_movie}")

        # 电影信息展示
        selected_movie_info = movie_metadata[movie_metadata['title'] == selected_movie].iloc[0]

        col1, col2 = st.columns([1, 3])
        with col1:
            poster_path = selected_movie_info.get('poster_path')
            if poster_path:
                poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                st.image(poster_url, use_container_width=True)

        with col2:
            st.write(f"**类型:** {', '.join(selected_movie_info['genres'])}")
            st.write(f"**导演:** {selected_movie_info.get('director', '未知')}")
            st.write(f"**主演:** {format_cast(selected_movie_info.get('top_cast', '未知'))}")
            st.markdown(
                f"**评分:** {display_rating(selected_movie_info.get('vote_average', 0))}",
                unsafe_allow_html=True
            )
            st.write(f"**上映日期:** {selected_movie_info.get('release_date', '未知')}")
            st.write(f"**片长:** {selected_movie_info.get('runtime', '未知')}分钟")
            production_countries = selected_movie_info.get('production_countries', [])
            if isinstance(production_countries, str):
                production_countries = production_countries.split('|')
            if not isinstance(production_countries, (list, tuple)):
                production_countries = []
            if production_countries:
                st.write(f"**国家:** {', '.join(production_countries)}")
            else:
                st.write(f"**国家:** 未知")
            st.write(f"**简介:** {selected_movie_info.get('overview', '暂无简介')}")

        # 显示推荐结果
        total_recs = len(recommendations)
        st.subheader(f"为你推荐的{total_recs}部电影:")

        # 按行展示推荐结果（每行5部电影）
        rows = total_recs // 5 + (1 if total_recs % 5 != 0 else 0)

        for row in range(rows):
            cols = st.columns(5)
            for i in range(5):
                index = row * 5 + i
                if index < total_recs:
                    movie = recommendations.iloc[index]

                    with cols[i]:
                        st.markdown(f"""
                            <p style='font-size:20px; font-weight: bold; text-align: center;'>
                                {movie['title']}
                            </p>
                        """, unsafe_allow_html=True)

                        # 计算相似度百分比
                        similarity_percent = f"{movie['similarity_score'] * 100:.1f}%"

                        poster_path = movie.get('poster_path')
                        if poster_path:
                            poster_url = f"https://image.tmdb.org/t/p/w300{poster_path}"
                            st.image(poster_url, use_container_width=True)

                        st.write(f"**相似度:** {similarity_percent}")
                        st.markdown(
                            f"**评分:** {display_rating(movie['vote_average'])}",
                            unsafe_allow_html=True
                        )
                        st.write(f"**类型:** {', '.join(movie['genres'])}")
                        st.write(f"**上映日期:** {movie.get('release_date', '未知')}")
                        # 添加查看详情按钮
                        # if st.button("查看详情", key=f"details_{index}"):
                        with st.expander(f"展开查看详情"):
                            st.write(f"**导演:** {movie.get('director', '未知')}")
                            st.write(f"**主演:** {format_cast(movie.get('top_cast', '未知'))}")
                            production_countries = movie.get('production_countries', [])
                            if isinstance(production_countries, str):
                                production_countries = production_countries.split('|')  # 处理字符串格式
                            if not isinstance(production_countries, (list, tuple)):
                                production_countries = []
                            if production_countries:
                                st.write(f"**国家:** {', '.join(production_countries)}")
                            else:
                                st.write(f"**国家:** 未知")
                            st.write(f"**片长:** {movie.get('runtime', '未知')}分钟")
                            st.write(f"**简介:** {movie.get('overview', '暂无简介')}")
    else:
        st.warning("未找到推荐结果，请选择其他电影。")
else:
    st.info("请从侧边栏选择一部电影以获取推荐。")

# 页脚信息
st.markdown("---")
st.caption("Author：王悦 wy11010101@outlook.com | 使用TMDB API数据")
