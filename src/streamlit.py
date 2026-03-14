import os
os.sys.path.append(os.path.abspath('app/model/'))
abs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app/model/')

import pandas as pd
import streamlit as st

from utils.model import NCF, __model_version__
from utils.utils import Utils, cols_dict, occupation, genre, css

@st.cache_data
def load_data():
    users_exp = pd.read_csv(abs_path + 'data/users_exp.csv').values
    users_imp = pd.read_csv(abs_path + 'data/users_imp.csv').values
    movies = pd.read_csv(abs_path + 'data/movies.csv')
    movies_og = pd.read_csv(abs_path + 'data/movies.dat', sep='::', names=cols_dict['items'], encoding='latin-1', engine='python')
    ratings = pd.read_csv(abs_path + 'data/ratings.dat', sep='::', names=cols_dict['ratings'], engine='python')

    return users_exp, users_imp, movies, movies_og, ratings

@st.cache_resource
def load_models():
    model_exp = NCF('explicit', gpu=False)
    model_exp.load_weights(abs_path + 'weights/explicit.pth', eval=True)

    model_imp = NCF('implicit', gpu=False)
    model_imp.load_weights(abs_path + 'weights/implicit.pth', eval=True)

    return model_exp, model_imp

users_exp, users_imp, movies, movies_og, ratings = load_data()
model_exp, model_imp = load_models()

# Giao diện
st.title('Hệ thống gợi ý NCF')

st.write(f'Phiên bản mô hình: {__model_version__}')

model_type = st.radio('Chọn loại mô hình', ['Implicit', 'Explicit'])

new_user = st.checkbox('Người dùng mới? (không cần ID)', value=True)

# Số lượng gợi ý
top_k = st.number_input('Số lượng gợi ý', min_value=1, max_value=20, value=10, step=1)

# Nhập ID người dùng cũ
user_id = st.number_input('ID người dùng (TỐI ĐA: 6040)', min_value=1, max_value=6040, value=3000, step=1, disabled=new_user)

# Thông tin người dùng mới
user_gender = st.selectbox('Giới tính', ['M', 'F'], disabled=not new_user)
user_age = st.number_input('Tuổi', min_value=1, max_value=99, value=25, step=1, disabled=not new_user)
user_occupation = st.selectbox('Nghề nghiệp', occupation, disabled=not new_user, help='Chọn nghề nghiệp của bạn', index=17)
user_genres = st.multiselect('Thể loại yêu thích', genre, disabled=not new_user, help='Chọn ít nhất 3 thể loại', default=['Comedy', 'Children', 'Animation'], max_selections=5)

# Nút lấy gợi ý
recommend = st.button('Lấy gợi ý')

# create the user dict
user = {
    'top_k': top_k,
    'id': user_id if not new_user else 9000,
    'age': user_age if new_user else None,
    'gender': user_gender if new_user else None,
    'occupation': user_occupation if new_user else None,
    'genres': user_genres if new_user else None
}

# Get recommendations
if recommend and 5 >= len(user_genres) >= 3:
    pred_movies, top_n_genres = Utils.pipeline(
        request=user,
        model=model_exp if model_type == 'Explicit' else model_imp,
        users=users_exp if model_type == 'Explicit' else users_imp,
        movies=movies,
        movies_og=movies_og,
        ratings=ratings,
        weights=[model_exp.user_embedding_mlp.weight.data.cpu().numpy(), model_exp.user_embedding_mf.weight.data.cpu().numpy()] if model_type == 'Explicit' else [model_imp.user_embedding_mlp.weight.data.cpu().numpy(), model_imp.user_embedding_mf.weight.data.cpu().numpy()],
        mode=model_type.lower()
    )

    # Hiển thị kết quả
    st.write(
        f'Top {top_k} gợi ý cho người dùng có ID {user_id}:'
        if not new_user
        else f'Top {top_k} gợi ý cho người dùng mới:'
    )
    # st.write(pred_movies, unsafe_allow_html=True)
    if not new_user:
        st.write(f'Các thể loại mà người dùng ID {user_id} thích nhất: {", ".join(top_n_genres)}')

    pred = 'điểm đánh giá' if model_type == 'Explicit' else 'điểm độ phù hợp'

    html = """<div class="card-container">"""
    for i, movie in enumerate(pred_movies):
        # create the movie card
        html += f"""<div class="card">
                <h5 class="card-title">{i + 1}</h5>
                <p class="card-text">Tiêu đề: <b style="font-size: 1.2em;">{movie['title']}</b></p>
                <p class="card-text">Thể loại: {movie['genre']}</p>
                <p class="card-text">Giá trị dự đoán ({pred}): {movie['predicted_score'] if model_type == 'Implicit' else movie['predicted_rating']}</p>
            </div>"""

    st.markdown(
        html + '</div>',
        unsafe_allow_html=True
    )

elif recommend and len(user_genres) < 3:
    st.write('Vui lòng chọn từ 3 đến 5 thể loại')

# Fixed footer
st.markdown(
    css,
    unsafe_allow_html=True
)