import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

pipe_lr = joblib.load(open("model/text_emotion.pkl", "rb"))

emotions_emoji_dict = {"anger": "\U0001F620", "disgust": "\U0001F92E", "fear": "\U0001F628\U0001F631", "happy": "\U0001F917", "joy": "\U0001F602", "neutral": "\U0001F610", "sad": "\U0001F614",
                       "sadness": "\U0001F614", "shame": "\U0001F633", "surprise": "\U0001F62E"}


def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


def main():
    # Header Bar
    st.markdown("""
    <div style="background-color:#4CAF50;padding:10px;border-radius:10px;text-align:center;">
        <h1 style="color:white;">Text Emotion Detection</h1>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Detect Emotions In Text")

    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict[prediction]
            st.write("{}:{}".format(prediction, emoji_icon))
            st.write("Confidence:{}".format(np.max(probability)))

        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
            st.altair_chart(fig, use_container_width=True)

    # Footer Bar with Profile Logos
    st.markdown("""
    <div style="background-color:#4CAF50;padding:10px;border-radius:10px;text-align:center;">
        <p style="color:white;font-size:16px;">Made By Harendra Singh</p>
        <p>
            <a href="https://github.com/Harendrasingh1" style="color:white;">
                <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20"> GitHub Profile
            </a> |
            <a href="https://www.linkedin.com/in/harendra8587/" style="color:white;">
                <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="20"> LinkedIn Profile
            </a> |
            <a href="https://leetcode.com/u/Thakurji01/" style="color:white;">
                <img src="https://upload.wikimedia.org/wikipedia/commons/1/19/LeetCode_logo_black.png" width="20"> LeetCode Profile
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
