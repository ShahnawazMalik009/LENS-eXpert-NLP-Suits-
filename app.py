import streamlit as st
import joblib
import pandas as pd
spam_model=joblib.load("spam_classifier.pkl")
news_model=joblib.load("news_cate.pkl")
language_model=joblib.load("lang_dec.pkl")
review_model=joblib.load("review.pkl")

st.title("🤖 LENSE eXpert(NLP Suits)")
st.write('***A Multi-Domain Text Intelligence System***')

tab1,tab2,tab3,tab4=st.tabs(["📨 Spam Classifier","🌐 Language Detection","😊 / 😞 Food Review Sentiment","📰 News Classification"])


with tab1:
    msg=st.text_input("Enter your text here")
    if st.button("Predict"):
        pred=spam_model.predict([msg])
        if pred[0]==0:
            st.image('stop-spam.png')
        else:
            st.image('notspam.png')
    
    uploaded_file=st.file_uploader("Upload a file",type=(['csv','txt']))            
    
    if uploaded_file:
        df=pd.read_csv(uploaded_file,header=None,names=['Msg'])
        pred=spam_model.predict(df.Msg)
        df.index=range(1,df.shape[0]+1)
        df["prediction"]=pred
        df["prediction"]=df["prediction"].map({0:'Spam',1:'Not spam'})
        st.dataframe(df)
   
          
with tab2:
    msg2=st.text_input('Enter your text here',key="k")
    btn=st.button('Predict',key="b2")
    if btn:
        pred2=language_model.predict([msg2])
        if pred2[0]=='English':
            st.success('English')
        if pred2[0]=='Hindi':
            st.success('Hindi')
        if pred2[0]=='French':
            st.success('French')
        if pred2[0]=='Spanish':
            st.success('Spanish')
        if pred2[0]=='Italian':
            st.success('Italian')
                
    
    uploaded_file=st.file_uploader("Upload a file",type=(['csv','txt']),key="j")            
    
    if uploaded_file:
        df=pd.read_csv(uploaded_file,header=None,names=['Pridiction'])
        pred=language_model.predict(df.Pridiction)
        st.dataframe(df)
   
    
    
    
with tab3:
    msg=st.text_input("Enter your text here",key='l')
    if st.button("Predict",key='p'):
        pred=review_model.predict([msg])
        if pred[0]==0:
            st.image('notlike.png')
        else:
            st.image('okay.png')
    
    uploaded_file=st.file_uploader("Upload a file",type=(['csv','txt']),key='m')            
    
    if uploaded_file:
        df=pd.read_csv(uploaded_file,header=None,names=['Msg'])
        pred=review_model.predict(df.Msg)
        df.index=range(1,df.shape[0]+1)
        df["prediction"]=pred
        df["prediction"]=df["prediction"].map({0:'👎Not Like',1:'👍Like'})
        st.dataframe(df)

with tab4:
    st.image("coming.webp")    
            
st.sidebar.image('img.jpg')           
with st.sidebar.expander("ℹ️ About us"):
    st.write("I am a Student trying to understand the concept of NLP")
with st.sidebar.expander("🧠 Technology Used"):
    st.write("Python 🐍,/nScikit-learn,/nNLTK / SpaCy,/nPandas, Matplotlib, Seaborn (for data handling and visualization)")   
with st.sidebar.expander("🌟 Key Highlights"):
    st.write("End-to-end text preprocessing pipelines,/nCustom stopword filtering and lemmatization,/nInteractive visualizations for each task,/nModular design for easy expansion to other NLP tasks")
with st.sidebar.expander("🏁 Conclusion"):
    st.write("This project demonstrates the versatility of NLP in handling various text-based problems with high efficiency. Each module showcases a different facet of language understanding, paving the way for smart text analytics solutions.")    
with st.sidebar.expander("📞 Contact us"):
    st.write("+918999045062")
    st.write("shahnawazmlik009@gmail.com")
    



