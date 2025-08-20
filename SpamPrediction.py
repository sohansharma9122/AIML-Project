import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

data=pd.read_csv(r"C:\Users\HP\Desktop\Email_Spam\spam.csv")

data.drop_duplicates(inplace=True)

data['Category']=data['Category'].replace(['ham','Spam'],['Not Spam','Spam'])

mess=data['Message']
cat=data['Category']

(mess_train, mess_test, cat_train, cat_test)=train_test_split(mess,cat,test_size=0.2)

cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)

#~~~~~~~~~~~~~~~~~
# create model
#~~~~~~~~~~~~~~~~

model = MultinomialNB()
model.fit(features, cat_train)


#~~~~~~~~~~~~~~~~~~~~~~~~
## Testing our data
#~~~~~~~~~~~~~~~~~~~~~~

features_test = cv.transform(mess_test)
# print(model.score(features_test, cat_test))

# ~~~~~~~~~~~~~~~~~~~~
## Predict Data
#~~~~~~~~~~~~~~~~~~~~~
def predict(message):
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result
st.header('Spam Ditection')
output = predict('Congratulation, you won a lottery')
input_mess = st.text_input('Enter Message Here')
if st.button('Validate'):
    output = predict(input_mess)
    st.markdown(output)


# if we want to run this write this Command on your terminal
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PS C:\Users\HP\Desktop\Email_Spam> python -m streamlit run spam.py
# >>

#   You can now view your Streamlit app in your browser.

#   Local URL: http://localhost:8501
#   Network URL: http://10.41.107.194:8501

