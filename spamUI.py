import streamlit as st
import pickle
from win32com.client import Dispatch
import pythoncom
import matplotlib.pyplot as plt
pythoncom.CoInitialize()

def speak(text):
	speak=Dispatch(("SAPI.SpVoice"))
	speak.Speak(text)

try:
	model = pickle.load(open('spam.pkl','rb'))
	cv=pickle.load(open('vectorizer.pkl','rb'))
	image = plt.imread('ROC.png')
except (IOError, OSError, pickle.PickleError, pickle.UnpicklingError):
	print("Not a valid file please try again")
	st.error("Not a valid file please try again")

def main():
	st.title("Email Spam Classification Application")
	st.write("Build with Streamlit & Python")
	activites=["Classification","ROC Graph"]
	choices=st.sidebar.selectbox("Select Activities",activites)
	if choices=="Classification":
		st.subheader("Classification")
		msg=st.text_input("Enter a text")
		if st.button("Process"):
			print(msg)
			data=[msg]
			print(data)
			vec=cv.transform(data).toarray()
			result=model.predict(vec)
			if result[0]==0:
				st.success("This is Not A Spam Email")
				speak("This is Not A Spam Email")
			else:
				st.error("This is A Spam Email")
				speak("This is A Spam Email")
	else:
		st.image(image, width=None)
main()

