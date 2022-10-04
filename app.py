import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib as plt
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import seaborn as sns
import os
from xgboost import XGBClassifier

#%matplotlib inline
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

import joblib
import pickle

import sqlite3
import datetime



conn = sqlite3.connect('data.db')
c = conn.cursor()

def create_table():
	c.execute('CREATE TABLE IF NOT EXISTS notestable(author TEXT,title TEXT,message TEXT)')


def add_data(author,title,message):
	c.execute('INSERT INTO notestable(author,title,message) VALUES (?,?,?)',(author,title,message))
	conn.commit()


def view_all_notes():
	c.execute('SELECT * FROM notestable')
	data = c.fetchall()
	# for row in data:
	# 	print(row)
	return data

class Monitor(object):
	"""docstring for Monitor"""

	conn = sqlite3.connect('data.db')
	c = conn.cursor()

class Monitor(object):
	"""docstring for ClassName"""
	conn = sqlite3.connect('data.db')
	c = conn.cursor()

	def __init__(self, Customer_age=None ,Gender=None ,Dependent_count=None , Education_level=None ,Marital_Status=None , Income_Category=None ,Card_Category=None ,Total_Relationship_Count=None ,Months_Inactive_12_mon=None ,Contacts_Count_12_mon=None ,Credit_Limit=None ,Total_Revolving_Bal=None ,Total_Amt_Chng_Q4_Q1=None ,Total_Trans_Amt=None ,Total_Ct_Chng_Q4_Q1=None ,Avg_Utilization_Ratio=None ,predicted_class=None,model_class=None, time_of_prediction=None ):
		super(Monitor, self).__init__()
		self.Customer_age = Customer_age
		self.Gender = Gender
		self.Dependent_count = Dependent_count
		self.Education_level = Education_level
		self.Marital_Status = Marital_Status
		self.Income_Category = Income_Category
		self.Card_Category = Card_Category
		self.Total_Relationship_Count = Total_Relationship_Count
		self.Months_Inactive_12_mon = Months_Inactive_12_mon
		self.Contacts_Count_12_mon = Contacts_Count_12_mon
		self.Credit_Limit = Credit_Limit
		self.Total_Revolving_Bal = Total_Revolving_Bal
		self.Total_Amt_Chng_Q4_Q1 = Total_Amt_Chng_Q4_Q1
		self.Total_Trans_Amt = Total_Trans_Amt
		self.Total_Ct_Chng_Q4_Q1 = Total_Ct_Chng_Q4_Q1
		self.Avg_Utilization_Ratio = Avg_Utilization_Ratio
		#self.native_country = native_country
		self.predicted_class = predicted_class
		self.model_class = model_class
		self.time_of_prediction = time_of_prediction

	def __repr__(self):
		# return "Monitor(age ={self.age},workclass ={self.workclass},fnlwgt ={self.fnlwgt},education ={self.education},education_num ={self.education_num},marital_status ={self.marital_status},occupation ={self.occupation},relationship ={self.relationship},race ={self.race},sex ={self.sex},capital_gain ={self.capital_gain},capital_loss ={self.capital_loss},hours_per_week ={self.hours_per_week},predicted_class ={self.predicted_class},model_class ={self.model_class})".format(self.age,self.workclass,self.fnlwgt,self.education,self.education_num,self.marital_status,self.occupation,self.relationship,self.race,self.sex,self.capital_gain,self.capital_loss,self.hours_per_week,self.native_country,self.predicted_class,self.model_class)
		"Monitor(Customer_age = {self.Customer_age},Gender = {self.Gender},Dependent_count = {self.Dependent_count},Education_level = {self.Education_level},Marital_Status = {self.Marital_Status},Income_Category = {self.Income_Category},Card_Category = {self.Card_Category},Total_Relationship_Count = {self.Total_Relationship_Count},Months_Inactive_12_mon = {self.Months_Inactive_12_mon},Contacts_Count_12_mon = {self.Contacts_Count_12_mon},Credit_Limit = {self.Credit_Limit},Total_Revolving_Bal = {self.Total_Revolving_Bal},Total_Amt_Chng_Q4_Q1 = {self.Total_Amt_Chng_Q4_Q1},Total_Trans_Amt = {self.Total_Trans_Amt}, Total_Ct_Chng_Q4_Q1 = {self.Total_Ct_Chng_Q4_Q1}, Avg_Utilization_Ratio = {self.Avg_Utilization_Ratio}, predicted_class = {self.predicted_class}, model_class = {self.model_class})".format(self=self)
	
	def create_table(self):
		self.c.execute('CREATE TABLE IF NOT EXISTS predictiontable(Customer_age NUMERIC, Gender NUMERIC, Dependent_count NUMERIC, Education_level NUMERIC, Marital_Status NUMERIC, Income_Category NUMERIC, Card_Category NUMERIC, Total_Relationship_Count NUMERIC, Months_Inactive_12_mon NUMERIC, Contacts_Count_12_mon NUMERIC, Credit_Limit NUMERIC, Total_Revolving_Bal NUMERIC, Total_Amt_Chng_Q4_Q1 NUMERIC, Total_Trans_Amt NUMERIC, Total_Ct_Chng_Q4_Q1 NUMERIC, Avg_Utilization_Ratio NUMERIC, predicted_class NUMERIC, model_class TEXT, time_of_prediction TEXT)')

	def add_data(self):
		self.c.execute('INSERT INTO predictiontable(Customer_age, Gender, Dependent_count, Education_level, Marital_Status, Income_Category, Card_Category, Total_Relationship_Count, Months_Inactive_12_mon, Contacts_Count_12_mon, Credit_Limit, Total_Revolving_Bal, Total_Amt_Chng_Q4_Q1, Total_Trans_Amt, Total_Ct_Chng_Q4_Q1, Avg_Utilization_Ratio, predicted_class, model_class, time_of_prediction) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(self.Customer_age,self.Gender,self.Dependent_count,self.Education_level,self.Marital_Status,self.Income_Category,self.Card_Category,self.Total_Relationship_Count,self.Months_Inactive_12_mon,self.Contacts_Count_12_mon,self.Credit_Limit,self.Total_Revolving_Bal,self.Total_Amt_Chng_Q4_Q1,self.Total_Trans_Amt,self.Total_Ct_Chng_Q4_Q1,self.Avg_Utilization_Ratio,self.predicted_class,self.model_class, self.time_of_prediction))
		self.conn.commit()

	def view_all_data(self):
		self.c.execute('SELECT * FROM predictiontable')
		data = self.c.fetchall()
		# for row in data:
		# 	print(row)
		return data


# To get value -- value mapping
def get_value(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return value

# get keys

def get_key(val, my_dict):
	for key,value in my_dict.items():
		if val == value:
			return key



# Load Models
def load_model_n_predict(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model

def main():

	""" salary predictor"""






	st.title("Customer EDA and Churn Prediction APP")
	activity = ["EDA", "Prediction", "Metrics", "About"]
	choice = st.sidebar.selectbox("Choose a Task", activity)

	# Load file
	df = pd.read_csv("BankChurners.csv", error_bad_lines=False)
	#EDA
	if choice == 'EDA':
		st.subheader("EDA Section")
		st.text("Eploratory Data Analysis")
		
		
		
		if st.checkbox("Customer's Status"):
			st.set_option('deprecation.showPyplotGlobalUse', True)
			fig = plt.figure(figsize=(10, 4))
			sns.countplot(x = "Attrition_Flag", data = df)
			st.pyplot(fig)	


		if st.button("Gender"):
			st.set_option('deprecation.showPyplotGlobalUse', True)
			fig = plt.figure(figsize=(10, 4))
			sns.countplot(x = "Gender", data = df)
			st.pyplot(fig)

		if st.checkbox("Dependent_count"):
			st.set_option('deprecation.showPyplotGlobalUse', True)
			fig = plt.figure(figsize=(10, 4))
			sns.countplot(x = "Dependent_count", data = df)
			st.pyplot(fig)


		if st.checkbox("Education_Level"):
			st.set_option('deprecation.showPyplotGlobalUse', True)
			fig = plt.figure(figsize=(10, 4))
			sns.countplot(x = "Education_Level", data = df)
			st.pyplot(fig)

		if st.checkbox("Marital_Status"):
			st.set_option('deprecation.showPyplotGlobalUse', True)
			fig = plt.figure(figsize=(10, 4))
			sns.countplot(x = "Marital_Status", data = df)
			st.pyplot(fig)

		if st.button("Income_Category"):
			st.set_option('deprecation.showPyplotGlobalUse', True)
			fig = plt.figure(figsize=(10, 4))
			sns.countplot(x = "Income_Category", data = df)
			st.pyplot(fig)


		if st.checkbox("Card_Category"):
			st.set_option('deprecation.showPyplotGlobalUse', True)
			fig = plt.figure(figsize=(10, 4))
			sns.countplot(x = "Card_Category", data = df)
			st.pyplot(fig)



		if st.checkbox("Credit Limit vs Total Revolving Bal vs Total Trans Amt"):
			Credit_Limit = np.random.randn(200) - 2
			Total_Revolving_Bal = np.random.randn(200)
			Total_Trans_Amt = np.random.randn(200) + 2

			# Group data together
			df = [Credit_Limit, Total_Revolving_Bal, Total_Trans_Amt]

			group_labels = ['Credit_Limit', 'Total_Revolving_Bal', 'Total_Trans_Amt']

			# Create distplot with custom bin_size
			fig = ff.create_distplot(
			        df, group_labels, bin_size=[.1, .25, .5])

			# Plot!
			st.plotly_chart(fig, use_container_width=True)
		

		if st.button("Months_on_book"):
			chart_data = pd.DataFrame(
			np.random.randn(20, 1),
			columns=['Months_on_book'])

			st.line_chart(chart_data)

		
		


		if st.button("Gender vs Churn"):
			chart_data = pd.DataFrame(
    		np.random.randn(50, 2),
    		columns=["Attrition_Flag", "Gender"])

			st.bar_chart(chart_data)	




			#preview

		if st.checkbox("Preview Dataset"):
			number = int(st.number_input ("Number to Show"))
			st.dataframe(df.head(number))
        
        #Showing Column/Rows

		if st.button("Column Names"):
			st.write(df.columns)
        
		if st.checkbox("Show Description"):
			st.write(df.describe())
        
        #Shape
		if st.checkbox("show shape of Dataset"):
			st.write(df.shape)	
			data_dim = st.radio("show Dimensions by", ("Rows", "Columns"))

			if data_dim == 'Rows':
				st.text("Number of Rows")
				st.write(df.shape[0])
			elif data_dim == 'Columns':
				st.text("Number of Columns")
				st.write(df.shape[1])
			else:
				st.write(df.shape)
		#selections

		if st.checkbox("Select Columns to Show"):
			all_columns = df.columns.tolist()
			selected_columns = st.multiselect("Select Colums", all_columns)
			new_df = df[selected_columns]
			st.dataframe(new_df)

		if st.checkbox("Select Rows To Show"):
			selected_index = st.multiselect("Select Rows", df.head(10).index)
			selected_rows = df.loc[selected_index]
			st.dataframe(selected_rows)

		#value Count
		if st.button("value_count"):
			st.text("Value Count By Class")
			st.write(df.iloc[:,0].value_counts())


		#Plot
		#st.set_option('deprecation.showPyplotGlobalUse', False)
		#if st.checkbox("Show Correlation Plot"):
			#plt.matshow(df.corr())
			#st.pyplot()

		if st.checkbox("Show Correlation Plots"):
			st.set_option('deprecation.showPyplotGlobalUse', False)
			st.write(sns.heatmap(df.corr(),annot=True, linewidths=0.5,center=0,cbar=False,cmap="YlGnBu"))
			st.pyplot()

		

	


	#Prediction

	elif choice == 'Prediction':
		st.subheader("Prediction Section")
		d_Education_Level = {"Graduate": 1, "High School": 2, "Uneducated": 3, "College": 4, "Post-Graduate": 5, "Doctorate": 6, "Unknown": np.nan}
		d_Marital_Status = {"Married": 1, "Single": 2, "Divorced": 3, "Unknown": np.nan}
		d_Income_Category = {"Less than $40K": 1, "$40K - $60K": 2, "$80K - $120K": 3, "$60K - $80K": 4, "$120K +": 5, "Unknown": np.nan}
		d_Card_Category =  {"Blue": 1, "Silver": 2, "Gold": 3, "Platinum": 4}
		d_Attrition_Flag = {"Existing Customer": 0, "Attrited Customer": 1}
		d_Gender = {"F": 1, "M": 2}

	#Ml mapping

		Customer_age = st.number_input("Select Age", 26,73)
		Gender = st.selectbox("Select Gender", tuple(d_Gender.keys()))
		Dependent_count = st.slider("Select Dependent Count", 0,5)
		Education_level = st.selectbox("Select Eductaion", tuple(d_Education_Level.keys()))
		Marital_Status = st.selectbox("Select Marital Status", tuple(d_Marital_Status.keys()))
		Income_Category = st.selectbox("Select income Category", tuple(d_Income_Category.keys()))
		Card_Category = st.selectbox("Select Card Category", tuple(d_Card_Category.keys()))
		Total_Relationship_Count = st.slider("Select Relationship Count", 1,6)
		Months_Inactive_12_mon = st.slider("Select Month Inactive", 0,6)    
		Contacts_Count_12_mon  = st.number_input("Select Contact_Count", 0,6) 
		Credit_Limit = st.number_input("Select Credit Limit", 1000,35000) 
		Total_Revolving_Bal = st.number_input("Select Total Revolving Balance", 0,2600) 
		Total_Amt_Chng_Q4_Q1 = st.number_input("Select Total amount Change", 0,4) 
		Total_Trans_Amt = st.number_input("Select Total Transcation Amount", 500,20000) 
		Total_Ct_Chng_Q4_Q1 = st.number_input("Select Total Count Change", 0,4)       
		Avg_Utilization_Ratio = st.slider("Select Average Utilization Ratio", 0.0,1.0) 



		k_Education_level = get_value(Education_level, d_Education_Level)
		k_Marital_Status = get_value(Marital_Status, d_Marital_Status)
		k_Income_Category = get_value(Income_Category, d_Income_Category)
		k_Card_Category = get_value(Card_Category, d_Card_Category)
		k_Gender = get_value(Gender, d_Gender)

		#Result of User Input

		Selected_Options = [Customer_age, Gender, Dependent_count, Education_level, Marital_Status, Income_Category, Card_Category, Total_Relationship_Count, Months_Inactive_12_mon, Contacts_Count_12_mon, Credit_Limit, Total_Revolving_Bal, Total_Amt_Chng_Q4_Q1, Total_Trans_Amt, Total_Ct_Chng_Q4_Q1, Avg_Utilization_Ratio ]
		Vectorized_result = [Customer_age,	k_Gender, Dependent_count, k_Education_level, k_Marital_Status, k_Income_Category, k_Card_Category, Total_Relationship_Count,	Months_Inactive_12_mon, Contacts_Count_12_mon, Credit_Limit, Total_Revolving_Bal, Total_Amt_Chng_Q4_Q1,Total_Trans_Amt, Total_Ct_Chng_Q4_Q1 ,	Avg_Utilization_Ratio ]
		sample_data = np.array(Vectorized_result).reshape(1,-1)
		st.info(Selected_Options)
		prettified_result = {"Customer_age":Customer_age, 
		"Gender":Gender, "Dependent_count":Dependent_count, 
		"Education_level":Education_level, "Marital_Status": Marital_Status, 
		"Income_Category": Income_Category, "Card_Category": Card_Category, 
		"Total_relationship_Count": Total_Relationship_Count, 
		"Months_Inactive_12_mon":Months_Inactive_12_mon, 
		"Contacts_Count_12_mon":Contacts_Count_12_mon, "Credit_Limit":Credit_Limit, 
		"Total_Revolving_Bal": Total_Revolving_Bal, "Total_Amt_Chng_Q4_Q1":Total_Amt_Chng_Q4_Q1, 
		"Total_Trans_Amt": Total_Trans_Amt, "Total_Ct_Chng_Q4_Q1":Total_Ct_Chng_Q4_Q1, 
		"Avg_Utilization_Ratio": Avg_Utilization_Ratio}

		st.json(prettified_result)
		#st.write(Vectorized_result)
		
		
		# MAKING PREDICTION
		st.subheader("Prediction")
		if st.checkbox("Make Prediction"):
			all_ml_list = {'LR':"LogisticRegression",
				'DT':"DecisionTree",
				'XGB':"XGBoosting"}


			#Model Selection
			model_choice = st.selectbox("Model Choice", all_ml_list)
			prediction_label = {"Churn": 1, "Stay": 0}
			'''
			if st.button("prediction_label"):
				filename = 'finalized_model.sav'
				loaded_model = pickle.load(open(filename, 'rb'))
				prediction = loaded_model.predict(sample_data)
			'''
				
			if st.button("prediction_label"):
				if model_choice == 'DT':
					model_predictor = load_model_n_predict("finalized_model_version2 - DT.sav")
					prediction = model_predictor.predict(sample_data)
				elif model_choice == 'LR':
					model_predictor = load_model_n_predict("finalized_model_version2 - LR.sav")
					prediction = model_predictor.predict(sample_data)
				elif model_choice == 'XGB':
					model_predictor = load_model_n_predict("finalized_model_version2 - XGB.sav")
					prediction = model_predictor.predict(sample_data)
					# st.text(prediction)




				#st.write(sample_data)
				#st.write(Vectorized_result)
				#prediction = model_predictor.predict(sample_data)
				#st.write(prediction)

				final_result = get_key(prediction, prediction_label)
				model_class = model_choice
				time_of_prediction = datetime.datetime.now()
				monitor = Monitor(Customer_age, Gender, Dependent_count, Education_level, Marital_Status, Income_Category, Card_Category, Total_Relationship_Count, Months_Inactive_12_mon, Contacts_Count_12_mon, Credit_Limit, Total_Revolving_Bal, Total_Amt_Chng_Q4_Q1, Total_Trans_Amt, Total_Ct_Chng_Q4_Q1, Avg_Utilization_Ratio,final_result,model_class, time_of_prediction)
				monitor.create_table()
				monitor.add_data()
				st.success("Predicting customer will :: {}".format(final_result))



		# CHOICE FOR COUNTRIES
	
	# if choice == 'countries':
	# 	st.text("Demographics")
	# 	d_native_country = {"Canada": 0, "Philippines": 1, "Thailand": 2, "Scotland": 3, "Germany": 4, "Portugal": 5, "India": 6, "China": 7, "Japan": 8, "Peru": 9, "France": 10, "Greece": 11, "Taiwan": 12, "Laos": 13, "Hong": 14, "El-Salvador": 15, "Outlying-US(Guam-USVI-etc)": 16, "Yugoslavia": 17, "Cambodia": 18, "Italy": 19, "Honduras": 20, "Puerto-Rico": 21, "Dominican-Republic": 22, "Vietnam": 23, "Poland": 24, "Hungary": 25, "Holand-Netherlands": 26, "Ecuador": 27, "South": 28, "Guatemala": 29, "United-States": 30, "Nicaragua": 31, "Trinadad&Tobago": 32, "Cuba": 33, "Jamaica": 34, "Iran": 35, "?": 36, "Haiti": 37, "Columbia": 38, "Mexico": 39, "England": 40, "Ireland": 41}
	# 	selected_countries = st.selectbox("Select Country",tuple(d_native_country.keys()))
	# 	st.text(selected_countries)

	# 	result_df = df2[df2['native-country'].str.contains(selected_countries)]
	# 	st.dataframe(result_df)
	

	# if st.checkbox("Select Columns To Show"):
	# 			result_df_columns = result_df.columns.tolist()
	# 			selected_columns = st.multiselect('Select',result_df_columns)
	# 			new_df = df2[selected_columns]
	# 			st.dataframe(new_df)

	# 			if st.checkbox("Plot"):
	# 				st.area_chart(df[selected_columns])
	# 				st.pyplot()
	
				
	#Metrics

	if choice == 'Metrics':
		st.subheader("Metrics")
		# Create your connection.
		cnx = sqlite3.connect('data.db')

		mdf = pd.read_sql_query("SELECT * FROM predictiontable", cnx)
		st.dataframe(mdf)

	# ABOUT CHOICE
	if choice == 'About':
		st.subheader("About")
		st.markdown("""
			### Predicting if a customer will stop using thier Credit card
			#### Built with Streamlit
			
			Data dictionary
			
			+ Customer_Age: Age in Years
			+ Gender: Gender of account holder
			+ Dependent_count: Number of dependents
			+ Education_Level: Educational Qualification of the account holder
			+ Marital_Status: Marital Status of account holder
			+ Income_Category: Annual Income Category of the account holder
			+ Card_Category: Type of Card
			+ Total_Relationship_Count: Total no. of products held by the customer
			+ Months_Inactive_12_mon: No. of months inactive in the last 12 months
			+ Contacts_Count_12_mon: No. of Contacts in the last 12 months
			+ Credit_Limit: Credit Limit on the Credit Card
			+ Total_Revolving_Bal: Total Revolving Balance on the Credit Card
			+ Total_Amt_Chng_Q4_Q1: Change in Transaction Amount (Q4 over Q1)
			+ Total_Trans_Amt: Total Transaction Amount (Last 12 months)
			+ Total_Ct_Chng_Q4_Q1: Change in Transaction Count (Q4 over Q1)
			+ Avg_Utilization_Ratio: Average Card Utilization Ratio
			### By
			+ Temi Olorunfemi PhD.....Jesus saves
			

			""")

		
	
if __name__== '__main__':
	main()
