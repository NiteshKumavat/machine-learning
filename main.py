import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import math
import pickle as pkl

sidebar = st.sidebar

sidebar.title("Navigation Bar")

active = sidebar.radio("Choose the Page ", options=["Home", "Currency Converter", "Supervised Learning", "Unsupervised Learning"])

st.set_page_config(active)

def learning(type) :
    types = {
        "Supervised Learning" : {
            "definition" : "A type of Machine Learning where the model is trained on labeled data, meaning each input has a corresponding correct output.",
            "Types" : {
                "Linear Regression" : {
                    "definition" : "A supervised learning algorithm used for predicting a continuous output based on the relationship between independent and dependent variables by fitting a straight line.",
                    "img" : "linear-regression.png",
                    "code" : '''
from sklearn.linear_model import LinearRegression
model = LinearRegression()
# x is your data from dataset
# y is your output from dataset
model.fit(x,y)
''',
                }, 
                "Logistic Regression" : {
                    "definition" : "A classification algorithm that predicts the probability of a binary outcome (like Yes/No or 0/1) using a sigmoid function.",
                    "img" : "logistic-regression.png",
                    "code" : '''
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
# x is your data from dataset
# y is your output from dataset
model.fit(x,y)
''',
                }, 
                "K-nn(K nearest neighbour)" : {
                    "definition" : "A non-parametric, instance-based learning algorithm that classifies a data point based on how its ‘K’ nearest neighbors are classified.",
                    "img" : "K-nn.webp",
                    "code" : '''
from sklearn.neighbours import KNeighborsClassifier
model = KNeighborsClassifier()
# x is your data from dataset
# y is your output from dataset
model.fit(x,y)
''',
                }, 
                "Decision Tree" : {
                    "definition" : "A model that makes decisions by splitting data into branches based on feature values, forming a tree structure where each leaf represents a prediction.",
                    "img" : "Decision-Tree.webp",
                    "code" : '''
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
# x is your data from dataset
# y is your output from dataset
model.fit(x, y)
''',
                }, 
                "Random Forest" : {
                    "definition" : "An ensemble learning method that builds multiple decision trees and combines their outputs to improve accuracy and reduce overfitting.",
                    "img" : "Random-forest-algorithm.webp",
                    "code" : '''
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
# x is your data from dataset
# y is your output from dataset
model.fit(x, y)
'''
                }
            }
        },
        "Unsupervised Learning" : {
            "definition" : "A type of Machine Learning where the model is trained on unlabeled data, and it tries to find hidden patterns or groupings in the data.",
            "Types" : {
                "K-means" : {
                    "definition" : "K-Means is an unsupervised learning algorithm used to group similar data points into K distinct clusters.",
                    "img" : "K-means.png",
                    "code" : '''
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)  
# X is your dataset
'''
                }
            }
        },
        "Reinforcement Learning" : {
            "definition" : "A type of Machine Learning where an agent learns by interacting with an environment and receives rewards or penalties based on its actions."
        }

    }

    st.info(types[type]["definition"])
    if type != "Reinforcement Learning" :
        options = list(types[type]["Types"].keys())
        options.insert(0, "None")
        if type == "Supervised Learning" :
            st.subheader("Supervised Learning")
            st.subheader("Types of Supervised Learning")
            st.markdown("**1. Regression**: Predicts continuous numeric values, like prices, temperatures, or sales amounts.")
            st.markdown("**2. Classification**: Predicts discrete categories or labels, like spam/not spam, pass/fail, or disease/no disease.")
        algo = st.selectbox("Choose the Algorithm", options=options)
        if algo != "None" :
            info = types[type]["Types"][algo]
            st.info(info["definition"])
            st.image(f"images/{info["img"]}")
            st.code(info["code"], language="python")
        

def home():
    st.title("Machine Learning Notes")
    st.info("Machine Learning (ML) is a branch of artificial intelligence (AI) that focuses on building systems that can learn from data, identify patterns, and make decisions with minimal human intervention.")
    st.header("Types of ML")
    type = st.selectbox("Choose the ML Type", options=["None", "Supervised Learning", "Unsupervised Learning", "Reinforcement Learning"])

    if type != "None":
        learning(type)


def currencyConverter():
    st.title("Currency Converter")

    currencies = {
        "Indian Rupee (₹)": "INR",
        "US Dollar ($)": "USD",
        "Chinese Yuan (¥ / CNY)": "CNY",
        "Euro (€)": "EUR",
        "South Korean Won (₩)": "KRW"
    }

    conversion_rates = {
        "INR": {"INR": 1, "USD": 1/85.75, "CNY": 1/11.98, "EUR": 1/100.28893, "KRW": 16.035},
        "USD": {"INR": 85.75, "USD": 1, "CNY": 7.168, "EUR": 1/1.169, "KRW": 1376},
        "CNY": {"INR": 11.98, "USD": 1/7.168, "CNY": 1, "EUR": 0.847, "KRW": 192.05},
        "EUR": {"INR": 100.28893, "USD": 1.169, "CNY": 1.18, "EUR": 1, "KRW": 1572},
        "KRW": {"INR": 0.0624, "USD": 1/1376, "CNY": 1/192.05, "EUR": 1/1572, "KRW": 1}
    }

    option1 = st.selectbox("From Currency", options=["None"] + list(currencies.keys()), key="from_currency")
    st.number_input("Enter the amount", key="value1", min_value=0.0, format="%.3f")

    option2 = st.selectbox("To Currency", options=["None"] + list(currencies.keys()), key="to_currency")

    if "value2" not in st.session_state:
        st.session_state.value2 = 0.0

    if option1 != "None" and option2 != "None":
        from_code = currencies[option1]
        to_code = currencies[option2]
        rate = conversion_rates[from_code][to_code]

        st.session_state.value2 = round(st.session_state.value1 * rate, 3)

    st.number_input("Converted Value", value=st.session_state.value2, key="value2")


def analyse(type, df) :
    if type == "Describe" :
        st.write(df.describe())
    elif type == "Columns" :
        st.write(df.columns)
    elif type == "information" :
        buffer = io.StringIO()
        df.info(buf = buffer)
        info_str = buffer.getvalue()
        st.code(info_str, language="text")

def supervised():
    st.title("Supervised Learning")
    file = st.file_uploader("Upload the data file (.csv)", type=["csv"])

    if file:
        st.header("Analysis")
        df = pd.read_csv(file)
        analysis = st.selectbox("Select the choice", options=["None", "Describe", "Columns", "information"])
        if analysis != "None":
            analyse(analysis, df)

        columns = list(df.columns)
        output_col = st.radio("Choose the column to predict", options=columns)

        label_mappings = {}

        for column in columns:
            if df[column].dtype == object:
                label = LabelEncoder()
                df[column] = label.fit_transform(df[column])
                mapping_dict = dict(zip(label.classes_, label.transform(label.classes_)))
                label_mappings[column] = mapping_dict

        with open("Supervised/label_mappings.pkl", "wb") as f:
            pkl.dump(label_mappings, f)

        df = df.dropna()
        x = df.drop([output_col], axis=1)
        y = df[output_col]

        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.1, random_state=42)

        models = {
            "Linear Regression": LinearRegression(),
            "K-nn": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor()
        }

        error = {}
        lowest = math.inf
        modelName = None

        for name, algo in models.items():
            model = algo
            model.fit(xtrain, ytrain)
            ypred = model.predict(xtest)
            mse = mean_squared_error(ypred, ytest)
            error[name] = mse

            with open(f"Supervised/{name.split()[0]}.pkl", "wb") as f:
                pkl.dump(model, f)

            if mse > 0 and lowest > mse:
                lowest = mse
                modelName = name

        st.header("Error Ploting (Mean Squared Error)")

        fig, ax = plt.subplots()
        ax.bar(error.keys(), error.values())
        ax.set_ylabel("Mean Squared Error")
        ax.set_title("Model Comparison")
        st.pyplot(fig)

        st.header(f"Predicting values by {modelName}")
        for column in columns:
            if column == output_col :
                continue
            elif column in label_mappings.keys() :
                st.selectbox(column, options=label_mappings[column].keys(), key=column)
            else :
                st.number_input(column, key=column)

        if st.button("Submit values"):
            answer = []
            for column in columns:
                if column == output_col :
                    continue
                elif column in label_mappings.keys() :
                    answer.append(label_mappings[column][st.session_state[column]])
                else :
                    answer.append(st.session_state[column])

            with open(f"Supervised/{modelName.split()[0]}.pkl", "rb") as f:
                ML_model = pkl.load(f)

            predicted_value = ML_model.predict([answer])
            st.write(modelName)
            st.success(f"{output_col} : {int(predicted_value)}")


def unsupervised():
    st.title("Unsupervised Learning")
    file = st.file_uploader('Upload your data file', type=[".csv"])

    if file:
        st.header("Analysis")
        df = pd.read_csv(file)

        analysis = st.selectbox("Select the choice", options=["None", "Describe", "Columns", "information"])
        if analysis != "None":
            analyse(analysis, df)

        df = df.dropna()

        label_mappings = {}
        for column in df.columns:
            if df[column].dtype == object:
                label = LabelEncoder()
                df[column] = label.fit_transform(df[column])
                mapping_dict = dict(zip(label.classes_, label.transform(label.classes_)))
                label_mappings[column] = mapping_dict

        with open("Unsupervised/label_mappings.pkl", "wb") as f:
            pkl.dump(label_mappings, f)

        st.header("K-Means Clustering")
        if st.checkbox("Apply K-Means"):
            numeric_df = df.select_dtypes(include=['float64', 'int64'])

            if numeric_df.shape[1] < 2:
                st.warning("Need at least 2 numerical columns for clustering.")
                return

            st.subheader("Elbow Method - Find Optimal Number of Clusters")
            wcss = []
            K_range = range(1, 11)
            for i in K_range:
                km = KMeans(n_clusters=i, random_state=42)
                km.fit(numeric_df)
                wcss.append(km.inertia_)

            fig_elbow, ax_elbow = plt.subplots()
            ax_elbow.plot(K_range, wcss, 'bo-')
            ax_elbow.set_xlabel("Number of Clusters (k)")
            ax_elbow.set_ylabel("WCSS (Inertia)")
            ax_elbow.set_title("Elbow Method")
            ax_elbow.grid(True)
            st.pyplot(fig_elbow)

            val_true = st.text_input("Enter the text which will display if the output is 1 : ")
            val_false = st.text_input("Enter the text which will display if the output is 0 : ")
            model = KMeans(n_clusters=2, random_state=42)
            model.fit(numeric_df)

            with open("Unsupervised/KMeans.pkl", "wb") as f:
                pkl.dump(model, f)

            columns = list(numeric_df.columns)
            for column in columns:
                if column in label_mappings.keys() :
                    st.selectbox(column, options=label_mappings[column].keys(), key=column)
                else :
                    st.number_input(f"Enter value for {column}", key=column)

            if st.button("Predict"):
                user_input = []
                for column in columns :
                    if column in label_mappings.keys() :
                        user_input.append(label_mappings[column][st.session_state[column]])
                    else :
                        user_input.append(st.session_state[column])

                with open("Unsupervised/KMeans.pkl", "rb") as f:
                    k_model = pkl.load(f)

                prediction = k_model.predict([user_input])
                st.write(prediction)
                st.write(f"{val_true if prediction == 1 else val_false}")


if active == "Home" :
    home()

elif active == "Currency Converter" :
    currencyConverter()

elif active == "Supervised Learning" :
    supervised()

elif active == "Unsupervised Learning" :
    unsupervised()

