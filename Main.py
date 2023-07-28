# Libraries To be installed
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Page attributes
st.set_page_config(page_title="Loan Approval", page_icon="💰", layout='wide')

# For safety of code
# st.markdown(
#     """
#     <style>
#     .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
#     .styles_viewerBadge_1yB5, .viewerBadge_link__1S137,
#     .viewerBadge_text__1JaDK {
#         display: none;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

def hide_github_logo():
    hide_css = """
        <style>
        .css-1wbqy5l.e3g6aar1 {
            display: none;
        }
        </style>
    """
    st.markdown(hide_css, unsafe_allow_html=True)

hide_github_logo()

# Authentication page
def login_page():

    # Authentication check function
    def check_login(username, password):
        if username == "admin" and password == "akash1":
            return True
        else:
            return False
        
    st.title(":violet[Login Page]")

    # Authentication details
    username = st.text_input(":orange[Username]")
    password = st.text_input(":orange[Password]", type="password")


    if st.button("Login"):
        if check_login(username, password):
            st.success("Logged in successfully!")
            st.write("Please wait model is running.......")
            AADI_DATAFRAME = pd.read_csv('loan_approval_dataset.csv')
            st.session_state['load_data'] = {'data':AADI_DATAFRAME}
            st.session_state.stage = 1
            st.button("Proceed",on_click=None,key='Proceed')
        else:
            st.error("Invalid username or password")

# Main Page 
def Default_Page():

    # Home Page having project description
    def Project_Description():
        st.title(":red[Loan Approval Prediction Model]")
        st.subheader(":green[This model having efficiency 91.947%]")
        st.subheader(":blue[Description:]")
        st.write("The Loan Approval Prediction model based on Logistic Regression is a powerful and accurate machine learning solution designed to assist financial institutions in making informed decisions about loan applications. By leveraging historical loan data, this model can effectively predict whether a loan applicant is likely to be approved or denied, thereby streamlining the loan approval process and reducing the risk of default.")
        st.subheader(":blue[Key Features:]")
        st.markdown(":orange[Data Preprocessing:]:Before training the model, the data is thoroughly preprocessed to handle missing values, normalize numerical features, and encode categorical variables. This ensures that the model receives clean and standardized input, leading to improved accuracy and robustness.")
        st.markdown(':orange[Logistic Regression Algorithm:]:The model is built on the foundation of the Logistic Regression algorithm, which is a widely-used and interpretable statistical method for binary classification problems. It excels in scenarios where the target variable is binary, making it an ideal choice for predicting loan approval outcomes.')
        st.write("In conclusion, the Loan Approval Prediction model based on Logistic Regression empowers financial institutions to make data-driven and well-informed decisions when evaluating loan applications. By leveraging the power of machine learning, it helps optimize loan approval processes, mitigate risk, and ultimately support the financial well-being of both lenders and borrowers.")
        
    # About the model
    def Model_Info():
        st.title(":red[Logistic Regression Model]")
        st.header(":green[Libraries Requirement]")
        st.subheader(":orange[For ML Model]")
        st.write("1. from sklearn.model_selection using train_test_split")
        st.write("2. from sklearn.preprocessing using StandarScalar")
        st.write("3. from sklearn.linear_model using LogisticRegression")
        st.write("4. from sklearn.metrices using classification_report")
        st.write("5. pickle")
        st.write("6. matplotlib.pyplot")
        st.write("7. numpy")
        st.write("8. pandas")

        st.subheader(":orange[For website]")
        st.write("1. streamlit")
        st.write("2. pandas")
        st.write("3. matplotlib.pyplot")
        st.write("4. pickle")

        st.write(":violet[Model is fit with the help of grid search]")
    # Data Preview and graphs
    def Data_Preview():
        st.title(":red[Generate Your own Graph]")
        
        df = st.session_state.load_data['data']
        if st.checkbox(":orange[Data Preview]",key = 'preview'):
            st.dataframe(df)
        x_axis_val = st.selectbox('Select X-Axis Value', options = df.columns)
        y_axis_val = st.selectbox('Select Y-Axis Value', options = df.columns)

        def plot(x,y):
            # fig = sns.set_style('whitegrid')
            # plot = sns.countplot(x = x,hue = y,data = df,palette = 'rainbow')
            # st.pyplot(plot.fig)
            fig = plt.figure(figsize=(10,6))

            ax = fig.add_subplot(111)
            df[x].hist(bins = 30,color = 'darkred',alpha = 0.7,ax = ax)

            ax.set_xlabel(x,fontsize = 18)
            ax.set_ylabel(y,fontsize = 18)
            ax.set_title(f"Graph between {x} and {y}",fontsize = 22)
            fig
        
        st.button("Plot",on_click=plot,args = [x_axis_val,y_axis_val],key = 'plot')
    
    # Model Prediction
    def Model_Pridiction():
        st.title(":red[Model Prediction]")
        st.header(":green[Provide Required Data]")
        left,middle,right = st.columns(3)
        st.session_state.Dependents = left.selectbox(":orange[Depenedents]",options = [1,2,3,4,5],key = 'dependents')
        st.session_state.Education = left.selectbox(":orange[Education]",options = ['Graduate','Not Graduate'],key = 'education')
        if st.session_state.Education == 'Not Graduate':
            st.session_state.Education_conv = 1
        else:
            st.session_state.Education_conv = 0
        st.session_state.Self_Employed = left.selectbox(":orange[Self Employed]",options = ['Yes','No'],key = 'self_employed')
        if st.session_state.Self_Employed == 'Yes':
            st.session_state.Self_Employed_conv = 1
        else:
            st.session_state.Self_Employed_conv = 0
        st.session_state.Loan_Amount_Term = left.selectbox(":orange[Loan Amount Term]",options = [360,180,480,300,240,120,84,60,36,12],key = 'Loan_amount_term')
        st.session_state.cibil_score = right.number_input(":violet[Cibil Score]",min_value = 350, max_value= 900,step=1,key ="cibil")
        st.session_state.income_annum = right.number_input(":violet[Annual Income]",min_value = 100000, max_value= 9900000,step=100000, key = 'income')
        st.session_state.loan_amount = right.number_input(":violet[Loan Amount]",min_value = 300000, max_value= 39500000,step=100000, key = "loanamount")
        st.session_state.resedential_asset = middle.number_input(":blue[Resedential Asset]",min_value = 0, max_value= 29100000,step=100000, key = "resedential")
        st.session_state.commercial_asset = middle.number_input(":blue[Commercial Asset]",min_value = 0, max_value= 29100000,step=100000, key = "commercial")
        st.session_state.luxury_asset = middle.number_input(":blue[Luxury Asset]",min_value = 0, max_value= 29100000,step=100000, key = "luxury")
        st.session_state.bank_asset = middle.number_input(":blue[Bank Asset]",min_value = 0, max_value= 29100000,step=100000, key = "bank")

        def predict():
            with open('classifier.pkl','rb') as file:
                clf = pickle.load(file)
                # st.write(st.session_state)   
                prediction = clf.predict([[st.session_state.Dependents,st.session_state.income_annum,st.session_state.loan_amount,st.session_state.Loan_Amount_Term,st.session_state.cibil_score,st.session_state.resedential_asset,st.session_state.commercial_asset,st.session_state.luxury_asset,st.session_state.bank_asset,st.session_state.Education_conv,st.session_state.Self_Employed_conv]])

                if prediction == 1:
                    st.header(":green[Congratulations!! Your Loan is Approved]")
                else:
                    st.header(":red[Sorry!! Your request is denied]")
        st.button("Predict",on_click=predict,key = 'predict')
    
    # logout Function
    def logout():
        st.subheader(":violet[Are you sure you want to logout??]")
        st.session_state.pop('stage')
        st.session_state.pop('status')
        def cancel():
            st.session_state.stage = 0
            st.session_state.status = 0
        left,middle,right = st.columns(3)
        left.button("Cancel",on_click = cancel,key = 'cancel')
        right.button(":red[logout]",key = 'confirm',on_click=None)
    def a():
        st.session_state.status = 0
    def b():
        st.session_state.status = 1
    def c():
        st.session_state.status = 2
    def d():
        st.session_state.status = 3
    def e():
        st.session_state.status = 4

    # Sidebar attributes and buttons
    st.sidebar.title(":green[Welcome admin]")
    st.sidebar.button(":violet[Home(Project Description)]",on_click=a,key='Project Description')
    st.sidebar.button(":violet[Model Info]",on_click=b,key='Model_Info')
    st.sidebar.button(":violet[Data Preview]",on_click=c,key='Data_Preview')
    st.sidebar.button(":violet[Model Prediction]",on_click=d,key='Model_Pridiction')
    st.sidebar.button(":violet[Logout]",on_click=e,key='logout button')

    # For the purpose navigation inside default page
    pagemap = {0:Project_Description,1:Model_Info,2:Data_Preview,3:Model_Pridiction,4:logout}
    if 'status' not in st.session_state.keys():
        st.session_state.status = 0
        pagemap[st.session_state.status]()
    else:
        pagemap[st.session_state.status]()

# It defines whether page run for authentication or for main page
if 'stage' not in st.session_state.keys():
    login_page()
else:
    Default_Page()
