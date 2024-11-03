import streamlit as st
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import bcrypt

# MongoDB connection URI
uri = "mongodb+srv://dyesilyurt04:SkJzmjOdz6Ra4pE9@cluster0.bijaw.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))

# Function to create a user-specific database
def create_user_cluster(username):
    db = client[username]  # Create a user-specific database
    return db

# Function to create collections for users and images
def create_collections(db):
    users_collection = db["users"]
    images_collection = db["images"]
    return users_collection, images_collection

# Function to register a new user
def register_user(users_collection, username, password):
    if users_collection.find_one({"username": username}):
        st.error("Username already exists.")
        return False

    hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    users_collection.insert_one({"username": username, "password": hashed_password})
    st.success("User registered successfully.")
    return True

# Function to log in an existing user
def login(users_collection, username, password):
    user = users_collection.find_one({"username": username})
    if user and bcrypt.checkpw(password.encode("utf-8"), user["password"]):
        return True
    st.error("Invalid username or password.")
    return False

# Function to upload multiple images
def upload_images(images_collection, files):
    for file in files:
        if file is not None:
            images_collection.insert_one({"image_data": file.read(), "filename": file.name})
    st.success("Images uploaded successfully!")

# Streamlit UI
st.title("Mushroom Reserve")

# User selection for register or login
option = st.selectbox("Choose an option:", ["Select an option", "Register", "Login"])

if option == "Register":
    st.header("Register for a New Account")
    username = st.text_input("Enter a username")
    password = st.text_input("Enter a password", type="password")
    
    if st.button("Register"):
        if username and password:
            db = create_user_cluster(username)
            users_collection, images_collection = create_collections(db)
            if register_user(users_collection, username, password):
                st.write("You can now upload images to your account.")
                while True:
                    files = st.file_uploader("Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
                    
                    if files and st.button("Submit Images"):
                        upload_images(images_collection, files)
                    
                    add_more = st.radio("Do you want to upload more images?", ["Yes", "No"])
                    if add_more == "No":
                        break
        else:
            st.error("Please enter both a username and a password.")

elif option == "Login":
    st.header("Login to Your Account")
    username = st.text_input("Enter your username")
    password = st.text_input("Enter your password", type="password")
    
    if st.button("Login"):
        if username and password:
            db = create_user_cluster(username)
            users_collection, images_collection = create_collections(db)
            if login(users_collection, username, password):
                st.success("Login successful!")
                st.write("You can now upload images to your account.")
                
                while True:
                    files = st.file_uploader("Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
                    
                    if files and st.button("Submit Images"):
                        upload_images(images_collection, files)
                    
                    add_more = st.radio("Do you want to upload more images?", ["Yes", "No"])
                    if add_more == "No":
                        break
            else:
                st.error("Login failed. Please check your credentials.")
        else:
            st.error("Please enter both a username and a password.")
