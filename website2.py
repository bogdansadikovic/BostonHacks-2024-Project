import streamlit as st
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import bcrypt
import time

# MongoDB connection URI
uri = "mongodb+srv://dyesilyurt04:SkJzmjOdz6Ra4pE9@cluster0.bijaw.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))

# Function to create a user-specific database
def create_user_cluster(username):
    db = client[username]
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
        st.session_state["username"] = username  # Store username in session state
        return True
    st.error("Invalid username or password.")
    return False

# Function to upload multiple images
def upload_images(images_collection, files):
    for file in files:
        if file is not None:
            images_collection.insert_one({"image_data": file.read(), "filename": file.name})
    st.success("Images uploaded successfully!")

# Initialize session state variables
if "username" not in st.session_state:
    st.session_state["username"] = None

# Streamlit UI
st.title("Mushroom Reserve")

# Check if a user is logged in
if st.session_state["username"]:
    st.success(f"Welcome, {st.session_state['username']}!")

    # Display the image upload interface for logged-in users
    db = create_user_cluster(st.session_state["username"])
    _, images_collection = create_collections(db)

    files = st.file_uploader(
        "Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="file_uploader"
    )
    
    if st.button("Submit Images"):
        if files:
            upload_images(images_collection, files)

    add_more = st.radio("Do you want to upload more images?", ["Yes", "No"], key="add_more")
    if add_more == "No":
        st.success("Thank you! Goodbye!")
        time.sleep(2)  # Pause for 2 seconds
        st.session_state["username"] = None  # Log out the user

else:
    # User selection for register or login
    option = st.selectbox("Choose an option:", ["Select an option", "Register", "Login"])

    if option == "Register":
        st.header("Register for a New Account")
        username = st.text_input("Enter a username", key="register_username")
        password = st.text_input("Enter a password", type="password", key="register_password")
        
        if st.button("Register"):
            if username and password:
                db = create_user_cluster(username)
                users_collection, _ = create_collections(db)
                register_user(users_collection, username, password)

    elif option == "Login":
        st.header("Login to Your Account")
        username = st.text_input("Enter your username", key="login_username")
        password = st.text_input("Enter your password", type="password", key="login_password")
        
        if st.button("Login"):
            if username and password:
                db = create_user_cluster(username)
                users_collection, _ = create_collections(db)
                if login(users_collection, username, password):
                    st.success("Login successful!")
