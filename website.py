import streamlit as st
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import bcrypt
import time
from PIL import Image
import io
import torch
import torch.nn as nn
import numpy as np
import torchvision
import os
import sys
import torchvision.transforms as transforms
from PIL import Image
import json

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

# Function to retrieve and display images
def display_images(images_collection):
    images = images_collection.find()
    for image_record in images:
        image_data = image_record["image_data"]
        image = Image.open(io.BytesIO(image_data))
        st.image(image, caption=image_record["filename"], width=150)

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

    # Display previously uploaded images
    st.subheader("Your Uploaded Images:")
    display_images(images_collection)

    # Allow user to upload new images
    st.subheader("Upload New Images")
    files = st.file_uploader("Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="file_uploader")
    
    if st.button("Submit Images"):
        if files:
            upload_images(images_collection, files)
            st.experimental_rerun()  # Refresh to show newly uploaded images immediately

    # Ask if user wants to upload more images
    add_more = st.radio("Do you want to upload more images?", ["Yes", "No"], key="add_more")
    if add_more == "No":
        st.success("Thank you! Goodbye!")
        time.sleep(2)  # Pause for 2 seconds
        st.session_state["username"] = None  # Log out the user
        st.experimental_rerun()  # Clear session state on logout

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
                    st.experimental_rerun()  # Rerun to load logged-in view

# # CNN WORK (BOGDAN'S PART):

# model_path = 'mushroom_classfier1.pth'


# class MushroomClassfier(nn.Module):
#     def __init__(self):
#         super(MushroomClassfier, self).__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),

#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),

#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),

#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),

#             nn.Dropout(0.5),
#         )
#         self.fc_layers = nn.Sequential(
#             nn.Linear(256 * 14 * 14, 512),
#             nn.ReLU(),
#             nn.Dropout(0.5),  
#             nn.Linear(512, 128),
#             nn.ReLU(),
#             nn.Linear(128, 2)  
#         )

#     def forward(self, x):
#         x = self.conv_layers(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc_layers(x)
#         return x


# # Load the model(Once it is finished):
# def load_model(model_path):
#     model = MushroomClassfier()
#     model.load_state_dict(torch.load(model_path, weights_only=True))
#     model.eval()
#     return model




# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  
#     # The below codes that are commented out were used for data augmentation, but I ultimately decided to take them out as it was more useful to keep the dataset normal. --> Kept for reasons of making model potetnially more accurate in future.
#     # transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
#     # transforms.RandomRotation(15),      # Rotate images by up to 15 degrees
#     # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     transforms.ToTensor(),
# ])

# def classify(image_path, model, transform):
#     image = Image.open(image_path).convert('RGB')
#     image = transform(image)
#     image = image.unsqueeze(0)

#     # Make a prediction:
#     output = model(image)
#     _, predicted = torch.max(output, 1)
#     class_index = predicted.item()
#     idx_to_class = {0: 'edible', 1:'poisonous'}
#     class_name = idx_to_class[class_index]
#     return class_name
