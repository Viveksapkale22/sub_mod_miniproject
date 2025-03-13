from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_bcrypt import Bcrypt
from pymongo import MongoClient
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)
bcrypt = Bcrypt(app)

# MongoDB Configuration
MONGO_URI = "mongodb+srv://cluster0.mcjuw.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(
    MONGO_URI,
    tls=True,
    tlsCertificateKeyFile=r"templates\X509-cert-7398551624606348947.pem"
)
db = client["UserDB"]
users_collection = db["users"]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/register', methods=['POST'])
def register():
    """User registration route."""
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')

    # Check if username already exists
    if users_collection.find_one({'username': username}):
        flash("Username already exists. Please choose another.", "error")
        return redirect(url_for('home'))

    # Hash the password before storing
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    users_collection.insert_one({'username': username, 'email': email, 'password': hashed_password})

    flash("Registration successful! Please login.", "success")
    return redirect(url_for('home'))

@app.route('/login', methods=['POST'])
def login():
    """User login route."""
    username = request.form.get('username')
    password = request.form.get('password')

    # Verify user exists and password matches
    user = users_collection.find_one({'username': username})
    if user and bcrypt.check_password_hash(user['password'], password):
        session['user'] = user['username']
        flash("Login successful!", "success")
        return redirect(url_for('home'))
    else:
        flash("Invalid username or password. Please try again!", "error")
        return redirect(url_for('home'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("You have been logged out.", "info")
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
