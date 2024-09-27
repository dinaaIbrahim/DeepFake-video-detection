import pickle
from flask import Flask, render_template, request, redirect , session , url_for
import os
from flask_sqlalchemy import SQLAlchemy
from flask_mysqldb import MySQL
from trial import process_classify_video

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# path = ''
# text = process_classify_video(path)


app = Flask(__name__,template_folder='templates',static_folder='static')

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/deepfake'
app.config["UPLOAD_FOLDER"] = "uploads"
db = SQLAlchemy(app)
# dina =''
class user(db.Model):
    id = db.Column(db.Integer, nullable = False, autoincrement = True)
    name = db.Column(db.String(255) , nullable = False)
    email = db.Column(db.String(255) , primary_key=True ,nullable= False)
    password = db.Column(db.String(255) ,nullable = False)

class video(db.Model):
    id = db.Column(db.Integer, primary_key=True , autoincrement = True)
    video_path = db.Column(db.String(255) , nullable = False)
    email = db.Column(db.String(255) , nullable= False)
    detection = db.Column(db.String(255) ,nullable = False)
# Change this to a secure secret key
app.config['SECRET_KEY'] = 'ddddssss'

# Load the pickled model


# return render_template('result.html',data=prediction)

@app.route('/')
def index():

    return render_template('index.html')


@app.route('/result')
def result():
    dina = session.get('dina', 'Upload Video First')
    words_to_find = ["Real", "Fake"]
    found_words = [word for word in words_to_find if word in dina]

    # image_file = 'real.png'  # default image
    if "Real" in found_words:
        image_file = 'realllll.png'
    elif "Fake" in found_words:
        image_file = 'real.png'  # Assuming you have a 'fake.png'

    return render_template('result.html',dina = session['dina'] , image_file=image_file)

@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = ' '
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']
        user_data = user.query.filter_by(email=email).first()
        if user_data and user_data.password==password:
            session['loggedin'] = True
            session['email'] = user_data.email
            return redirect(url_for('upload'))
        else:
            msg= 'incorrect username or password'

    return render_template ('login.html' ,msg=msg)

@app.route('/register' ,methods=['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST' and 'name'in request.form and 'email' in request.form and 'password' in request.form and "confirm_password" in request.form :
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        data = user(id=2,name=name , email=email,password=password)
        user_duplicate = user.query.filter_by(email=email).first()
        if user_duplicate:
            msg = 'User already registered'
        elif password ==confirm_password : 
            db.session.add(data)
            db.session.commit()
            return redirect(url_for('login'))
        else :
            msg = "Password doesn't match"
        
        
            
    return render_template('register.html', msg=msg)




@app.route('/upload', methods=['GET', 'POST'])
def upload():
    # if request.method == 'POST' :
    #     video = request.form['video']
    #     if 'video' not in request.files:
    #         return redirect(request.url)

    #     video = request.files['video']

    #     if video.filename == '':
    #         return redirect(request.url)
        
    #     file_path = os.path.join(os.path.dirname(__file__), 'soft_voting_results.pkl')

    #     with open(file_path, 'rb') as model_file:

    #         model = pickle.load(model_file)

    #     # Function to detect video authenticity using a pickled deep learning model
    #     def detect_fake_video(video):
            
    #         prediction = model.predict(video)  
    #         return prediction


    #     result_message = detect_fake_video(video)
    #     return render_template('result.html', result_message=result_message)

    # return render_template('upload.html')

    if request.method == 'POST':
        if 'loggedin' not in session:
            return redirect(url_for('login'))
        email = session['email']
        file = request.files['video']
        main_file_name, file_extension = os.path.splitext(file.filename)
        file_extension = file_extension[1:].lower()  # Remove the leading dot from the extension
        new_file_name = main_file_name + '.' + file_extension
                # Create the full file path
        # file_path = 
        # print(new_file_name)
        # print(file_path)
        # # Save the file
        # uploads_path = app.config["UPLOAD_FOLDER"]
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], new_file_name)
        file.save(file_path)
        
        print(file_path)
        dina = process_classify_video(file_path)
        print(dina)
        session['dina'] = dina
        
        entry = video(email=email, video_path=new_file_name , detection = dina)
        db.session.add(entry)
        db.session.commit()
        return redirect(url_for('result'))
    else:
        return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)

