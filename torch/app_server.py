from wtforms import Form, validators, SubmitField, SelectField
from flask import Flask, render_template, request, send_file
import sys
# Tornado web server
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from recommend import recommend_songs, load_data

#Debug logger
import logging 
root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)

# Create app
app = Flask(__name__)

SONGS_CHOICES, images, labels, new_model = load_data()

class ReusableForm(Form):
    """User entry form for entering specifics for generation"""
    # Starting seed
    seed = SelectField("Choose an anchor song in the playlists to recommend:", validators=[
                     validators.InputRequired()], choices=SONGS_CHOICES)

    # Submit button
    submit = SubmitField("Enter")

# Route to stream music
@app.route('/<path:filename>')
def download_file(filename):
    return send_file(filename)

# Home page
@app.route("/", methods=['GET', 'POST'])
def home():
    """Home page of app with form"""
  
    # Create form
    form = ReusableForm(request.form)

    # On form entry and all conditions met
    if request.method == 'POST' and form.validate():
        # Extract information
        seed = request.form['seed']
        for i, k in SONGS_CHOICES:
            if i == seed:
                song = k
        print(song)
        if song != ' ':
            return_dict = recommend_songs(song, images, labels, new_model)
            general_Data = {
            'title': 'Music Player'}
            print(return_dict)
            stream_entries = return_dict
            return render_template('site.html', entries=stream_entries, **general_Data)

    # Send template information to index.html
    return render_template('index.html', form=form)

if __name__ == "__main__":
    port = 5000
    http_server = HTTPServer(WSGIContainer(app))
    logging.debug("Started Server, Kindly visit http://localhost:" + str(port))
    http_server.listen(port)
    IOLoop.instance().start()