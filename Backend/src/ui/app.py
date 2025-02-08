app = Flask(__name__)

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    detector = DeepfakeDetector()
    # Start video capture and detection loop
