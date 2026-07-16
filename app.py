from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from predictor import get_predictor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'phishguard-secret-key-2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize predictor
predictor = get_predictor()


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/url-checker')
def url_checker():
    """URL phishing checker page"""
    return render_template('url_checker.html')


@app.route('/email-checker')
def email_checker():
    """Email phishing checker page"""
    return render_template('email_checker.html')


@app.route('/sms-checker')
def sms_checker():
    """SMS phishing checker page"""
    return render_template('sms_checker.html')


@app.route('/qr-checker')
def qr_checker():
    """QR code phishing checker page"""
    return render_template('qr_checker.html')


@app.route('/api/check-url', methods=['POST'])
def check_url():
    """API endpoint to check if a URL is phishing"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()

        if not url:
            return jsonify({
                'success': False,
                'error': 'Please provide a URL'
            }), 400

        # Make prediction
        result = predictor.predict_url(url)

        if 'error' in result and result['prediction'] is None:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500

        return jsonify({
            'success': True,
            'result': result
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/check-email', methods=['POST'])
def check_email():
    """API endpoint to check if an email is phishing"""
    try:
        data = request.get_json()
        email_text = data.get('email_text', '').strip()

        if not email_text:
            return jsonify({
                'success': False,
                'error': 'Please provide email text'
            }), 400

        # Make prediction
        result = predictor.predict_text(email_text)

        if 'error' in result and result['prediction'] is None:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500

        return jsonify({
            'success': True,
            'result': result
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/check-sms', methods=['POST'])
def check_sms():
    """API endpoint to check if an SMS is phishing"""
    try:
        data = request.get_json()
        sms_text = data.get('sms_text', '').strip()

        if not sms_text:
            return jsonify({
                'success': False,
                'error': 'Please provide SMS text'
            }), 400

        # Make prediction (using same model as email)
        result = predictor.predict_text(sms_text)

        if 'error' in result and result['prediction'] is None:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500

        return jsonify({
            'success': True,
            'result': result
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/check-qr', methods=['POST'])
def check_qr():
    """API endpoint to check if a QR code contains a phishing URL"""
    try:
        if 'qr_image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Please upload a QR code image'
            }), 400

        file = request.files['qr_image']

        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400

        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Make prediction
            result = predictor.predict_qr_code(filepath)

            # Clean up uploaded file
            os.remove(filepath)

            if 'error' in result and result['prediction'] is None:
                return jsonify({
                    'success': False,
                    'error': result.get('error', 'Unknown error'),
                    'message': result.get('message')
                }), 400

            return jsonify({
                'success': True,
                'result': result
            })

        except Exception as e:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
            raise e

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/graphs')
def graphs():
    """Model performance graphs page"""
    return render_template('graphs.html')


@app.route('/graphs/<filename>')
def serve_graph(filename):
    """Serve graph images from the graphs folder"""
    graphs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'graphs')
    return send_from_directory(graphs_dir, filename)


@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')


@app.errorhandler(404)
def not_found(e):
    """404 error handler"""
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(e):
    """500 error handler"""
    return render_template('500.html'), 500


if __name__ == '__main__':
    print("=" * 60)
    print("PhishGuard - AI-Powered Phishing Detection System")
    print("=" * 60)
    print("\nStarting Flask application...")
    print("Access the application at: http://127.0.0.1:5000")
    print("\nPress CTRL+C to quit")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)
