"""
Flask app entrypoint for Vercel deployment
"""
from api_server import app

# This is the WSGI application for Vercel
if __name__ == '__main__':
    app.run()
