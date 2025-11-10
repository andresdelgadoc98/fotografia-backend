from src import create_app
from dotenv import load_dotenv
import os
load_dotenv()


PORT = int(os.getenv("PORT", 5001))
app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=False)