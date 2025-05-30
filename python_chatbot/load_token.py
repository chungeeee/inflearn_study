from dotenv import load_dotenv
import os
load_dotenv()
TOKEN = os.getenv("MY_TOKEN")
print(TOKEN)
