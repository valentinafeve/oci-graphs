import os
import oracledb
from dotenv import load_dotenv

load_dotenv()

def get_conn():
    # Create Oracle Database 23ai connection 
    ORA_USER = os.getenv("ORA_USER")
    ORA_PASS = os.getenv("ORA_PASS")
    ORA_DSN  = os.getenv("ORA_DSN")
    wallet_path = os.path.join(os.getcwd(), "wallet")
    WALLET_PASSWORD = os.getenv("WALLET_PASSWORD")
    CONFIG_DIR = wallet_path

    connection = oracledb.connect(
        user            = ORA_USER,
        password        = ORA_PASS,
        dsn             = ORA_DSN,
        config_dir      = CONFIG_DIR,
        wallet_location = wallet_path,
        wallet_password = WALLET_PASSWORD
    )
    return connection
