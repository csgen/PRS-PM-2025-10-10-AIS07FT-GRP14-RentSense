import os
from dotenv import load_dotenv


def get_database_url_async():
    load_dotenv()
    db_url = os.getenv("CLOUD_DATABASE_URL")
    print(f"数据库URL: {db_url}")
    return db_url

def get_openmap_token():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(current_dir, '.env.openmap')
    load_dotenv(env_path)
    url_first = os.getenv('OPEN_MAP_URL_FIRST')
    url_second = os.getenv('OPEN_MAP_URL_SECOND')
    url_routing = os.getenv('OPEN_MAP_ROUTING_URL')
    token = os.getenv('OPEN_MAP_TOKEN')
    return url_first, url_second, url_routing, token

def get_openmap_library_url(): # 查询图书馆信息用
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(current_dir, '.env.openmap')
    load_dotenv(env_path)
    library_url = os.getenv('OPEN_MAP_LIBRARY_URL')
    token = os.getenv('OPEN_MAP_TOKEN')
    return library_url, token

def get_neo4j_info():
    url = os.getenv('NEO4J_URI')
    username = os.getenv('NEO4J_USERNAME')
    password = os.getenv('NEO4J_PASSWORD')
    return url, username, password

if __name__ == "__main__":
    # print(os.getcwd())
    # get_database_url()
    url, token  = get_openmap_token()
    print(f'url:{url}')