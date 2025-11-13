from pymongo import MongoClient

def test_mongodb_connection():
    try:
        # Try to connect to MongoDB
        client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=2000)
        # Force a connection attempt
        client.server_info()
        print("✅ MongoDB is running and accessible")
        
        # Test database and collection creation
        db = client['starbucks_stock']
        users = db['users']
        print("✅ Can access database and collections")
        
        # Test user count
        user_count = users.count_documents({})
        print(f"ℹ️ Current number of users: {user_count}")
        
        return True
    except Exception as e:
        print(f"❌ MongoDB Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_mongodb_connection()