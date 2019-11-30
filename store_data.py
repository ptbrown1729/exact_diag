# run clusters and store results in MySql database
import mysql.connector

db_name = "edresults"

# create db
# cursor = db.cursor()
# cursor.execute("CREATE DATABASE edresults")

# print results
# cursor.execute("SHOW DATABASES")
# databases = cursor.fetchall()
# print(databases)

# connect
db = mysql.connector.connect(
    host = "localhost",
    user = "root",
    passwd = "b4krl4b",
    database = db_name
)

cursor = db.cursor()
cursor.execute("CREATE TABLE users (id INT(11) NOT NULL AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255), user_name VARCHAR(255))")

# show tables
cursor.execute("SHOW TABLES")
tables = cursor.fetchall()

# show columns of table "users"
cursor.execute("DESC users")
print(cursor.fetchall())

# insert row into table "users"
query = "INSERT INTO users (name, user_name) VALUES (%s, %s)"
values = ("Mary Margaret", "mm12663737")
cursor.execute(query, values)
db.commit()

# print all
query = "SELECT * FROM users"
cursor.execute(query)
print(cursor.fetchall())

# select columns
query = "SELECT user_name FROM users"
cursor.execute(query)
print(cursor.fetchall())
