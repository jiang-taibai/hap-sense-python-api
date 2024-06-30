import mysql.connector
from mysql.connector import Error


class Statistics:
    def __init__(self, date, population, households, planting_households):
        self.date = date
        self.population = population
        self.households = households
        self.planting_households = planting_households

    def __str__(self):
        return f"Statistics({self.date}, {self.population}, {self.households}, {self.planting_households})"

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def from_db_record(record):
        return Statistics(record[0], record[1], record[2], record[3])

    def to_array(self):
        return [self.date, self.population, self.households, self.planting_households]


def get_data():
    all_statistics = []
    try:
        connection = mysql.connector.connect(
            host='coderjiang.com',
            port='20000',
            user='project_hap_sense_user_001',
            password='hap123',
            database='hap-sense',
        )
        if connection.is_connected():
            print("Connected to MySQL Server version ", connection.get_server_info())
        cursor = connection.cursor()
        query = "SELECT * FROM statistics;"
        cursor.execute(query)
        rows = cursor.fetchall()
        print("Total number of rows in statistics is: ", cursor.rowcount)
        for row in rows:
            all_statistics.append(Statistics.from_db_record(row))

    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")
        return all_statistics


def main():
    get_data()


if __name__ == '__main__':
    main()
