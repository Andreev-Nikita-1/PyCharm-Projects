# encoding: UTF-8
import argparse
import cherrypy
import psycopg2 as pg_driver
from sqlalchemy import create_engine

from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata
import json

parser = argparse.ArgumentParser(description='Hello DB web application')
parser.add_argument('--pg-host', help='PostgreSQL host name', default='localhost')
parser.add_argument('--pg-port', help='PostgreSQL port', default=5432)
parser.add_argument('--pg-user', help='PostgreSQL user', default='postgres')
parser.add_argument('--pg-password', help='PostgreSQL password', default='')
parser.add_argument('--pg-database', help='PostgreSQL database', default='postgres')

args = parser.parse_args()


@cherrypy.expose
class App(object):

    @cherrypy.expose
    def dump(self):
        global metadata
        """ Returns the entire content of a database as lists of dicts"""
        engine = create_engine(
            f'postgresql://{args.pg_user}:{args.pg_password}@{args.pg_host}:{args.pg_port}/{args.pg_database}')
        meta = metadata
        meta.reflect(bind=engine)  # http://docs.sqlalchemy.org/en/rel_0_9/core/reflection.html
        result = {}
        for table in meta.sorted_tables:
            result[table.name] = [dict(row) for row in engine.execute(table.select())]
        cherrypy.response.headers['Content-Type'] = 'text/plain; charset=utf-8'
        return json.dumps(result, indent=4, sort_keys=True, default=str)

    @cherrypy.expose
    def set_rating(self, visit_id, rating_spec, apartment_id="redundant parameter"):
        ratings_list = rating_spec.split(',')
        ratings = dict()
        for i in range(0, len(ratings_list), 2):
            ratings[int(ratings_list[i])] = ratings_list[i + 1]
        with pg_driver.connect(user=args.pg_user, password=args.pg_password, host=args.pg_host,
                               port=args.pg_port, database=args.pg_database) as db:
            cur = db.cursor()
            cur.execute("""
            INSERT INTO HousingReview(contract_id, location_rate, cleanliness_rate, friendliness_rate, review)
            VALUES (%s, %s, %s, %s, 'automated review via set_rating');
            """, (visit_id, ratings[0], ratings[1], ratings[2]))
        return "success, go to /dump and check"

    @cherrypy.expose
    def tax_change(self, country_id, tax_rate_percents, week_number):
        with pg_driver.connect(user=args.pg_user, password=args.pg_password, host=args.pg_host,
                               port=args.pg_port, database=args.pg_database) as db:
            cur = db.cursor()
            cur.execute("""
            UPDATE PricePerWeek
            SET cost = cost * (1 + %s::Numeric(10, 2) / 100)
            WHERE week_number >= %s AND conditions_id IN (
            SELECT RC.id FROM
            RentConditions RC
            JOIN Housing H ON H.id = RC.housing_id
            WHERE H.country_id = %s
            ); 
            """, (tax_rate_percents, week_number, country_id))
        return "success, go to /dump and check"

    @cherrypy.expose
    def rating(self, apartment_id=None):
        with pg_driver.connect(user=args.pg_user, password=args.pg_password, host=args.pg_host,
                               port=args.pg_port, database=args.pg_database) as db:
            cur = db.cursor()
            if apartment_id is None:
                cur.execute("""
                SELECT C.country,
                    D.title,
                    (D.location_rate + D.cleanliness_rate + D.friendliness_rate) / 3, 
                    D.average_cost 
                FROM (SELECT H.country_id AS country_id,
                        H.title AS title,
                        AVG(HR.location_rate) AS location_rate,
                        AVG(HR.cleanliness_rate) AS cleanliness_rate,
                        AVG(HR.friendliness_rate) AS friendliness_rate,
                        AVG(PPW.cost) AS average_cost
                    FROM HousingReview HR JOIN RentContract RCT ON RCT.id = HR.contract_id
                    JOIN RentConditions RCS ON RCS.id = RCT.conditions_id
                    JOIN PricePerWeek PPW ON RCS.id = PPW.conditions_id
                    JOIN Housing H ON H.id = RCS.housing_id
                    GROUP BY H.id) AS D
                JOIN Country C ON C.id = D.country_id
                ORDER BY D.country_id ASC,
	                D.location_rate+D.cleanliness_rate+D.friendliness_rate DESC,
	                D.title ASC;""")
            else:
                cur.execute("""
                SELECT
                    H.title AS title,
                    AVG(HR.location_rate) AS location_rate,
                    AVG(HR.cleanliness_rate) AS cleanliness_rate,
                    AVG(HR.friendliness_rate) AS friendliness_rate
                FROM HousingReview HR JOIN RentContract RCT ON RCT.id = HR.contract_id
                RIGHT JOIN RentConditions RCS ON RCS.id = RCT.conditions_id
                JOIN Housing H ON H.id = RCS.housing_id
                WHERE H.id = %s
                GROUP BY H.id;""", apartment_id)

            result = []
            data = cur.fetchall()
            for p in data:
                result.append(p)
            cherrypy.response.headers['Content-Type'] = 'text/plain; charset=utf-8'
            return "\n".join([str(p) for p in data])


if __name__ == '__main__':
    cherrypy.quickstart(App())
