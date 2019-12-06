# encoding: UTF-8
import argparse
import cherrypy
import psycopg2 as pg_driver

# coding: utf-8
from sqlalchemy import Boolean, CheckConstraint, Column, Date, Enum, ForeignKey, Integer, Numeric, Table, Text, UniqueConstraint, text
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata


class Account(Base):
    __tablename__ = 'account'

    id = Column(Integer, primary_key=True, server_default=text("nextval('account_id_seq'::regclass)"))
    first_name = Column(Text, nullable=False)
    second_name = Column(Text, nullable=False)
    email = Column(Text, nullable=False, unique=True)
    telephone_number = Column(Text, nullable=False, unique=True)
    sex = Column(Enum('male', 'female', name='sex'))
    birthday = Column(Date)
    photo_path = Column(Text)


class Country(Base):
    __tablename__ = 'country'
    __table_args__ = (
        CheckConstraint('commission >= (0)::numeric'),
    )

    id = Column(Integer, primary_key=True, server_default=text("nextval('country_id_seq'::regclass)"))
    country = Column(Text, nullable=False)
    commission = Column(Numeric(10, 2))


class Genre(Base):
    __tablename__ = 'genres'

    type = Column(Text, primary_key=True)


class Event(Base):
    __tablename__ = 'event'

    id = Column(Integer, primary_key=True, server_default=text("nextval('event_id_seq'::regclass)"))
    title = Column(Text, nullable=False)
    genre = Column(ForeignKey('genres.type'))
    country_id = Column(ForeignKey('country.id'), nullable=False)
    city = Column(Text, nullable=False)
    lat = Column(Numeric(7, 5), nullable=False)
    lon = Column(Numeric(8, 5), nullable=False)
    start_date = Column(Date, nullable=False)
    finish_date = Column(Date, nullable=False)

    country = relationship('Country')
    genre1 = relationship('Genre')


class Housing(Base):
    __tablename__ = 'housing'
    __table_args__ = (
        CheckConstraint('beds_number >= 0'),
        CheckConstraint('chambers_number >= 0'),
        CheckConstraint('habitant_capacity >= 0')
    )

    id = Column(Integer, primary_key=True, server_default=text("nextval('housing_id_seq'::regclass)"))
    owner_id = Column(ForeignKey('account.id'), nullable=False)
    country_id = Column(ForeignKey('country.id'), nullable=False)
    city = Column(Text, nullable=False)
    address = Column(Text, nullable=False)
    lat = Column(Numeric(7, 5), nullable=False)
    lon = Column(Numeric(8, 5), nullable=False)
    title = Column(Text, nullable=False)
    chambers_number = Column(Integer, nullable=False)
    beds_number = Column(Integer, nullable=False)
    habitant_capacity = Column(Integer, nullable=False)

    country = relationship('Country')
    owner = relationship('Account')


t_housingfeatures = Table(
    'housingfeatures', metadata,
    Column('housing_id', ForeignKey('housing.id'), nullable=False, unique=True),
    Column('wifi', Boolean, server_default=text("false")),
    Column('iron', Boolean, server_default=text("false")),
    Column('kitchen_items', Boolean, server_default=text("false")),
    Column('electric_kettle', Boolean, server_default=text("false")),
    Column('smoking_permission', Boolean, server_default=text("false")),
    Column('bath', Boolean, server_default=text("false")),
    Column('pool', Boolean, server_default=text("false")),
    Column('jacuzzi', Boolean, server_default=text("false")),
    Column('balcony', Boolean, server_default=text("false")),
    Column('conditioner', Boolean, server_default=text("false")),
    Column('metro_walking_access', Boolean, server_default=text("false")),
    Column('parking', Boolean, server_default=text("false"))
)


class Rentcondition(Base):
    __tablename__ = 'rentconditions'
    __table_args__ = (
        CheckConstraint('cleaning_price >= (0)::numeric'),
    )

    id = Column(Integer, primary_key=True, server_default=text("nextval('rentconditions_id_seq'::regclass)"))
    housing_id = Column(ForeignKey('housing.id'), nullable=False, unique=True)
    cleaning_price = Column(Numeric(10, 2))

    housing = relationship('Housing', uselist=False)


t_priceperweek = Table(
    'priceperweek', metadata,
    Column('conditions_id', ForeignKey('rentconditions.id'), nullable=False),
    Column('week_number', Integer, nullable=False),
    Column('cost', Integer, nullable=False),
    CheckConstraint('(week_number >= 0) AND (week_number <= 53)'),
    CheckConstraint('cost >= 0'),
    UniqueConstraint('conditions_id', 'week_number')
)


class Rentcontract(Base):
    __tablename__ = 'rentcontract'
    __table_args__ = (
        CheckConstraint('cost >= (0)::numeric'),
        CheckConstraint('person_number > 0')
    )

    id = Column(Integer, primary_key=True, server_default=text("nextval('rentcontract_id_seq'::regclass)"))
    conditions_id = Column(ForeignKey('rentconditions.id'), nullable=False)
    renter_id = Column(ForeignKey('account.id'), nullable=False)
    person_number = Column(Integer, nullable=False)
    start_date = Column(Date, nullable=False)
    finish_date = Column(Date, nullable=False)
    cost = Column(Numeric(10, 2), nullable=False)
    comment = Column(Text)

    conditions = relationship('Rentcondition')
    renter = relationship('Account')


t_housingreview = Table(
    'housingreview', metadata,
    Column('contract_id', ForeignKey('rentcontract.id'), nullable=False, unique=True),
    Column('location_rate', Integer),
    Column('cleanliness_rate', Integer),
    Column('friendliness_rate', Integer),
    Column('review', Text),
    CheckConstraint('(cleanliness_rate >= 1) AND (cleanliness_rate <= 5)'),
    CheckConstraint('(friendliness_rate >= 1) AND (friendliness_rate <= 5)'),
    CheckConstraint('(location_rate >= 1) AND (location_rate <= 5)')
)


t_renterreview = Table(
    'renterreview', metadata,
    Column('contract_id', ForeignKey('rentcontract.id'), nullable=False, unique=True),
    Column('rate', Integer),
    Column('review', Text),
    CheckConstraint('(rate >= 1) AND (rate <= 5)')
)


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
    def hello(self):
        return "Hello DB"

    @cherrypy.expose
    def rating(self, apartment_id=None):
        with pg_driver.connect(user=args.pg_user, password=args.pg_password, host=args.pg_host,
                               port=args.pg_port, database=args.pg_database) as db:
            cur = db.cursor()
            if apartment_id is None:
                cur.execute("""SELECT location_rate, cleanliness_rate, friendliness_rate, average_cost FROM
(SELECT H.country_id as country,
	H.title as title,
	AVG(HR.location_rate) AS location_rate,
	AVG(HR.cleanliness_rate) AS cleanliness_rate,
	AVG(HR.friendliness_rate) AS friendliness_rate,
	AVG(PPW.cost) AS average_cost
FROM HousingReview HR JOIN RentContract RCT ON RCT.id = HR.contract_id
RIGHT JOIN RentConditions RCS ON RCS.id = RCT.conditions_id
JOIN PricePerWeek PPW ON RCS.id = PPW.conditions_id
JOIN Housing H ON H.id = RCS.housing_id
GROUP BY H.id) AS D
ORDER BY country ASC,
	location_rate+cleanliness_rate+friendliness_rate DESC,
	title ASC;""")
            else:
                cur.execute("""SELECT location_rate, cleanliness_rate, friendliness_rate FROM
(SELECT H.country_id as country,
	H.title as title,
	AVG(HR.location_rate) AS location_rate,
	AVG(HR.cleanliness_rate) AS cleanliness_rate,
	AVG(HR.friendliness_rate) AS friendliness_rate
FROM HousingReview HR JOIN RentContract RCT ON RCT.id = HR.contract_id
RIGHT JOIN RentConditions RCS ON RCS.id = RCT.conditions_id
JOIN Housing H ON H.id = RCS.housing_id WHERE H.id = %s
GROUP BY H.id) AS D
ORDER BY country ASC,
	location_rate+cleanliness_rate+friendliness_rate DESC,
	title ASC;""", apartment_id)

            result = []
            planets = cur.fetchall()
            for p in planets:
                result.append(p)
            cherrypy.response.headers['Content-Type'] = 'text/plain; charset=utf-8'
            return "\n".join([str(p) for p in planets])


if __name__ == '__main__':
    cherrypy.quickstart(App())
