DROP TYPE IF EXISTS SEX CASCADE;
DROP TABLE IF EXISTS Account CASCADE;
DROP TABLE IF EXISTS Country CASCADE;
DROP TABLE IF EXISTS Housing CASCADE;
DROP TABLE IF EXISTS HousingFeatures CASCADE;
DROP TABLE IF EXISTS RentConditions CASCADE;
DROP TABLE IF EXISTS PricePerWeek CASCADE;
DROP TABLE IF EXISTS RentContract CASCADE;
DROP TABLE IF EXISTS HousingReview CASCADE;
DROP TABLE IF EXISTS RenterReview CASCADE;
DROP TABLE IF EXISTS Genres CASCADE;
DROP TABLE IF EXISTS Event CASCADE;


CREATE TYPE SEX AS ENUM ('Male', 'Female');

CREATE TABLE Account (
id SERIAL PRIMARY KEY,
first_name TEXT NOT NULL,
second_name TEXT NOT NULL,
email TEXT UNIQUE NOT NULL,
telephone_number TEXT UNIQUE NOT NULL,
sex SEX,
birthday DATE,
photo_path TEXT);

CREATE TABLE Country (
id SERIAL PRIMARY KEY,
country TEXT UNIQUE NOT NULL);




CREATE TABLE Housing (
id SERIAL PRIMARY KEY,
owner_id INT REFERENCES Account NOT NULL,
country_id INT REFERENCES Country NOT NULL,
city TEXT NOT NULL,
adress TEXT NOT NULL,
lat NUMERIC (7, 5) NOT NULL,
lon NUMERIC (8, 5) NOT NULL,
title TEXT NOT NULL,
chambers_number INT NOT NULL CHECK(chambers_number >= 0),
beds_number INT NOT NULL CHECK(beds_number >= 0),
habitant_capacity INT NOT NULL CHECK(habitant_capacity >= 0));

CREATE TABLE HousingFeatures (
housing_id INT REFERENCES Housing UNIQUE NOT NULL,
wifi BOOLEAN DEFAULT false,
iron BOOLEAN DEFAULT false,
kitchen_items BOOLEAN DEFAULT false,
electric_kettle BOOLEAN DEFAULT false,
smoking_permission BOOLEAN DEFAULT false,
bath BOOLEAN DEFAULT false,
pool BOOLEAN DEFAULT false,
jacuzzi BOOLEAN DEFAULT false,
balcony BOOLEAN DEFAULT false,
conditioner BOOLEAN DEFAULT false,
metro_walking_access BOOLEAN DEFAULT false,
parking BOOLEAN DEFAULT false);

CREATE TABLE RentConditions (
id SERIAL PRIMARY KEY,
housing_id INT REFERENCES Housing UNIQUE NOT NULL,
cleaning_price NUMERIC(10,2) CHECK(cleaning_price >= 0));

CREATE TABLE PricePerWeek (
conditions_id INT REFERENCES RentConditions NOT NULL,
week_number INT NOT NULL CHECK(week_number >=0 AND week_number <= 53),
cost NUMERIC(10,2) NOT NULL CHECK(cost >=0),
UNIQUE(conditions_id, week_number));

CREATE TABLE RentContract (
id SERIAL PRIMARY KEY,
conditions_id INT REFERENCES RentConditions NOT NULL,
renter_id INT REFERENCES Account NOT NULL,
person_number INT NOT NULL CHECK(person_number > 0),
start_date DATE NOT NULL,
finish_date DATE NOT NULL,
cost NUMERIC(10,2) CHECK(cost >= 0) NOT NULL,
UNIQUE(conditions_id, start_date),
comment TEXT);

CREATE TABLE HousingReview (
contract_id INT REFERENCES RentContract UNIQUE NOT NULL,
location_rate INT CHECK(location_rate >=1 AND location_rate <=5),
cleanliness_rate INT CHECK(cleanliness_rate >=1 AND cleanliness_rate <=5),
friendliness_rate INT CHECK(friendliness_rate >=1 AND friendliness_rate <=5),
review TEXT);

CREATE TABLE RenterReview (
contract_id INT REFERENCES RentContract UNIQUE NOT NULL,
rate INT CHECK(rate >=1 AND rate <=5),
review TEXT);

CREATE TABLE Genres (
type TEXT PRIMARY KEY);

CREATE TABLE Event (
id SERIAL PRIMARY KEY,
title TEXT NOT NULL,
genre TEXT REFERENCES Genres(type),
country_id INT REFERENCES Country NOT NULL,
city TEXT NOT NULL,
lat NUMERIC (7, 5) NOT NULL,
lon NUMERIC (8, 5) NOT NULL,
start_date DATE NOT NULL,
finish_date DATE NOT NULL);



 
INSERT INTO Account (first_name,second_name,email,telephone_number,sex,birthday) VALUES 
('Jena','Gardner','Vivamus@dictum.co.uk','+7 (426) 783-81-20','Male','1987-08-23'),('Charity','Evans','Maecenas.malesuada.fringilla@elitpharetra.net','+7 (952) 158-85-21','Female','1962-12-04'),
('Sheila','Meyers','enim.Curabitur@tortorNunc.net','+7 (945) 907-93-98','Female','2004-02-03'),
('Lynn','Silva','eu@nibh.co.uk','+7 (975) 440-53-41','Female','2011-06-13'),
('Adrian','Rowland','Cras.pellentesque.Sed@quisaccumsanconvallis.ca','+7 (966) 846-71-84','Male','1968-05-29'),
('Wang','Daniels','turpis@quis.co.uk','+7 (401) 466-27-79','Female','1991-08-07'),
('Leilani','Cochran','mus@Integeridmagna.org','+7 (987) 533-53-52','Male','1984-06-02'),
('Ariel','Velez','semper.Nam@lacus.ca','+7 (851) 302-89-10','Female','1973-07-27'),
('Jason','Sherman','elementum.purus.accumsan@facilisisSuspendisse.net','+7 (341) 598-91-60','Female','2018-11-23'),
('Britanney','Powell','Proin@pede.com','+7 (485) 417-75-92','Female','1965-07-16'),
('Luke','Whitley','et.magnis.dis@etnetuset.ca','+7 (955) 726-75-63','Female','2011-10-16'),
('Yvonne','Cash','tristique.aliquet@Curabitur.co.uk','+7 (491) 542-05-16','Female','1969-11-01'),
('Kay','Brewer','senectus@Curabiturvellectus.net','+7 (341) 945-12-34','Female','1969-01-11'),
('Clayton','Adams','pellentesque.Sed.dictum@Crasvulputatevelit.org','+7 (385) 119-94-15','Male','2017-12-23'),
('Zenaida','Mcpherson','magnis.dis@purusin.edu','+7 (931) 678-05-65','Female','1995-05-30'),
('Erasmus','Cherry','et.malesuada@ullamcorper.org','+7 (473) 171-81-61','Female','1991-11-08'),
('Eagan','Walker','dictum.ultricies@auctorquis.net','+7 (427) 114-64-12','Female','1978-03-25'),
('Dylan','Keith','bibendum.ullamcorper.Duis@antebibendum.org','+7 (813) 635-97-04','Female','1995-06-16'),
('Ignacia','Gillespie','non.lorem.vitae@Fuscefermentumfermentum.edu','+7 (960) 702-93-00','Female','1981-08-08'),
('Brynne','Howard','arcu.eu.odio@egestas.org','+7 (906) 502-89-41','Male','1988-12-07'),
('Akeem','Small','Vivamus.molestie@diameu.edu','+7 (499) 910-60-64','Male','1986-10-29'),
('Rina','Rowland','in.cursus@blandit.org','+7 (877) 632-20-71','Female','1994-03-10'),
('Yvonne','Shields','et.malesuada@afacilisisnon.com','+7 (835) 578-81-96','Female','2010-12-24'),
('Aladdin','Mckee','fringilla.mi.lacinia@IntegerurnaVivamus.ca','+7 (974) 972-60-73','Male','1961-07-24'),
('Shelby','Kemp','sapien@Proin.co.uk','+7 (982) 946-01-72','Female','2013-03-07'),
('Meghan','Reynolds','sem@hendrerit.org','+7 (942) 687-15-99','Male','1992-02-02'),
('Lynn','Park','Mauris.molestie@eleifendnuncrisus.com','+7 (938) 946-58-16','Male','2004-04-18'),
('Jaime','Porter','urna.Vivamus@iaculisquis.org','+7 (965) 423-06-59','Male','1967-11-01'),
('Oren','Gill','Donec.egestas@duiquisaccumsan.com','+7 (933) 305-72-31','Female','2017-09-25'),
('Mercedes','Higgins','Etiam.bibendum.fermentum@magnaUt.net','+7 (473) 102-62-42','Female','2013-05-14'),
('Camden','Molina','mauris.blandit@ullamcorper.com','+7 (910) 612-12-83','Male','2014-09-20'),
('Lane','Rush','euismod@Curabitur.edu','+7 (912) 644-15-38','Female','1985-05-18'),
('Jerry','Munoz','velit.Aliquam.nisl@lorem.edu','+7 (818) 754-77-15','Male','2016-12-04'),
('Zelda','Joseph','vel@telluseu.com','+7 (900) 137-88-45','Male','1992-10-21'),
('Ora','Blackwell','et.magnis.dis@necleoMorbi.edu','+7 (472) 608-92-19','Female','2001-03-04'),
('Warren','King','dolor.vitae.dolor@elit.com','+7 (345) 517-62-34','Female','1970-04-19'),
('Kerry','Henson','id.ante.dictum@enimCurabiturmassa.co.uk','+7 (980) 466-15-19','Male','1986-12-05'),
('Chadwick','Byers','cursus.et.eros@Duisgravida.co.uk','+7 (923) 658-94-81','Female','1971-11-25'),
('Rae','Garrett','eros.Proin.ultrices@ornarefacilisiseget.edu','+7 (424) 878-91-58','Female','1989-12-18'),
('Ifeoma','Mckinney','convallis@nisiAeneaneget.org','+7 (485) 333-62-60','Male','1976-12-14'),
('Davis','Britt','lorem.vehicula.et@sit.com','+7 (851) 527-01-67','Male','1995-12-16'),
('Wynter','Lowery','cursus.et@ipsum.edu','+7 (902) 796-08-72','Male','1974-07-20'),
('TaShya','Levy','venenatis.lacus@imperdietdictum.net','+7 (912) 104-03-94','Female','1988-08-18'),
('Tate','Perez','sed.consequat.auctor@Integer.com','+7 (936) 660-31-36','Female','2014-12-11'),
('Davis','Mooney','In.mi@ante.net','+7 (861) 912-62-08','Female','1970-05-13'),
('Kyla','Mcknight','dolor@sapien.org','+7 (861) 807-67-18','Male','2001-05-15'),
('Gavin','Waller','faucibus@Nunclectuspede.com','+7 (848) 952-08-13','Male','2003-04-19'),
('Quentin','Barber','eget.nisi.dictum@Duisami.net','+7 (496) 571-67-63','Male','1969-07-03'),
('Eagan','Mcmillan','Nunc.sed.orci@nonsapien.net','+7 (959) 170-55-44','Male','1993-11-13'),
('Ross','Wise','quis.diam.luctus@turpis.com','+7 (862) 537-28-02','Male','1985-04-01'),
('Maryam','Kelly','eget@cursusaenim.net','+7 (494) 729-00-19','Female','1966-07-05'),
('Octavia','Rasmussen','augue.porttitor.interdum@sagittisfelis.ca','+7 (998) 474-24-66','Female','1987-06-22'),
('Clayton','Hoffman','Duis.mi@etnuncQuisque.com','+7 (992) 427-99-60','Female','1977-11-20'),
('Elizabeth','Bray','sapien.molestie.orci@vitaeorci.edu','+7 (866) 819-76-32','Male','2005-03-13'),
('Craig','Garcia','mus.Proin.vel@sempercursus.co.uk','+7 (472) 947-14-36','Male','1981-03-13'),
('Luke','Montoya','urna.Nunc.quis@pellentesque.org','+7 (984) 995-65-53','Female','1987-01-15'),
('Noble','Mcpherson','ac.risus.Morbi@ligulaconsectetuer.co.uk','+7 (487) 986-34-09','Female','1986-10-21'),
('Jolene','Oneil','orci.Ut.sagittis@liberoet.net','+7 (424) 687-15-32','Female','2012-01-04'),
('Cooper','Carey','cursus.purus.Nullam@mattis.net','+7 (815) 496-28-62','Male','2014-10-25'),
('Kirby','Fuentes','Cras.vehicula@Nullaaliquet.co.uk','+7 (343) 284-37-87','Male','1981-05-03'),
('Xanthus','Ferguson','ut.odio.vel@eratnequenon.net','+7 (871) 907-83-45','Male','1999-05-11'),
('Halla','Hancock','sollicitudin.orci.sem@non.ca','+7 (961) 715-76-04','Female','2013-09-15'),
('Jermaine','Farmer','Cras.lorem@elit.co.uk','+7 (388) 490-23-50','Female','1992-04-20'),
('Tyrone','Stanton','nec@sapiencursusin.co.uk','+7 (926) 682-71-83','Female','1994-04-30'),
('Brittany','Trevino','mauris.a.nunc@vitaeeratvel.net','+7 (909) 346-09-48','Male','1997-06-01'),
('Flavia','Davidson','ullamcorper.eu@rutrumlorem.net','+7 (991) 833-34-29','Female','1976-01-14'),
('Liberty','Padilla','sit@Aenean.com','+7 (426) 977-72-46','Male','1976-11-06'),
('Deirdre','Holland','elit.pharetra.ut@Phasellus.edu','+7 (813) 961-94-67','Male','1974-01-03'),
('Irma','George','feugiat.Sed.nec@dictumProin.co.uk','+7 (499) 485-56-13','Female','2017-08-17'),
('Keith','Burgess','eu@Phasellus.com','+7 (347) 247-04-76','Male','1975-02-26'),
('Anastasia','Jackson','dictum@ut.co.uk','+7 (924) 335-58-02','Female','1994-01-10'),
('Hyacinth','Bernard','ac.feugiat.non@etpedeNunc.net','+7 (481) 375-48-55','Male','1997-01-14'),
('Aileen','Gregory','Class@infelisNulla.org','+7 (919) 663-75-77','Female','1971-05-11'),
('Amanda','Peterson','molestie.dapibus@tinciduntadipiscingMauris.co.uk','+7 (341) 298-98-04','Male','2003-02-19'),
('Lars','Anderson','tellus@risusMorbimetus.ca','+7 (814) 113-46-41','Female','1963-09-18'),
('Leslie','Clay','eget.mollis@duiFusce.co.uk','+7 (484) 894-81-25','Male','1997-10-17'),
('Noah','Holloway','Donec.dignissim.magna@turpisvitae.org','+7 (302) 809-08-86','Female','1976-08-11'),
('Aphrodite','Simpson','Nunc.laoreet.lectus@Maecenas.edu','+7 (395) 442-29-06','Male','1988-12-31'),
('Nigel','Little','lacus@Praesentinterdumligula.net','+7 (989) 108-34-91','Male','2010-09-23'),
('Magee','Sanchez','tellus.non.magna@duilectus.co.uk','+7 (421) 631-43-27','Female','1965-07-10'),
('Aidan','Mays','arcu.Sed@nonquam.edu','+7 (984) 265-22-02','Male','1991-10-13'),
('Sara','Chandler','elementum@dignissim.ca','+7 (934) 678-92-20','Female','2000-03-01'),
('Jorden','Downs','Curabitur.vel.lectus@quam.net','+7 (843) 520-70-99','Female','2006-04-28'),
('Taylor','Ware','Nullam.feugiat@Inornaresagittis.ca','+7 (944) 127-00-14','Male','2006-11-12'),
('Amy','Santana','felis@imperdietdictum.edu','+7 (485) 944-34-72','Female','1982-01-15'),
('Belle','Justice','luctus.et.ultrices@etrisusQuisque.com','+7 (940) 854-45-36','Male','1973-10-05'),
('Helen','Mcknight','orci@fames.com','+7 (848) 500-44-42','Male','1969-06-12'),
('Cassandra','Richards','nec.quam@elit.ca','+7 (495) 103-10-88','Male','1980-10-31'),
('Montana','Everett','cursus.purus.Nullam@nonmassa.org','+7 (475) 299-44-28','Female','2001-04-30'),
('Leilani','Mercer','nonummy.ac.feugiat@ante.com','+7 (979) 784-72-94','Female','1967-09-15'),
('Ezekiel','Myers','ridiculus.mus.Proin@lacusQuisqueimperdiet.ca','+7 (871) 170-21-19','Male','1972-12-02'),
('Tad','Burnett','sed.leo@tristique.org','+7 (942) 136-03-51','Female','2004-08-16'),
('Cameran','Rowland','quis.accumsan.convallis@Integerin.net','+7 (924) 609-16-99','Male','2015-09-16'),
('Kristen','Compton','dui.Fusce.diam@est.ca','+7 (923) 762-75-34','Male','2006-10-15'),
('Basia','Coleman','auctor.odio@dignissimMaecenas.co.uk','+7 (913) 988-24-07','Female','1969-08-03'),
('Quentin','Chang','ac.mattis@Naminterdum.co.uk','+7 (920) 483-30-02','Female','1989-01-04'),
('Nicole','Roy','lorem.lorem.luctus@ascelerisquesed.edu','+7 (383) 149-98-45','Male','2013-10-18'),
('Quon','Johns','Phasellus@tristiquepellentesque.ca','+7 (991) 235-87-87','Male','1973-02-08'),
('Fletcher','Whitley','accumsan.interdum.libero@ipsum.ca','+7 (872) 597-41-14','Male','1987-07-08'),
('Maryam','Dalton','eget.ipsum@aptent.net','+7 (855) 171-71-89','Male','1990-03-26');

INSERT INTO Country (country) VALUES ('Uganda'),('Ghana'),('South Georgia and The South Sandwich Islands'),('Kazakhstan'),('Moldova'),('Dominica'),('Romania'),('Greece'),('Central African Republic'),('American Samoa');
INSERT INTO Country (country) VALUES ('Guernsey'),('Algeria'),('Malaysia'),('Timor-Leste'),('Portugal'),('Mauritania'),('Greenland'),('Bosnia and Herzegovina'),('Andorra'),('Netherlands');

INSERT INTO Housing (owner_id,country_id,city,adress,lat,lon,title,chambers_number,beds_number,habitant_capacity) VALUES (73,8,'Maracanaú','1712 Malesuada Street','77.27223','141.7302','ligula',7,1,4),
(95,9,'Paal','2561 Sed St.','78.98473','98.98745','Nulla',8,8,8),
(37,17,'Montgomery','Ap #249-9120 Pellentesque, Av.','40.913','-14.58674','Mauris',4,5,9),
(97,17,'Södertälje','Ap #457-3204 Erat Av.','66.14728','138.54874','neque',5,3,8),
(91,15,'Cowdenbeath','303-2123 Erat Av.','26.10356','-86.07419','ante.',4,6,3),
(49,19,'San Massimo','Ap #531-2453 Ac Rd.','-74.03042','-168.49106','sapien',1,6,6),
(90,15,'Jakarta','Ap #877-9913 Pellentesque Road','10.75939','178.69461','orci,',8,8,4),
(47,18,'Sevsk','6950 Ullamcorper St.','-43.92663','46.21289','nonummy',5,6,10),
(52,6,'Bismil','175-5547 Dis Rd.','-8.05609','23.8668','vestibulum,',7,8,10),
(43,11,'Kingussie','1049 Ornare Ave','-8.15423','-118.29056','dis',7,9,10),
(82,16,'Cajazeiras','256 Ridiculus Street','-75.89181','153.6499','quis',5,2,4),
(40,5,'Guadalupe','697-9462 Tellus Av.','12.09065','-0.70486','egestas',1,9,9),
(49,7,'Reading','Ap #474-7353 Curabitur Rd.','20.81573','132.78008','litora',2,8,8),
(13,16,'Lillois-WitterzŽe','P.O. Box 667, 8936 Fames Rd.','27.7802','-118.69293','ligula.',2,1,5),
(72,2,'Alexandria','328-9555 Ligula. St.','63.39055','50.68458','pede',3,2,1),
(42,20,'Chepstow','P.O. Box 664, 1832 Tincidunt Road','-32.43898','-172.37414','lorem',9,1,7),
(24,1,'Ozherelye','856-9709 Magna St.','35.07998','-179.57353','ac,',9,1,7),
(9,15,'Machilipatnam','Ap #346-3088 Mauris. Rd.','-54.29608','-64.41243','facilisis',3,8,9),
(31,15,'Jasper','P.O. Box 449, 1225 Facilisis Av.','-9.39981','66.16364','aliquet',4,10,5),
(12,11,'Rostov','349-1376 Sit Av.','-61.6128','164.08813','sociis',10,5,8),
(15,14,'Hilo','P.O. Box 543, 9532 Vulputate Rd.','17.33798','121.29646','auctor,',9,4,3),
(64,1,'Montague','872-1813 Posuere, Rd.','-63.3938','129.4733','sem.',7,4,2),
(17,19,'Birori','P.O. Box 638, 8420 Mi. St.','-29.4218','33.39373','euismod',10,8,2),
(4,8,'Saint-Dié-des-Vosges','8416 Velit. Rd.','-14.04925','-10.98948','lorem',6,10,10),
(29,13,'Ipswich','704-8824 Proin Av.','19.91105','177.99343','neque',6,4,9),
(83,1,'Wyoming','1126 Cursus St.','-39.65492','-133.34767','lacus.',9,5,1),
(26,12,'Lamorteau','P.O. Box 858, 8387 Urna, St.','54.86105','-160.31056','mattis.',5,4,2),
(54,2,'Traiskirchen','990-8239 Nulla St.','-13.06041','111.02867','arcu.',3,9,3),
(36,5,'Anthisnes','P.O. Box 169, 2017 Metus. Rd.','28.48004','77.18992','diam',7,10,9),
(51,16,'Portigliola','Ap #501-9378 Diam Rd.','52.97976','155.11533','elit',5,5,8),
(19,17,'La Plata','P.O. Box 908, 3442 Vestibulum Street','-18.15477','-3.0089','nisi',5,7,4),
(35,4,'Tumbler Ridge','197-4258 Iaculis St.','-50.23089','-29.99255','posuere',3,10,2),
(29,7,'Padang Panjang','Ap #808-8139 Non, Street','-42.70469','-96.85789','tellus.',10,5,3),
(69,2,'Lang','357-1151 Mauris. Rd.','36.50091','-133.93871','elit.',5,2,6),
(3,12,'Toernich','P.O. Box 924, 4047 Cras Ave','-41.88324','11.80623','vitae',6,8,10),
(17,12,'Whyalla','Ap #990-5572 Turpis Rd.','54.84402','67.75123','Aenean',4,9,7),
(5,8,'Ohain','452 Nulla St.','-74.72416','-57.5617','amet',6,7,6),
(2,10,'Samara','P.O. Box 940, 5639 Mauris Rd.','-17.58381','135.66394','velit',10,9,1),
(74,8,'Gembloux','9773 Vel Ave','29.83269','-76.61136','Donec',6,5,7),
(47,9,'Blankenfelde-Mahlow','P.O. Box 637, 9933 Fusce Ave','-1.80632','-17.84803','enim.',1,5,3),
(77,14,'Alsemberg','P.O. Box 928, 3083 Lorem, Rd.','-62.35769','-69.84639','Nunc',8,2,5),
(56,9,'Giugliano in Campania','5594 Vivamus Av.','-53.85066','121.89286','euismod',2,5,9),
(61,8,'Valleyview','893-371 Blandit Ave','26.75229','54.39044','sem',6,1,1),
(58,18,'Grasse','P.O. Box 443, 5263 Interdum Av.','31.28999','7.26981','sed',9,8,8),
(74,3,'San Vicente','452-1735 A St.','79.12778','-73.3496','Praesent',2,7,7),
(94,13,'Kotlas','127-605 Risus. Av.','-16.72071','74.43009','risus.',5,5,9),
(51,12,'Requínoa','271-5533 Sit Rd.','-27.93311','159.42824','lectus.',1,8,7),
(75,6,'Oswestry','8214 Augue St.','-13.17196','107.11729','ipsum',5,10,6),
(47,5,'Langley','P.O. Box 255, 863 Auctor Road','14.60677','-102.74224','Proin',3,7,6),
(63,2,'Pskov','P.O. Box 850, 1440 Egestas. Ave','-45.4149','-145.94626','enim',6,3,5);

INSERT INTO HousingFeatures (housing_id,wifi,iron,kitchen_items,electric_kettle,smoking_permission,bath,pool,jacuzzi,balcony,conditioner,metro_walking_access,parking) VALUES (1,'true','true','true','true','true','false','true','false','true','false','false','true'),
(2,'true','true','false','true','true','false','false','true','true','true','true','true'),
(3,'true','false','false','false','true','true','false','false','false','false','true','true'),
(4,'false','true','false','true','false','true','true','false','true','false','false','true'),
(5,'true','true','false','false','true','true','false','false','true','true','true','true'),
(6,'true','true','true','false','true','false','false','false','true','false','true','true'),
(7,'true','true','true','false','false','false','false','true','false','false','false','true'),
(8,'false','false','true','false','true','false','true','false','true','false','true','true'),
(9,'false','true','false','false','true','true','true','true','false','true','true','false'),
(10,'false','true','true','false','true','false','true','false','true','true','false','false'),
(11,'false','false','true','false','false','true','true','false','true','true','true','false'),
(12,'true','true','false','true','true','false','true','true','true','true','false','false'),
(13,'false','false','true','false','false','true','true','true','false','true','false','false'),
(14,'true','true','true','true','false','true','true','false','false','false','true','false'),
(15,'false','false','false','false','false','true','false','false','false','false','false','true'),
(16,'false','true','true','true','false','false','true','false','false','false','false','true'),
(17,'true','true','true','true','false','true','true','true','true','false','false','true'),
(18,'true','true','false','true','false','true','true','true','true','true','false','false'),
(19,'true','false','true','true','true','true','false','true','false','true','false','false'),
(20,'false','true','true','true','true','true','true','true','true','false','false','true'),
(21,'false','false','false','false','true','false','false','true','true','true','true','false'),
(22,'false','false','false','true','false','true','true','true','false','false','true','false'),
(23,'false','false','false','true','false','true','false','false','true','false','false','false'),
(24,'false','false','true','true','true','true','true','true','false','false','false','false'),
(25,'false','false','false','true','false','true','false','false','false','false','false','false'),
(26,'false','false','false','false','true','true','false','false','true','true','false','false'),
(27,'false','true','false','true','false','false','true','false','true','true','true','false'),
(28,'false','false','false','true','false','true','false','true','false','false','true','false'),
(29,'false','false','true','true','true','true','false','false','false','false','true','false'),
(30,'true','true','true','false','true','true','false','false','false','false','true','false'),
(31,'false','false','true','true','false','false','false','true','true','true','true','true'),
(32,'false','false','false','false','true','false','false','false','false','true','true','true'),
(33,'false','false','true','false','false','true','true','false','false','true','true','false'),
(34,'true','true','false','true','false','true','false','true','true','true','false','false'),
(35,'true','true','false','false','true','true','true','true','true','false','false','false'),
(36,'true','false','true','true','true','false','true','true','false','false','true','false'),
(37,'false','true','true','false','true','false','false','false','false','false','true','true'),
(38,'true','false','false','false','false','false','true','false','true','false','true','true'),
(39,'true','true','false','false','false','false','true','false','false','false','false','false'),
(40,'true','true','false','true','false','true','true','false','true','false','true','true'),
(41,'true','false','false','false','false','false','true','true','false','false','false','true'),
(42,'false','true','false','true','true','true','true','false','true','false','false','true'),
(43,'false','false','true','true','true','true','false','false','true','true','true','true'),
(44,'true','true','false','true','true','false','false','false','true','true','true','true'),
(45,'false','false','false','false','true','false','false','false','true','true','true','true'),
(46,'true','true','false','true','false','false','true','false','false','false','false','true'),
(47,'true','true','false','false','false','false','false','true','true','false','false','false'),
(48,'true','false','true','true','true','false','false','false','false','true','true','true'),
(49,'false','false','true','true','false','false','false','true','false','true','false','true'),
(50,'false','true','true','true','false','true','true','true','true','false','false','true');


INSERT INTO RentConditions (housing_id,cleaning_price) VALUES (1,'6.33'),(2,'9.21'),(3,'10.03'),(4,'6.42'),(5,'7.43'),(6,'10.83'),(7,'5.12'),(8,'12.72'),(9,'8.57'),(10,'8.01'),(11,'10.26'),(12,'10.95'),(13,'12.29'),(14,'12.08'),(15,'14.28'),(16,'9.65'),(17,'12.18'),(18,'13.75'),(19,'10.01'),(20,'12.72'),(21,'10.52'),(22,'8.98'),(23,'8.48'),(24,'14.5'),(25,'7.34'),(26,'12.44'),(27,'8.91'),(28,'8.5'),(29,'9.68'),(30,'8.89'),(31,'6.92'),(32,'10.86'),(33,'10.5'),(34,'9.9'),(35,'8.97'),(36,'7.27'),(37,'8.55'),(38,'10.42'),(39,'11.59'),(40,'8.15'),(41,'11.71'),(42,'11.25'),(43,'12.29'),(44,'11.3'),(45,'8.17'),(46,'9.63'),(47,'9.16'),(48,'9.48'),(49,'10.33'),(50,'10.31');

INSERT INTO RentContract (conditions_id,renter_id,person_number,start_date,finish_date,cost) VALUES (14,45,5,'2020-11-05','2019-12-18','90.3'),(19,71,3,'2020-03-02','2019-07-16','97.65'),(2,41,4,'2020-01-11','2020-03-27','55.47'),(2,63,5,'2019-01-06','2020-01-02','105.12'),(4,66,5,'2020-04-11','2019-05-23','91.73'),(27,1,1,'2020-10-08','2019-10-06','99.15'),(30,94,1,'2020-08-15','2020-07-21','92.2'),(4,56,2,'2020-03-06','2019-08-07','76.33'),(39,73,2,'2018-12-31','2019-07-12','74.86'),(13,86,5,'2019-01-14','2019-10-03','110.62'),(11,60,2,'2019-12-23','2019-08-03','118.11'),(43,75,4,'2020-04-29','2019-11-22','90'),(42,69,2,'2019-02-12','2019-04-22','77.68'),(31,97,2,'2020-10-29','2020-09-03','112.11'),(43,79,2,'2020-07-30','2019-08-20','109.91'),(8,53,3,'2019-02-28','2019-09-21','111.43'),(6,75,5,'2019-06-17','2019-04-22','104.55'),(20,17,1,'2019-12-22','2019-01-26','83.36'),(19,85,5,'2019-09-23','2019-12-22','117.61'),(35,61,3,'2020-04-11','2019-11-03','90.69');

INSERT INTO PricePerWeek (conditions_id,week_number,cost) VALUES (1,5,'10.71'),(2,19,'7.57'),(3,11,'9.03'),(4,7,'7.63'),(5,22,'8.65'),(6,4,'9.05'),(7,33,'9.94'),(8,16,'7.7'),(9,31,'10.29'),(10,15,'10.75'),(11,28,'9.83'),(12,38,'11.24'),(13,23,'12.55'),(14,31,'10.25'),(15,34,'11.15'),(16,15,'10.61'),(17,40,'9.8'),(18,36,'10.69'),(19,4,'11.83'),(20,38,'10.86'),(21,9,'9.32'),(22,32,'11.2'),(23,25,'12.82'),(24,22,'10.18'),(25,29,'7.98'),(26,16,'5.99'),(27,38,'10.23'),(28,36,'11.65'),(29,22,'12.01'),(30,23,'9.87'),(31,9,'8.38'),(32,1,'8.23'),(33,4,'9.71'),(34,8,'12.02'),(35,29,'7.84'),(36,31,'7.03'),(37,26,'9.44'),(38,24,'10.5'),(39,6,'13.48'),(40,20,'11.13'),(41,3,'10.74'),(42,15,'9.3'),(43,24,'6.3'),(44,17,'9.15'),(45,40,'10.15'),(46,1,'12.15'),(47,11,'9.65'),(48,34,'10.02'),(49,36,'7.22'),(50,9,'13.47');

INSERT INTO HousingReview (contract_id,location_rate,cleanliness_rate,friendliness_rate,review) VALUES 
(1,1,4,5,'ac'),
(2,2,2,2,'dolor sit'),
(3,5,1,2,'neque. Nullam ut nisi a odio semper cursus.'),
(4,5,5,3,'sapien molestie orci tincidunt adipiscing. Mauris molestie pharetra nibh.'),
(5,2,2,5,'mus. Proin vel arcu eu odio tristique pharetra. Quisque'),
(6,4,4,4,'natoque penatibus et magnis dis parturient montes, nascetur ridiculus'),
(7,4,4,4,'montes, nascetur ridiculus mus. Donec'),
(8,2,4,5,'risus, at fringilla purus mauris'),
(9,3,2,2,'est, vitae sodales nisi magna sed'),
(10,4,3,2,'Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis'),
(11,3,4,3,'Donec'),
(12,3,5,4,'consequat nec, mollis vitae, posuere'),
(13,3,3,5,'eleifend vitae, erat. Vivamus nisi. Mauris nulla.'),
(14,1,4,2,'sed dictum eleifend,'),
(15,3,2,4,'non quam. Pellentesque habitant morbi tristique senectus et netus et malesuada fames');

INSERT INTO RenterReview (contract_id,rate,review) VALUES 
(1,5,'ut ipsum ac mi eleifend egestas. Sed pharetra, felis eget varius ultrices, mauris ipsum porta'),
(2,1,'egestas blandit. Nam nulla magna,'),
(3,2,'magna sed dui. Fusce aliquam,'),
(4,4,'eu, placerat eget, venenatis a, magna. Lorem ipsum'),
(5,4,'ligula. Nullam feugiat placerat velit. Quisque'),
(6,2,'bibendum sed, est. Nunc laoreet lectus quis massa. Mauris vestibulum, neque sed dictum eleifend,'),
(7,4,'ut, nulla. Cras eu'),
(8,2,'tempor, est ac mattis semper, dui lectus rutrum urna, nec luctus'),
(9,2,'sit amet risus. Donec egestas. Aliquam'),
(10,2,'eu neque pellentesque massa lobortis ultrices.'),
(11,3,'sodales elit erat vitae'),
(12,1,'tempor diam dictum sapien. Aenean massa. Integer vitae'),
(13,5,'posuere cubilia Curae; Phasellus ornare. Fusce mollis. Duis sit amet diam'),
(14,3,'vulputate mauris sagittis placerat. Cras dictum ultricies ligula. Nullam enim. Sed nulla ante, iaculis nec,'),
(15,1,'placerat. Cras dictum ultricies');
