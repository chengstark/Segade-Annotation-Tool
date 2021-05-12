from django.test import TestCase
import os
import sqlite3
from os import path
# Create your tests here.
username = 1
db_path = path.join(os.getcwd(), 'multiuser/multiuser_outputs/{}/annotation_results.db'.format(username)).replace('\\',
                                                                                                                  '/')
print(db_path)
a = sqlite3.connect(db_path)
a.close()
anno_db = sqlite3.connect(
            path.join(os.getcwd(), 'multiuser/multiuser_outputs/{}/annotation_results.db'.format(username)))
anno_db.close()
