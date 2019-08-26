# DJANGO TERMINAL COMMANDS

Creates a *project_name* directory in the current directory

        $ django-admin startproject project_name

Creates the *app* specified and creates the directory; if placed in the same directory as *manage.py*. 

        $ python manage.py startapp app_name

Starts the *manage.py* file which starts the server

        $ python manage.py runserver

Creates migrations

        $ python manage.py makemigrations

Sends migrations to the database

        $ python manage.py migrate

Create a super user for the admin panel

        $ python manage.py createsuperuser

Collects all static files into one folder

        $ python manage.py collectstatic