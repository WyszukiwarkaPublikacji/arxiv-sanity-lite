"""
This script sends emails (one mail per user) with notifications generated for users in generate_notification.py. 
Content of notification is based on saved publications
in their notification-enabled folders.

Configuration Requirements:
SendGrid API Key:
Create a file named 'sendgrid_api_key.txt' in the project root
Place your own SendGrid API key there

Sender Email:
set from_email in send_email to your own sender email configured in sendgrid
"""

import json
import os
from datetime import datetime

import sendgrid
from flask import Flask, render_template_string
from sendgrid.helpers.mail import Content, Email, Mail, To
from sqlalchemy import and_, select, update

from db.SQLLite.OrmDB import Notifications, Users, engine

# Sender email
SENDER_EMAIL = "wyszukiwarka@gmail.com"
# Flask for rendering html templates
app = Flask(__name__)

def generate_mail_for_user(user_id: int):
    with engine.connect() as connection:
        # Query notifications using Core Table
        query = select(Notifications).where(
            and_(
                Notifications.c.user_id == user_id,
                Notifications.c.status == 'ready',
                Notifications.c.deleted == False
            )
        )
        
        result = connection.execute(query)
        notifications_of_user = result.fetchall()
        
        if not notifications_of_user:
            print(f"no notifications for user with id: {user_id}")
            return None, []

        folders_with_publications = {}  

        for notification in notifications_of_user:
            publication_ids = json.loads(notification.publication_ids) if isinstance(notification.publication_ids, str) else notification.publication_ids
            
            if notification.folder_id not in folders_with_publications:
                folders_with_publications[notification.folder_id] = []
            folders_with_publications[notification.folder_id].append(publication_ids)

        folder_with_publications_ids = []
        for folder_id in folders_with_publications.keys():
            publication_ids = folders_with_publications[folder_id]
            folder_with_publications_ids.append({"folder_id": folder_id, "publication_ids": publication_ids})  
        
        template = """
        <!DOCTYPE HTML>
        <html>
        <head>
        </head>
        <body>
        <h1>Powiadomienia użytkownika z id {{user_id}}</h1>
        {% for folder in folders %}
        <div class="folder">
            <div class="folder-title">Folder z id {{ folder.folder_id }}</div>
            <ul class="publication-list">
            {% for publication_id in folder.publication_ids %}
            <li>Publikacja ID: {{ publication_id }}</li>
            {% endfor %}
            </ul>
        </div>
        {% endfor %}
        </body>
        </html>
        """
        with app.app_context():
            return render_template_string(template, user_id=user_id, folders=folder_with_publications_ids), notifications_of_user

def send_email(to, html, date): 
    if not to or not html:
        print("no email or html")
        return None
    
    print("sending email...")

    if not os.path.isfile('sendgrid_api_key.txt'):
        print("brak pliku sendgrid_api_key.txt")
        return None
    api_key = open('sendgrid_api_key.txt', 'r').read().strip()

    sg = sendgrid.SendGridAPIClient(api_key=api_key) 
    from_email = Email(f"{SENDER_EMAIL}")
    to_email = To(to)
    subject = date + " notification"
    content = Content("text/html", html)
    mail = Mail(from_email, to_email, subject, content)

    try:
        response = sg.client.mail.send.post(request_body=mail.get())
        print(f"Status code: {response.status_code}")
        return response.status_code
    except Exception as e:
        print(f"blad przy wysylaniu: {e}")
        return None

def set_to_sent(sent_notifications_list, time):
    if not sent_notifications_list:
        print("no notifications to be updated")
        return 0
    
    count_folders = 0
    with engine.connect() as connection:
        trans = connection.begin()
        for sent_notification in sent_notifications_list:
            try: 
                update_stmt = update(Notifications).where(
                    Notifications.c.id == sent_notification.id
                ).values(
                    status='sent',
                    sent_at=time
                )
                
                result = connection.execute(update_stmt)
                if result.rowcount > 0:
                    count_folders += 1
                else:
                    print(f"Warning: Notification with id {sent_notification.id} not found in database")
            except Exception as e:
                print(f"Error while updating notification {sent_notification.id}: {e}") 
        
        try:
            trans.commit()
            print(f"modified db, updated notification_status of {count_folders} to sent")
            return count_folders
        except Exception as e:
            trans.rollback()
            print(f"Error committing changes to database: {e}")
            return 0
        
if __name__ == "__main__":
    tnow = datetime.now()
    tnow_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with engine.connect() as connection:
        try: 
            query = select(Users).join(
                Notifications,
                Users.c.id == Notifications.c.user_id
            ).where(
                and_(
                    Notifications.c.status == 'ready',
                    Notifications.c.deleted == False
                )
            ).distinct()
            
            result = connection.execute(query)
            users_with_notifs = result.fetchall()
            
            print(f"Znaleziono {len(users_with_notifs)} użytkowników z powiadomieniami")
        except Exception as e:
            print(f"Błąd podczas pobierania użytkowników: {e}")
            users_with_notifs = []
        
    for user in users_with_notifs:
        try:
            user_html, user_rows = generate_mail_for_user(user.id)

            print(f"notification for a user with id {user.id} is ready to go")
            
            email = f"{user.id}"
            result = send_email(email, user_html, tnow_str+f" {user.email}")
            if result == 202:
                print("email sent")
                set_to_sent(user_rows, tnow)
        except Exception as e:
            print(f"error while working with notification for user with id:{user.id}, error: {e}")
            continue