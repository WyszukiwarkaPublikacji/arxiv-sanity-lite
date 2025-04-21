import json
import os
from datetime import datetime

import sendgrid
from flask import Flask, render_template_string
from sendgrid.helpers.mail import Content, Email, Mail, To
from sqlalchemy import and_, func, select, update

from db.SQLLite.OrmDB import (Notifications, SavedFolders, SavedPublications,
                              Users, engine)


class NotificationGenerator:
    #def __init__(self):

        
    def get_publications(self, folder_id: int): 
        """
        Tymczasowa funkcja, docelowo powinna korzystac z algorytmu znajdywania podobnych publikacji w folderach 
        i zwracac liste najswiezszych sposrod wybranych przez algorytm publikacji

        obecnie funkcja wybiera pierwsze 4 publikacje w folderze
        """
        with engine.connect() as connection:
            query = select(SavedPublications.c.publication_id).where(
                SavedPublications.c.folder_id == folder_id
            ).limit(4)
            
            result = connection.execute(query)
            publication_ids = [row[0] for row in result]
        
        return publication_ids

    def define_notification_for_user(self, user_id: int): 
        with engine.connect() as connection:
            query = select(SavedFolders).where(
                and_(
                    SavedFolders.c.user_id == user_id,
                    SavedFolders.c.notifications_set == True
                )
            )
            
            result = connection.execute(query)
            folders_with_notifications_on = result.fetchall()
        
        notifications_data = {}
        
        for folder in folders_with_notifications_on:
            publications_in_notification = self.get_publications(folder.id)
            
            if publications_in_notification:
                notifications_data[folder.id] = publications_in_notification
        
        return notifications_data

    def get_next_notification_id(self, connection): 
        query = select(func.max(Notifications.c.id)).select_from(Notifications)
        result = connection.execute(query)
        max_id = result.scalar()
        
        if max_id is None:
            next_id = 1
        else:
            next_id = max_id + 1
        
        return next_id

    def generate_all_notifications(self):
        print("Starting notification generation...")
        try: 
            with engine.connect() as connection:
                trans = connection.begin()

                try:
                    query = select(Users)
                    result = connection.execute(query)
                    all_users = result.fetchall()
                    
                    total_users = len(all_users)
                    processed_users = 0
                    total_notifications = 0
                    
                    print(f"Found {total_users} users to process")
                    
                    for user in all_users:
                        try: 
                            user_notifications = self.define_notification_for_user(user.id)
                            
                            if not user_notifications:
                                print(f"No notifications for user {user.id}")
                                processed_users += 1
                                continue
                    
                            print(f"Saving notifications for user {user.id}")
                            notifications_count = 0
                            
                            for folder_id, publication_ids in user_notifications.items():
                                
                                if not publication_ids:
                                    continue
                                
                                tnow = datetime.now()
                                
                                # chamskie genereowanie nastepnego id bo autoincrement Notifications.id nie dziala na nowej wersji Orm i dopiero to ogarnalem, program dzialaby bez tej funkcji gdyby autoincrement dzialal
                                next_id = self.get_next_notification_id(connection)
                                
                                insert_stmt = Notifications.insert().values(
                                    id=next_id,
                                    user_id=user.id,
                                    folder_id=folder_id,
                                    publication_ids=json.dumps(publication_ids),
                                    status='ready',
                                    generated_at=tnow,
                                    deleted=False
                                )
                                
                                try:
                                    connection.execute(insert_stmt)
                                    notifications_count += 1
                                except Exception as e:
                                    print(f"Error inserting notification: {e}")
                            
                            if notifications_count > 0:
                                total_notifications += notifications_count
                                print(f"Saved {notifications_count} notifications for user with id:{user.id}")
                            
                            processed_users += 1
                            
                        except Exception as e:
                            print(f"Error processing user {user.id}: {e}")
                            processed_users += 1
                    
                    print(f"Notification generation completed: {processed_users}/{total_users} users, total {total_notifications} notifications")   
                    trans.commit() 
                    return processed_users, total_users, total_notifications
                except Exception as e:
                    trans.rollback()
                    print(f"Critical error during notification generation: {e}")
                    return 0, total_users, 0
        except Exception as e:
            print(f"Database connection error {e}")
            return 0,0,0



class PendingNotificationSender:
    """
    This class sends emails (one mail per user) with notifications generated for users in generate_notification.py. 
    Content of notification is based on saved publications
    in their notification-enabled folders.

    Configuration Requirements:
    SendGrid API Key:
    Create a file named 'sendgrid_api_key.txt' in the project root
    Place your own SendGrid API key there

    Sender Email:
    set from_email in send_email to your own sender email configured in sendgrid
    """
    def __init__(self,sender_email , api_key_file = 'sendgrid_api_key.txt'):
        self.api_key_file = api_key_file
        self.SENDER_EMAIL = sender_email
        self.app = Flask(__name__)

    def get_api_key(self):
        if not os.path.isfile(self.api_key_file):
            print("sendgrid api key file not found")
            return None
        return open(self.api_key_file, 'r').read().strip()

    def generate_mail_for_user(self, user_id: int):
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
            with self.app.app_context():
                return render_template_string(template, user_id=user_id, folders=folder_with_publications_ids), notifications_of_user

    def send_email(self, to, html, date): 
        if not to or not html:
            print("no email or html")
            return None
        
        print("sending email...")

        if not os.path.isfile(self.api_key_file):
            print(f"brak pliku {self.api_key_file}")
            return None
        api_key = open(self.api_key_file, 'r').read().strip()

        sg = sendgrid.SendGridAPIClient(api_key=api_key) 
        from_email = Email(f"{self.SENDER_EMAIL}")
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

    def mark_as_sent(self, sent_notifications_list, time):
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
            
    def send_all_pending_notifications(self):
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
                user_html, user_notifs = self.generate_mail_for_user(user.id)

                print(f"notification for a user with id {user.id} is ready to go")
                
                email = f"{user.email}"
                result = self.send_email(email, user_html, tnow_str+f" {user.email}")
                if result == 202:
                    print("email sent")
                    self.mark_as_sent(user_notifs, tnow)
            except Exception as e:
                print(f"error while working with notification for user with id:{user.id}, error: {e}")
                continue
        return 

if __name__ == "__main__":
    generator = NotificationGenerator()
    processed_users, total_users, total_notifications = generator.generate_all_notifications()
    if total_notifications > 0:
        print(f"Generated {total_notifications} notifications for {processed_users} users")
        sender = PendingNotificationSender('wyszukiwarka@gmail.com')
        sender.send_all_pending_notifications()
    else:
        print("no pending notifications to send")