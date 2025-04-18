from db.SQLLite.OrmDB import Users, Notifications, SavedFolders, SavedPublications, engine
from sqlalchemy import select, and_, func
from datetime import datetime
import json

def get_publications(folder_id: int): 
    with engine.connect() as connection:
        query = select(SavedPublications.c.publication_id).where(
            SavedPublications.c.folder_id == folder_id
        ).limit(4)
        
        result = connection.execute(query)
        publication_ids = [row[0] for row in result]
    
    return publication_ids

def define_notification_for_user(user_id: int): 
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
        publications_in_notification = get_publications(folder.id)
        
        if publications_in_notification:
            notifications_data[folder.id] = publications_in_notification
    
    return notifications_data

def get_next_notification_id(connection): 
    query = select(func.max(Notifications.c.id)).select_from(Notifications)
    result = connection.execute(query)
    max_id = result.scalar()
    
    if max_id is None:
        next_id = 1
    else:
        next_id = max_id + 1
    
    return next_id

if __name__ == "__main__":
    print("Starting notification generation...")
    try: 
        with engine.connect() as connection:
            trans = connection.begin()
            query = select(Users)
            result = connection.execute(query)
            all_users = result.fetchall()
            
            total_users = len(all_users)
            processed_users = 0
            total_notifications = 0
            
            print(f"Found {total_users} users to process")
            
            for user in all_users:
                try: 
                    user_notifications = define_notification_for_user(user.id)
                     
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
                        
                        # chamskie genereowanie nastepnego id bo autoincrement Notifications.id nie dziala na nowej wersji Orm i dopiero do ogarnalem, program dzialaby bez tej funkcji gdyby autoincrement dzialal
                        next_id = get_next_notification_id(connection)
                        
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
            print("exiting the block")   
            trans.commit() 
    except Exception as e:
        trans.rollback()
        print(f"Critical error during notification generation: {e}")