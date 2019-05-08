from model import ClassPredictor
from vk_token import token
import torch
from config import hello_messages, hello_text
import numpy as np
import cv2
import vk_api
from vk_api.longpoll import VkLongPoll, VkEventType
import random
import urllib.request

num_attaches = 1
users = {}
model = ClassPredictor()
keys_photo = ["photo_75","photo_130","photo_604"]

def send_prediction_on_photo(user_id, img_path):
    print("Got image from {}".format(user_id))
   
    class_, prob_ = model.predict(img_path)
    vk.messages.send(user_id = event.user_id, message = "Я думаю это - {} и уверен в этом на {:.1f} %".format(class_, prob_), random_id = random.randint(0,1e32))
    print("Sent Answer to user, predicted: {}".format(class_))


if __name__ == '__main__':
    import logging
    print("Classifying bot was started")
    print(f"Your token - {token}")
    logging.basicConfig( format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',level=logging.INFO)
    
    vk_session = vk_api.VkApi(token= token)
    longpoll = VkLongPoll(vk_session, preload_messages=True)
    vk = vk_session.get_api()
    
    #send_prediction_on_photo(0, "sd")
 
    for event in longpoll.listen():
        if event.type == VkEventType.MESSAGE_NEW and event.to_me:
                print(f"Received a new message from {event.user_id}")
                if event.attachments: 
                    print(event.message_data)
                    url = ""
                    count_attach = 0;
                    img_path = "img.jpg"
                    for attach in event.message_data['attachments']:
                        for key, val in attach['photo'].items():
                            if(key in keys_photo):
                                url = val
                        if url == "":
                            for it in attach['photo']['sizes']:
                                for key, val in it.items():
                                    if(key == "url"):
                                        url = val
                        print(f"Url : {url}")
                        urllib.request.urlretrieve(url, img_path)
                        count_attach+=1
                        if(count_attach>=num_attaches):
                            break
                    if users.setdefault(event.user_id) is not None :
                        users[event.user_id] += 1
                    else:
                        users[event.user_id] = 1

                    send_prediction_on_photo(event.user_id, img_path)
                elif event.text:
                    vk.messages.send(user_id = event.user_id, message = "Я еще не до конца тебя понимаю. Лучше скинь картинку, в этом я разбираюсь лучше :)", random_id = random.randint(0,1e32))
                    
    print("Bot is stopped")