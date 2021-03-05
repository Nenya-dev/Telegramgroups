# https://towardsdatascience.com/introduction-to-the-telegram-api-b0cd220dbed2
import asyncio
from collections import Counter

import bson
import pymongo
from telethon.sync import TelegramClient
from telethon.errors.rpcerrorlist import SessionPasswordNeededError, ChannelPrivateError
from telethon import functions
from pymongo import MongoClient
import queue
import configparser
import pickle
import json

parser = configparser.ConfigParser()
config_path = r'config.ini'
parser.read(config_path)

client_db = MongoClient(parser['mongoDB']["client"])
db = client_db['Telegram_Test']
posts = db["Querdenker_Test"]
list_groups = db.GroupList
my_col = db["QUERDENKER_Chat"]
query = {"Message": "/http/"}

# my keys
api_id = parser.getint('telegramKey', 'api_id')
api_hash = parser['telegramKey']["api_hash"]
phone = parser['telegramKey']["phone"]
username = parser['telegramKey']["username"]

client = TelegramClient(phone, api_id, api_hash)
client.connect()


def query_db_url():
    url_list = []
    my_doc = my_col.find(query)
    for url in my_doc:
        url_list.append(url)

    print(url_list)
    return url_list


# client connect
async def connect():
    client = TelegramClient(phone, api_id, api_hash)
    try:
        await client.connect()

    except OSError:
        print('Failed to connect')


async def authorization():
    # Authorized
    if not client.is_user_authorized():
        await client.send_code_request(phone)
        try:
            await client.sign_in(phone, input('Enter the code: '))
        except SessionPasswordNeededError:
            await client.sign_in(password=input('Password: '))


# get channel
def get_channel():
    entity = client.get_input_entity('')
    print(entity)
    result_1 = client(functions.channels.GetFullChannelRequest())
    print(result_1.stringify())


def get_message_history():
    not_found_groups = []
    i = 1
    with open('QD-test.txt', 'r', encoding='utf8') as file:
        groups = file.readlines()

    group_list = [x.strip() for x in groups]

    while i < len(group_list):
        try:
            for message in client.iter_messages(group_list[i], limit=100):
                # messages_dict = {'ID': message.sender_id, 'Message': message.text, "Fwd_group": message.from_id,
                #  "Reply": message.reply_to, "Group": message.is_group, "Channel": message.is_channel,
                #   "Forwarded": message.forwards, "Media": message.media, 'Date': message.date}
                messages_dict = {'ID': message.sender_id, 'Message': message.text, "Group": message.is_group,
                                 "Fwd_group": message.from_id,
                                 "Channel": message.is_channel, "Forwarded": message.forwards, 'Date': message.date}

                posts.insert_one(messages_dict)
        except ValueError:
            not_found_groups.append(group_list[i])
            pass

        i = i + 1


async def get_fwd_channel():
    # crawling groups
    queue_new_channels = queue.Queue()
    queue_new_channels.put('https://t.me/QuerdenkerChat')

    # result
    group_list = []

    while not queue_new_channels.empty():
        group = queue_new_channels.get()
        if group in group_list:
            continue
        group_list.append(group)

        if len(group_list) > 20:
            continue

        print('processing group', group)

        async for message in client.iter_messages(group, limit=100):
            # print('message')
            if message.fwd_from is None:
                continue
            try:
                fwd = message.fwd_from
                channel_name = await client.get_entity(fwd.from_id)
                queue_new_channels.put(channel_name.title)

            except Exception as error:
                # print(error)
                pass

    print(group_list)
    print(len(group_list))
    c = Counter(group_list)
    print(c)

    with open('QD-test.txt', 'w', encoding='utf-8-sig') as file:
        for g in group_list:
            file.write(g + '\n')

    return group_list


def search_groups():
    with TelegramClient(phone, api_id, api_hash) as client:
        result = client(functions.contacts.SearchRequest(
            q='Corona',
            limit=100
        ))
        # print(result.stringify())

    i = 0
    while i < 3:
        title = result.chats[i].title
        group_id = result.chats[i].id
        print(title, '(id: ', group_id, ')')
        i = i + 1


if __name__ == '__main__':
    # get_message_history()
    # asyncio.get_event_loop().run_until_complete(connect())
    # asyncio.get_event_loop().run_until_complete(get_fwd_channel())
    get_message_history()
    # query_db_url()
