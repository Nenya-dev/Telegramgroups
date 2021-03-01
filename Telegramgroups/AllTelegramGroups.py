# https://towardsdatascience.com/introduction-to-the-telegram-api-b0cd220dbed2
import asyncio
from collections import Counter

from telethon.sync import TelegramClient
from telethon.errors.rpcerrorlist import SessionPasswordNeededError, ChannelPrivateError
from telethon import functions
from pymongo import MongoClient
import queue
import config as cfg


client_db = MongoClient(cfg.mongo_db["client"])
db = client_db[cfg.mongo_db["db"]]
posts = db.QUERDENKER_Chat
list_groups = db.GroupList

# my keys
api_id = cfg.telegram_key["api_id"]
api_hash = cfg.telegram_key["api_hash"]
phone = cfg.telegram_key["phone"]
username = cfg.telegram_key["username"]


# client connect
async def connect():
    client = TelegramClient(phone, api_id, api_hash)
    try:
        await client.connect()
    except OSError:
        print('Failed to connect')


async def authorization():
    client = TelegramClient(phone, api_id, api_hash)
    # Authorized
    if not client.is_user_authorized():
        await client.send_code_request(phone)
        try:
            await client.sign_in(phone, input('Enter the code: '))
        except SessionPasswordNeededError:
            await client.sign_in(password=input('Password: '))


channel = 'ZDF Magazin Royale ✅'
# channel_entity = client.get_entity(channel)

group_name = "https://t.me/memehub_meta"
# group_entity = client.get_entity(group_name)

group_list_news = ['https://t.me/coronadiewahrheit']

# me = client.get_me()

'''
# get history messages
posts = client(functions.messages.GetHistoryRequest(
    peer=group_entity,
    limit=10,
    offset_date=None,
    offset_id=0,
    max_id=0,
    min_id=0,
    add_offset=0,
    hash=0)
)
'''


# get channel
def get_channel():
    with TelegramClient(phone, api_id, api_hash) as client:
        entity = client.get_input_entity('https://t.me/best_memes_meme_hub')
        print(entity)
        result_1 = client(functions.channels.GetFullChannelRequest(
            channel='@best_memes_meme_hub'
        ))
        print(result_1.stringify())


def get_message_history():
    with TelegramClient(phone, api_id, api_hash) as client:
        for message in client.get_messages(group_name, limit=10000):
            messages_dict = {'ID': message.sender_id, 'Message': message.text, 'Date': message.date}
            posts.insert_one(messages_dict)

        # print('one post: {0}'.format(result.inserted_id))
        # print(messages_dict)


async def get_fwd_channel():
    # crawling groups
    queue_new_channels = queue.Queue()
    # queue_new_channels.put('https://t.me/QuerdenkerChat')
    queue_new_channels.put('https://t.me/coronadiewahrheit')
    # result
    group_list = []
    channel_list = []

    while not queue_new_channels.empty():
        group = queue_new_channels.get()
        if group in group_list:
            continue
        group_list.append(group)

        if len(group_list) > 20:
            continue
        print('processing group', group)

        async with TelegramClient(phone, api_id, api_hash) as client:
            async for message in client.iter_messages(group, limit=10000):
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
    asyncio.get_event_loop().run_until_complete(connect())
    asyncio.get_event_loop().run_until_complete(get_fwd_channel())
