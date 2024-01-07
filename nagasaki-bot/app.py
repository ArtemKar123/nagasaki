from telegram.ext import MessageHandler, filters
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes
import telegram
import requests
import base64
from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter
import numpy as np
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.meld import Meld
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
import os

port = int(os.environ.get('PORT', 5001))
token = os.environ.get('TG_API_KEY', "")

shanten = Shanten()
calculator = HandCalculator()


def tile_comparator(x):
    if x == 0:
        return 5
    else:
        return x


def find_win_tile(tiles):
    tiles = sorted(tiles, key=lambda x: x['center'][0])
    d = []
    for i, tile in enumerate(tiles[:-1]):
        d.append(tiles[i + 1]['center'][0] - tile['center'][0])
    d.append(tiles[len(tiles) - 1]['center'][0] - tiles[len(tiles) - 2]['center'][0])

    return tiles[np.argmax(d)]['tile']


def meld_type(tiles):
    t = tiles[0]
    for tile in tiles:
        if tile != t:
            return Meld.CHI
    return Meld.PON


def process_data(data):
    tiles = {
        'm': [],
        'p': [],
        's': [],
        'z': []
    }
    print(data)

    clustered = {}
    for tile in data['tiles']:
        cluster = tile['cluster']
        if cluster not in clustered:
            clustered[cluster] = []
        clustered[cluster].append(tile)

    winning_cluster = sorted(list(clustered), key=lambda i: len(clustered[i]))[0]
    far_right_cluster = \
        sorted(list(clustered), key=lambda i: max(clustered[i], key=lambda xx: xx['center'][0])['center'][0],
               reverse=True)[0]
    open_cluster = []

    if len(list(clustered)) < 3:
        hand_cluster = far_right_cluster
    else:
        open_cluster = far_right_cluster

        open_cluster = clustered[open_cluster]
        for c in list(clustered):
            if c != winning_cluster and c != open_cluster:
                hand_cluster = clustered[c]
                break

    winning_cluster = clustered[winning_cluster]

    open_melds = []
    open_hand = ""
    if len(open_cluster) > 0:
        mean_tile_size = np.mean([x['size'] for x in data['tiles']])
        mean_open_y = np.mean([x['center'][1] for x in open_cluster])
        max_y_diff = abs(max([x['center'][1] for x in open_cluster], key=lambda y: abs(y - mean_open_y)) - mean_open_y)

        if max_y_diff > mean_tile_size / 2:
            # vertical
            open_cluster = sorted(open_cluster, key=lambda x: x['center'][1])
        else:
            # horizontal
            open_cluster = sorted(open_cluster, key=lambda x: x['center'][0])

        for i in range(int(len(open_cluster) / 3)):
            meld = open_cluster[i * 3:i * 3 + 3]
            meld = sorted(meld, key=lambda x: x['center'][0])
            open_hand += (''.join(
                f"{x['tile'][0]}" if x['tile'][0] != '0' else "<b>5</b>" for x in
                sorted(meld, key=lambda x: tile_comparator(x['tile']))) +
                          f"{meld[0]['tile'][1]}\n")
            open_melds.append([tile['tile'] for tile in meld])

    for tile in data['tiles']:
        n = tile['tile'][0]
        suit = tile['tile'][1]
        tiles[suit].append(int(n))

    hand = ""
    for suit in tiles.keys():
        if len(tiles[suit]) == 0:
            continue
        hand += ''.join(f"<b>5</b>" if x == 0 else str(x) for x in sorted(tiles[suit], key=tile_comparator)) + suit

    responce = "Hand is\n" + hand

    if len(data['tiles']) == 14:

        converted_tiles = TilesConverter.string_to_34_array(
            man=None if len(tiles['m']) == 0 else ''.join(str(tile_comparator(x)) for x in tiles['m']),
            pin=None if len(tiles['p']) == 0 else ''.join(str(tile_comparator(x)) for x in tiles['p']),
            sou=None if len(tiles['s']) == 0 else ''.join(str(tile_comparator(x)) for x in tiles['s']),
            honors=None if len(tiles['z']) == 0 else ''.join(
                str(x) for x in tiles['z']))

        shanten_n = shanten.calculate_shanten(converted_tiles)
        responce += f"\nShanten: {shanten_n}"

        if shanten_n < 0:
            converted_tiles = TilesConverter.string_to_136_array(
                man=None if len(tiles['m']) == 0 else ''.join(str(x) for x in tiles['m']),
                pin=None if len(tiles['p']) == 0 else ''.join(str(x) for x in tiles['p']),
                sou=None if len(tiles['s']) == 0 else ''.join(str(x) for x in tiles['s']),
                honors=None if len(tiles['z']) == 0 else ''.join(
                    str(x) for x in tiles['z']), has_aka_dora=True)

            win_tile = winning_cluster[0]['tile']  # find_win_tile(data['tiles'])

            responce += f"\nWinning tile: {win_tile}"

            if len(open_hand) > 0:
                responce += f"\nOpen melds:\n{open_hand}"

            print(win_tile)
            win_tile = TilesConverter.string_to_136_array(man=None if win_tile[1] != 'm' else win_tile[0],
                                                          pin=None if win_tile[1] != 'p' else win_tile[0],
                                                          sou=None if win_tile[1] != 's' else win_tile[0],
                                                          honors=None if win_tile[1] != 'z' else win_tile[0],
                                                          has_aka_dora=True)[0]
            melds = []
            for meld in open_melds:
                suit = meld[0][1]
                melds.append(
                    Meld(
                        meld_type=meld_type(meld),
                        tiles=TilesConverter.string_to_136_array(
                            man=None if suit != 'm' else ''.join(x[0] for x in meld),
                            pin=None if suit != 'p' else ''.join(x[0] for x in meld),
                            sou=None if suit != 's' else ''.join(x[0] for x in meld),
                            honors=None if suit != 'z' else ''.join(x[0] for x in meld),
                            has_aka_dora=True
                        )
                    )
                )

            result = calculator.estimate_hand_value(converted_tiles, win_tile, melds=melds,
                                                    config=HandConfig(
                                                        options=OptionalRules(has_aka_dora=True, has_open_tanyao=True)))
            print(result)
            responce += f"\nYaku list: {result.yaku}"

    return responce


async def process_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    message_id = update.message.message_id

    # handle image file received
    file = await context.bot.getFile(update.message.photo[-1].file_id)
    image_bytes = await file.download_as_bytearray()

    img_base64 = base64.b64encode(image_bytes).decode('utf-8')

    # use Docker network
    response = requests.post(f'http://nagasaki:{port}/tiles', json={"image": img_base64})

    # use default address
    # response = requests.post(f'http://127.0.0.1:{port}/tiles', json={"image": img_base64})

    await context.bot.sendMessage(chat_id=chat_id, text=process_data(data=response.json()),
                                  reply_to_message_id=message_id, parse_mode=telegram.constants.ParseMode.HTML)


def main():
    application = ApplicationBuilder().token(token).build()

    handler = MessageHandler(filters.PHOTO & (~filters.COMMAND), process_image)

    application.add_handler(handler)

    print('Started')
    application.run_polling()


if __name__ == '__main__':
    main()
