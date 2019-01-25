#!/usr/bin/env python3
"""
Creates wallpaper with personal top albums from Last.fm.
"""

import argparse
import configparser
import datetime
import hashlib
import logging
import math
import os
import pylast
import requests
import shutil

from random import randrange
from PIL import Image, ImageFilter

DEFAULT_CONFIG_FILE_PATH = os.path.expanduser('~/.config/lastfm_wallpaper.ini')
DEFAULT_SERVER_NAME = 'default'
DEFAULT_API_URL = 'https://www.last.fm/api'
DEFAULT_ALBUM_COVER_DIR = os.path.expanduser('~/.cache/lastfm_wallpaper')
DEFAULT_MAX_COVER_COUNT = 12
DEFAULT_ROW_COUNT = 3
DEFAULT_SPACE = 50

API_GET_TOKEN = 'auth.gettoken'
API_TOP_ALBUMS = 'user.gettopalbums'

MISSING_CONFIG_ERROR = """\
You have to have your own unique two values for API_KEY and API_SECRET
Obtain yours from https://www.last.fm/api/account/create for Last.fm
and save them in ~/.config/lastfm_wallpaper.ini in following format.

    [{}]
    api_key = xxxxxxxxxxxxxxx
    api_secret = xxxxxxxxxxxxxxx
    user = login_name
"""

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class TupleArgument:
    def __init__(self, argument, separator):
        self.x, self.y = map(int, argument.split(separator))


class Size(TupleArgument):
    def __init__(self, size):
        super().__init__(size, 'x')


class Coordinates(TupleArgument):
    def __init__(self, coordinates):
        super().__init__(coordinates, ',')


def parse_config(config_path, server):
    config = configparser.ConfigParser(defaults={
        'URL': DEFAULT_API_URL,
    })
    config.read(config_path)
    try:
        return config[server]
    except KeyError:
        logger.error(MISSING_CONFIG_ERROR.format(server))
        exit(1)


def image_path(image_dir, base_name):
    return os.path.join(image_dir, '{}.png'.format(base_name))


def download_cover(album, path, cache_dir):
    album_id = '{} //// {}'.format(album.artist, album.title)
    cache_base_name = hashlib.sha256(album_id.encode('utf-8')).hexdigest()
    cached_path = os.path.join(cache_dir, cache_base_name)

    if os.path.isfile(cached_path):
        logger.info('Using cover: %s', album)
    else:
        try:
            cover_url = album.get_cover_image(pylast.COVER_MEGA)
            if not cover_url:
                logger.warning('Missing cover: %s', album)
                return False
        except pylast.WSError as e:
            logger.warning('Missing cover: %s (%s)', album, e)
            return False

        logger.info('Downloading cover: %s', album)
        r = requests.get(cover_url, stream=True)
        with open(cached_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    shutil.copyfile(cached_path, path)
    return True


def download_covers(api_key, api_secret, user, album_dir, from_date, to_date, max_count):
    cache_dir = os.path.join(album_dir, '.cache')
    os.makedirs(cache_dir, exist_ok=True)

    network = pylast.LastFMNetwork(
        api_key=api_key,
        api_secret=api_secret,
        username=user
    )

    user = network.get_user(user)
    top_items = user.get_weekly_album_charts(
        from_date=from_date.strftime('%s'),
        to_date=to_date.strftime('%s'))
    count = 0
    for top_item in top_items:
        album = top_item.item
        path = image_path(album_dir, count + 1)
        if download_cover(album, path, cache_dir):
            count += 1
            if count == max_count:
                break

    return count


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--config', default=DEFAULT_CONFIG_FILE_PATH,
        help='config file path')
    parser.add_argument(
        '--server', default=DEFAULT_SERVER_NAME,
        help='server name (section in config file)')
    parser.add_argument(
        '--dir', default=DEFAULT_ALBUM_COVER_DIR,
        help='directory to store album covers')
    parser.add_argument(
        '--size', default='1920x1080', type=Size,
        help='wallpaper size')
    parser.add_argument(
        '--count', default=DEFAULT_MAX_COVER_COUNT, type=int,
        help='maximum cover count')
    parser.add_argument(
        '--rows', default=DEFAULT_ROW_COUNT, type=int,
        help='number of rows')
    parser.add_argument(
        '--space', default=DEFAULT_SPACE, type=int,
        help='space between items')
    parser.add_argument(
        '--cached', action='store_true',
        help='use already downloaded covers')
    parser.add_argument(
        '--shadow-offset', default='1,1', type=Coordinates,
        help='shadow offset')
    parser.add_argument(
        '--shadow-blur', default=4, type=int,
        help='shadow offset')
    parser.add_argument(
        '--shadow-color', default='black',
        help='shadow color')
    parser.add_argument(
        '--border-color', default='black',
        help='border color')
    parser.add_argument(
        '--border-size', default=10, type=int,
        help='border size')
    parser.add_argument(
        '--days', default=7, type=int,
        help='number of days to consider')
    parser.add_argument(
        '--hours', default=0, type=int,
        help='number of additional hours to consider')
    parser.add_argument(
        '--base', default='random',
        help=('base image file'
              '; "random" to pick one of the covers'
              '; "top" to pick top album cover'))
    parser.add_argument(
        '--base-blur', default=3, type=int,
        help='base image blur')

    return parser.parse_args()


def background_image(path, width, height, blur):
    background = Image.open(path, 'r')
    background = background.convert('RGBA')
    extent = math.floor(max(width, height))
    x = (extent - width) // 2
    y = (extent - height) // 2
    background = background.resize((extent, extent), resample=Image.BICUBIC)
    background = background.crop((x, y, x + width, y + height))
    return background.filter(ImageFilter.GaussianBlur(radius=blur))


def main():
    args = parse_args()

    width, height = args.size.x, args.size.y
    album_dir = args.dir
    max_count = args.count

    config = parse_config(args.config, args.server)

    if args.cached:
        count = max_count
    else:
        to_date = datetime.datetime.now()
        from_date = to_date - datetime.timedelta(days=args.days) - datetime.timedelta(hours=args.hours)
        count = download_covers(
            api_key=config['api_key'],
            api_secret=config['api_secret'],
            user=config['user'],
            album_dir=album_dir,
            from_date=from_date,
            to_date=to_date,
            max_count=max_count
        )

    if count <= 0:
        logger.error("No albums in given time range")
        exit(1)

    if args.base == 'random':
        i = randrange(1, count)
        path = image_path(album_dir, i)
    elif args.base == 'top':
        path = image_path(album_dir, 1)
    else:
        path = args.base
    background = background_image(path, width, height, blur=args.base_blur)

    rows = args.rows
    columns = math.ceil(count / rows)
    space = args.space
    extent = min((height - space * rows) // rows, (width - space * columns) // columns)
    padding_x = (width - extent * columns) // (columns + 1)
    padding_y = (height - extent * rows) // (rows + 1)

    shadow_size = int(extent * 1.2)
    shadow = Image.new('RGBA', (shadow_size, shadow_size))
    shadow_pos = (shadow_size - extent) // 2
    shadow.paste(args.shadow_color, (shadow_pos, shadow_pos, extent + shadow_pos, extent + shadow_pos))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=args.shadow_blur))

    border = args.border_size

    for i in reversed(range(count)):
        path = image_path(album_dir, i + 1)
        img = Image.open(path, 'r')
        img = img.resize((extent - 2 * border, extent - 2 * border), resample=Image.BICUBIC)

        row, column = divmod(i, columns)
        x = column * extent + (column + 1) * padding_x
        y = row * extent + (row + 1) * padding_y

        dest = [x - shadow_pos + args.shadow_offset.x, y - shadow_pos + args.shadow_offset.y]
        src = [0, 0]
        for j in range(2):
            if dest[j] < 0:
                src[j] = -dest[j]
                dest[j] = 0
        background.alpha_composite(shadow, dest=tuple(dest), source=tuple(src))

        background.paste(args.border_color, box=(x, y, x + extent, y + extent))

        background.paste(img, box=(x + border, y + border))

    path = image_path(album_dir, 'wallpaper')
    background.save(path)
    print('Wallpaper saved: {}'.format(path))


if __name__ == "__main__":
    main()
