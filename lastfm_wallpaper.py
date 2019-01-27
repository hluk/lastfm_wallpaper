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
import random
import requests
import shutil

from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps, PngImagePlugin

try:
    import numpy
except ImportError:
    pass

DEFAULT_CONFIG_FILE_PATH = os.path.expanduser('~/.config/lastfm_wallpaper.ini')
DEFAULT_SERVER_NAME = 'default'
DEFAULT_ALBUM_COVER_DIR = os.path.expanduser('~/.cache/lastfm_wallpaper')
DEFAULT_MAX_COVER_COUNT = 12
DEFAULT_ROW_COUNT = 3
DEFAULT_SPACE = 50

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_SIZE = '{}x{}'.format(DEFAULT_WIDTH, DEFAULT_HEIGHT)

API_GET_TOKEN = 'auth.gettoken'
API_TOP_ALBUMS = 'user.gettopalbums'

MISSING_CONFIG_ERROR = """\
You have to have your own unique two values for API_KEY and API_SECRET Obtain
yours from https://www.last.fm/api/account/create and save them in following
format in file "{}".

    [{}]
    api_key = xxxxxxxxxxxxxxx
    api_secret = xxxxxxxxxxxxxxx
    user = login_name
"""

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class TupleArgument:
    def __init__(self, argument, separator=','):
        self.x, self.y = map(int, argument.split(separator))


class Size(TupleArgument):
    def __init__(self, size):
        super().__init__(size, 'x')


class Layout:
    def __init__(self, background, rows, columns, width, height, space, angle_range):
        self.background = background
        self.rows = rows
        self.columns = columns
        self.space = space

        self.angle_range = angle_range
        self.angles = []

        self.extent = extent = min((height - space * rows) // rows, (width - space * columns) // columns)
        self.padding_x = (width - extent * columns) // (columns + 1)
        self.padding_y = (height - extent * rows) // (rows + 1)

    def paste(self, cell, img, offset=(0, 0)):
        img = rotate(img, self.angle(cell))

        row, column = divmod(cell, self.columns)
        x = column * self.extent + self.extent // 2 + (column + 1) * self.padding_x
        y = row * self.extent + self.extent // 2 + (row + 1) * self.padding_y
        paste(img, x + offset[0], y + offset[1], self.background)

    def angle(self, cell):
        if self.angle_range.x >= self.angle_range.y:
            return 0

        count_to_add = cell + 1 - len(self.angles)
        self.angles.extend(
            random.randrange(self.angle_range.x, self.angle_range.y)
            for _ in range(count_to_add)
        )

        return self.angles[cell]


class CoverLoader:
    def __init__(self, album_dir):
        self.album_dir = album_dir
        self.cache = {}

    def cover(self, index, extent):
        path = self.cover_path(index)
        img = self.cache.get(path)
        if not img:
            img = Image.open(path, 'r')
            img = img.convert('RGBA')
            self.cache[path] = img
        return img.resize((extent, extent), resample=Image.BICUBIC)

    def cover_path(self, index):
        return image_path(self.album_dir, index + 1)


def parse_config(config_path, server):
    config = configparser.ConfigParser()
    config.read(config_path)
    try:
        return config[server]
    except KeyError:
        logger.error(MISSING_CONFIG_ERROR.format(config_path, server))
        exit(1)


def image_info(**args):
    info = PngImagePlugin.PngInfo()
    for k, v in args.items():
        info.add_itxt(k, v)
    return info


def image_path(image_dir, base_name):
    return os.path.join(image_dir, '{}.png'.format(base_name))


def download_cover(album, path, cache_dir):
    album_id = '{} //// {}'.format(album.artist, album.title)
    cache_base_name = hashlib.sha256(album_id.encode('utf-8')).hexdigest()
    cached_path = os.path.join(cache_dir, cache_base_name) + '.png'

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
        r.raw.decode_content = True
        img = Image.open(r.raw)
        info = image_info(artist=album.artist.name, album=album.title)
        img.save(cached_path, pnginfo=info)

    shutil.copyfile(cached_path, path)
    return True


def lastfm_user(api_key, api_secret, user):
    network = pylast.LastFMNetwork(
        api_key=api_key,
        api_secret=api_secret,
        username=user
    )

    return network.get_user(user)


def download_covers(user, album_dir, from_date, to_date, max_count):
    cache_dir = os.path.join(album_dir, '.cache')
    os.makedirs(cache_dir, exist_ok=True)

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
        '--info', action='store_true',
        help='print list of albums in the last wallpaper and exit')

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
        '--size', default=DEFAULT_SIZE, type=Size,
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
        '--angle-range', default='0,0', type=TupleArgument,
        help='random cover rotation')

    parser.add_argument(
        '--shadow-offset', default='1,1', type=TupleArgument,
        help='shadow offset')
    parser.add_argument(
        '--shadow-blur', default=4, type=int,
        help='shadow blur')
    parser.add_argument(
        '--shadow-color', default='black',
        help='shadow color')

    parser.add_argument(
        '--border-color', default='black',
        help='border color; "auto" to auto-detect based on cover')
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
    parser.add_argument(
        '--base-brightness', default=80, type=int,
        help='base image brightness percentage')
    parser.add_argument(
        '--base-noise', default=30, type=int,
        help='base image noise percentage')
    parser.add_argument(
        '--base-color', default=50, type=int,
        help='base image color percentage')

    parser.add_argument(
        '--cover-brightness', default=100, type=int,
        help='cover image brightness percentage')
    parser.add_argument(
        '--cover-noise', default=10, type=int,
        help='cover image noise percentage')
    parser.add_argument(
        '--cover-color', default=100, type=int,
        help='cover image color percentage')
    parser.add_argument(
        '--cover-glow', default=40, type=int,
        help='cover glow amount')

    parser.add_argument(
        '--random-seed', default=-1, type=int,
        help='seed number to initialize random number generator; random if negative')

    return parser.parse_args()


def background_image(path, width, height, blur_radius):
    background = Image.open(path, 'r')
    background = background.convert('RGBA')
    extent = math.floor(max(width, height))
    x = (extent - width) // 2
    y = (extent - height) // 2
    background = background.resize((extent, extent), resample=Image.BICUBIC)
    background = background.crop((x, y, x + width, y + height))
    return blur(background, blur_radius)


def add_noise(img, noise_percentage):
    if noise_percentage <= 0:
        return img

    if not numpy:
        logger.warning('To add noise to image, install numpy')
        return img

    noise = numpy.random.normal(0, 1, (img.height, img.width, 3))
    a = numpy.amin(noise)
    b = numpy.amax(noise)
    noise = (255 * (noise - a) / (b - a)).astype(numpy.uint8)
    noise_image = Image.fromarray(noise, mode='RGB').convert('RGBA')
    return ImageChops.blend(img, noise_image, noise_percentage / 100)


def brighter(img, brightness_percentage):
    return ImageEnhance.Brightness(img).enhance(brightness_percentage / 100)


def colorize(img, colorize_percentage):
    return ImageEnhance.Color(img).enhance(colorize_percentage / 100)


def rotate(img, angle):
    if (angle % 360) == 0:
        return img

    # Expand first to have smoother edges.
    img = ImageOps.expand(img, 4, fill=0)
    return img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0, 0))


def blur(img, radius):
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def glow(img, amount):
    extent1 = img.width
    img = colorize(img, 200)
    img = brighter(img, 200)
    img = blur(img, amount)

    extent2 = int(img.width * 0.6)
    mask = img.convert('L')
    mask = mask.resize((extent2, extent2))
    d = (extent1 - extent2) // 2
    mask = ImageOps.expand(mask, d, 'black')
    mask = mask.resize((img.width, extent1))
    mask = blur(mask, amount)
    img.putalpha(mask)

    return img


def paste(img, x, y, background):
    dest = [x - img.width // 2, y - img.height // 2]
    src = [0, 0]
    for j in range(2):
        if dest[j] < 0:
            src[j] = -dest[j]
            dest[j] = 0
    background.alpha_composite(img, dest=tuple(dest), source=tuple(src))


def init_random_seed(seed):
    if seed < 0:
        return

    random.seed(a=seed, version=2)
    if numpy:
        numpy.random.seed(seed)


def print_info(album_dir):
    path = image_path(album_dir, 'wallpaper')
    if not os.path.isfile(path):
        logger.error("No wallpaper found")
        exit(1)

    img = Image.open(path, 'r')

    info = img.info
    albums = info.pop('albums', None)

    info['image'] = path
    info['resolution'] = '{}x{}'.format(img.width, img.height)

    for k, v in sorted(info.items()):
        print('# {}: {}'.format(k, v))

    if albums:
        print('# albums:\n\n{}'.format(albums))

    exit(0)


def main():
    args = parse_args()

    album_dir = args.dir

    if args.info:
        print_info(album_dir)

    init_random_seed(args.random_seed)

    width, height = args.size.x, args.size.y
    max_count = args.count

    config = parse_config(args.config, args.server)

    user = lastfm_user(
        api_key=config['api_key'],
        api_secret=config['api_secret'],
        user=config['user']
    )

    image_info_dict = {}

    if args.cached:
        count = max_count
    else:
        to_date = datetime.datetime.now()
        from_date = to_date - datetime.timedelta(days=args.days) - datetime.timedelta(hours=args.hours)
        image_info_dict['dates'] = '{}..{}'.format(from_date.date(), to_date.date())
        count = download_covers(
            user=user,
            album_dir=album_dir,
            from_date=from_date,
            to_date=to_date,
            max_count=max_count
        )

    if count <= 0:
        logger.error("No albums in given time range")
        exit(1)

    scale = max(1, int(width / DEFAULT_WIDTH))
    base_blur = args.base_blur * scale

    loader = CoverLoader(album_dir)

    if args.base == 'random':
        i = random.randrange(count)
        path = loader.cover_path(i)
    elif args.base == 'top':
        path = loader.cover_path(0)
    else:
        path = args.base
    background = background_image(path, width, height, blur_radius=base_blur)

    background = add_noise(background, args.base_noise)
    background = brighter(background, args.base_brightness)
    background = colorize(background, args.base_color)

    rows = args.rows
    columns = math.ceil(count / rows)
    space = args.space * scale
    layout = Layout(background, rows, columns, width, height, space, args.angle_range)
    extent = layout.extent

    shadow_size = int(extent * 1.2)
    shadow = Image.new('RGBA', (shadow_size, shadow_size))
    shadow_pos = (shadow_size - extent) // 2
    shadow.paste(args.shadow_color, (shadow_pos, shadow_pos, extent + shadow_pos, extent + shadow_pos))
    shadow_blur = args.shadow_blur * scale
    shadow = blur(shadow, shadow_blur)
    shadow_offset = (args.shadow_offset.x * scale, args.shadow_offset.y * scale)

    border = args.border_size * scale

    for i in range(count):
        layout.paste(i, shadow, shadow_offset)

    if args.cover_glow > 0:
        for i in reversed(range(count)):
            extent1 = int(extent * (100 + args.cover_glow) / 100)
            img = loader.cover(i, extent1)
            img = glow(img, extent // 10)
            layout.paste(i, img)

    albums = []

    for i in reversed(range(count)):
        extent1 = extent - 2 * border
        img = loader.cover(i, extent1)
        artist = img.info.get('artist')
        album = img.info.get('album')
        if artist and album:
            albums.insert(0, '{} - {}'.format(artist, album))

        border_color = args.border_color
        if border_color == 'auto':
            img1 = img.resize((1, 1), resample=Image.BILINEAR)
            img1 = colorize(img1, 200)
            img1 = brighter(img1, 50)
            border_color = img1.getpixel((0, 0))

        img = ImageOps.expand(img, border, border_color)

        img = add_noise(img, args.cover_noise)
        img = brighter(img, args.cover_brightness)
        img = colorize(img, args.cover_color)

        layout.paste(i, img)

    albums = '\n'.join(albums)
    if args.cached:
        logger.info('Using cached covers for albums:\n%s', albums)

    image_info_dict['albums'] = albums
    image_info_dict['url'] = user.get_url()

    path = image_path(album_dir, 'wallpaper')
    info = image_info(**image_info_dict)
    background.save(path, pnginfo=info)
    print('Wallpaper saved: {}'.format(path))


if __name__ == "__main__":
    main()
