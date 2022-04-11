#!/usr/bin/env python3
"""
Creates wallpaper with personal top albums from Last.fm.
"""

import argparse
import configparser
import datetime
import glob
import hashlib
import logging
import math
import os
import re
import shutil
import sys
from functools import lru_cache

import numpy
import pylast
import requests
from PIL import (
    Image,
    ImageChops,
    ImageEnhance,
    ImageFilter,
    ImageMath,
    ImageOps,
    PngImagePlugin,
)
from urllib3.util.retry import Retry

DEFAULT_CONFIG_FILE_PATH = os.path.expanduser("~/.config/lastfm_wallpaper.ini")
DEFAULT_SERVER_NAME = "default"
DEFAULT_ALBUM_COVER_DIR = os.path.expanduser("~/.cache/lastfm_wallpaper")
DEFAULT_MAX_COVER_COUNT = 12
DEFAULT_SPACE = 50

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_SIZE = "{}x{}".format(DEFAULT_WIDTH, DEFAULT_HEIGHT)

DEFAULT_MAX_TAGS_TO_MATCH = 2

MISSING_CONFIG_ERROR = """\
You have to have your own unique two values for API_KEY and API_SECRET Obtain
yours from https://www.last.fm/api/account/create and save them in following
format in file "{}".

    [{}]
    api_key = xxxxxxxxxxxxxxx
    api_secret = xxxxxxxxxxxxxxx
    user = login_name
"""

SEARCH_PATHS_EXAMPLE = os.path.pathsep.join(
    (
        os.path.join("~", "Music", "{artist} - {album}", "cover.*"),
        os.path.join("~", "Music", "*", "{artist} - {album}", "cover.*"),
    )
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def session():
    session = requests.Session()
    retry = Retry(
        total=3,
        read=3,
        connect=3,
        backoff_factor=1,
        status_forcelist=(500, 502, 503, 504),
        method_whitelist=Retry.DEFAULT_METHOD_WHITELIST.union(("POST",)),
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class DownloadCoverError(RuntimeError):
    pass


class TupleArgument:
    def __init__(self, argument, separator=","):
        self.x, self.y = map(int, argument.split(separator))


class SizeArgument(TupleArgument):
    def __init__(self, size):
        super().__init__(size, "x")


class LayoutArgument:
    def __init__(self, value):
        if value is None:
            self.positions = None
        else:
            self.positions, self.rows, self.columns = parse_layout(value)


class Layout:
    def __init__(
        self,
        positions,
        background,
        rows,
        columns,
        width,
        height,
        space,
        angle_range,
    ):
        self.positions = positions
        self.background = background
        self.rows = rows
        self.columns = columns
        self.space = space

        self.angle_range = angle_range
        self.angles = []

        self.extent = extent = min(
            (height - space * rows) // rows,
            (width - space * columns) // columns,
        )
        self.padding_x = (width - extent * columns) // (columns + 1)
        self.padding_y = (height - extent * rows) // (rows + 1)

    def paste(self, cell, img, offset=(0, 0)):
        img = rotate(img, self.angle(cell))

        row, column = self.position(cell, self.columns)
        x = (
            column * self.extent
            + self.extent // 2
            + (column + 1) * self.padding_x
        )
        y = row * self.extent + self.extent // 2 + (row + 1) * self.padding_y
        paste(img, x + offset[0], y + offset[1], self.background)

    def position(self, cell, columns):
        if self.positions:
            return self.positions[cell]

        return divmod(cell, columns)

    def angle(self, cell):
        if self.angle_range.x >= self.angle_range.y:
            return 0

        count_to_add = cell + 1 - len(self.angles)
        self.angles.extend(
            numpy.random.randint(self.angle_range.x, self.angle_range.y)
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
            img = Image.open(path, "r")
            img = img.convert("RGBA")
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
        raise SystemExit(MISSING_CONFIG_ERROR.format(config_path, server))


def parse_layout(value):
    try:
        positions = [
            [int(x) for x in position.split(",")]
            for position in value.split(" ")
        ]
        min_x = min(positions, key=lambda p: p[0])[0]
        max_x = max(positions, key=lambda p: p[0])[0]
        min_y = min(positions, key=lambda p: p[1])[1]
        max_y = max(positions, key=lambda p: p[1])[1]
        rows = max_x - min_x + 1
        columns = max_y - min_y + 1
        positions = [(x - min_x, y - min_y) for x, y in positions]
        return positions, rows, columns
    except Exception as e:
        logger.exception("Failed to parse positions argument: %s", e)
        raise


def image_info(**args):
    info = PngImagePlugin.PngInfo()
    for k, v in args.items():
        info.add_itxt(k, v)
    return info


def image_path(image_dir, base_name):
    return os.path.join(image_dir, "{}.png".format(base_name))


def fix_name(name):
    """
    Fixes album/artist name.
    """
    name = str(name).strip()
    if name.endswith(")"):
        return name.rsplit("(", 1)[0].rstrip()
    return name


def get_cover_image_from_lastfm(album):
    return album.get_cover_image(pylast.SIZE_MEGA)


def get_cover_image_from_deezer(album):
    url = "https://api.deezer.com/search/autocomplete"
    try:
        resp = session().get(
            url,
            params={
                "q": f"{fix_name(album.artist)} - {fix_name(album.title)}",
            },
        )
        resp.raise_for_status()
        return resp.json()["albums"]["data"][0]["cover_xl"]
    except Exception as e:
        logger.warning("Failed to fetch cover from %r: %s", url, e)


def cover_for_album(album):
    try:
        cover_url = get_cover_image_from_deezer(
            album
        ) or get_cover_image_from_lastfm(album)
        if not cover_url:
            raise DownloadCoverError("Cover URL not available")
    except Exception as e:
        raise DownloadCoverError("Failed to get cover URL: {}".format(e))

    return cover_url


def download_raw(url):
    try:
        r = session().get(url, stream=True)
    except requests.exceptions.RequestException as e:
        raise DownloadCoverError("Failed to download cover: {}".format(e))

    if r.status_code != 200:
        raise DownloadCoverError("Failed to download cover: {}".format(r.text))

    r.raw.decode_content = True
    return r.raw


def cache_path_for_album(album, cache_dir):
    album_id = "{} //// {}".format(album.artist, album.title)
    cache_base_name = hashlib.sha256(album_id.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, cache_base_name) + ".png"


def save_cover(album, raw_or_path, path):
    img = Image.open(raw_or_path)

    # Workaround for opening 16bit greyscale images.
    # See: https://github.com/python-pillow/Pillow/issues/2574
    if img.mode == "I" and numpy.array(img).max() > 255:
        logger.warning("Fixing 16bit image")
        img = ImageMath.eval("img/256", {"img": img})

    img = img.convert("RGBA")
    info = image_info(artist=album.artist.name, album=album.title)
    img.save(path, pnginfo=info)


def download_cover(album, cache_path):
    cover_url = cover_for_album(album)
    raw = download_raw(cover_url)
    save_cover(album, raw, cache_path)


def lastfm_user(api_key, api_secret, user):
    network = pylast.LastFMNetwork(
        api_key=api_key, api_secret=api_secret, username=user
    )

    return network.get_user(user)


def to_pattern(text):
    case_insensitive = "".join(
        "[{}{}]".format(c.lower(), c.upper()) if c.isalpha() else c
        for c in str(text)
    )
    return "*{}*".format(case_insensitive)


def find_album(album, search):
    for pattern in search:
        artist = to_pattern(album.artist)
        album_title = to_pattern(album.title)
        path = pattern.format(artist=artist, album=album_title)
        paths = glob.iglob(path)
        try:
            return next(paths)
        except StopIteration:
            pass


def get_cover_for_album(album, path, cache_path, search):
    if os.path.isfile(cache_path):
        logger.info('Album "%s": Using cached cover', album)
    else:
        found = find_album(album, search)
        if found:
            logger.info('Album "%s": Getting cover from "%s"', album, found)
            save_cover(album, found, cache_path)
        else:
            try:
                logger.info('Album "%s": Downloading cover', album)
                download_cover(album, cache_path)
            except DownloadCoverError as e:
                logger.warning(e)
                return False

    shutil.copyfile(cache_path, path)
    return True


def download_covers(
    user, album_dir, from_date, to_date, max_count, search, tag_re, max_tags
):
    cache_dir = os.path.join(album_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)

    top_items = user.get_weekly_album_charts(
        from_date=from_date.strftime("%s"), to_date=to_date.strftime("%s")
    )

    count = 0

    for top_item in top_items:
        album = top_item.item

        if tag_re:
            tags = album.artist.get_top_tags()
            if not any(
                tag_re.match(tag.item.get_name()) for tag in tags[:max_tags]
            ):
                tag_names = ", ".join(tag.item.get_name() for tag in tags)
                logger.info("No matching tags: %s (%s)", album, tag_names)
                continue

        path = image_path(album_dir, count + 1)
        cache_path = cache_path_for_album(album, cache_dir)
        if get_cover_for_album(
            album, path=path, cache_path=cache_path, search=search
        ):
            count += 1
            if count == max_count:
                break

    return count


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="print list of albums in the last wallpaper and exit",
    )

    parser.add_argument(
        "--config", default=DEFAULT_CONFIG_FILE_PATH, help="config file path"
    )
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER_NAME,
        help="server name (section in config file)",
    )
    parser.add_argument(
        "--dir",
        default=DEFAULT_ALBUM_COVER_DIR,
        help="directory to store album covers",
    )
    parser.add_argument(
        "--size",
        default=DEFAULT_SIZE,
        type=SizeArgument,
        help="wallpaper size",
    )
    parser.add_argument(
        "--count",
        default=DEFAULT_MAX_COVER_COUNT,
        type=int,
        help="maximum cover count",
    )
    parser.add_argument(
        "--rows", default=-1, type=int, help="number of rows; -1 to deduce"
    )
    parser.add_argument(
        "--columns",
        default=-1,
        type=int,
        help="number of columns; -1 to deduce",
    )
    parser.add_argument(
        "--space", default=DEFAULT_SPACE, type=int, help="space between items"
    )
    parser.add_argument(
        "--cached", action="store_true", help="use already downloaded covers"
    )
    parser.add_argument(
        "--search",
        default="",
        help='album search patterns (e.g. "{}")'.format(SEARCH_PATHS_EXAMPLE),
    )
    parser.add_argument(
        "--tags", default="", help="tag search pattern; regular expression"
    )
    parser.add_argument(
        "--max-tags",
        default=DEFAULT_MAX_TAGS_TO_MATCH,
        type=int,
        help="maximum number tags to search",
    )

    parser.add_argument(
        "--angle-range",
        default="0,0",
        type=TupleArgument,
        help="random cover rotation",
    )

    parser.add_argument(
        "--shadow-offset",
        default="1,1",
        type=TupleArgument,
        help="shadow offset",
    )
    parser.add_argument(
        "--shadow-blur", default=4, type=int, help="shadow blur"
    )
    parser.add_argument("--shadow-color", default="black", help="shadow color")

    parser.add_argument(
        "--border-color",
        default="black",
        help='border color; "auto" to auto-detect based on cover',
    )
    parser.add_argument(
        "--border-size", default=10, type=int, help="border size"
    )

    parser.add_argument(
        "--days", default=7, type=int, help="number of days to consider"
    )
    parser.add_argument(
        "--hours",
        default=0,
        type=int,
        help="number of additional hours to consider",
    )
    parser.add_argument(
        "--days-ago",
        default=0,
        type=int,
        help="consider end date X days ago instead of today",
    )

    parser.add_argument(
        "--base",
        default="random",
        help=(
            "base image file"
            '; "random" to pick one of the covers'
            "; <number> to pick the cover at given position"
        ),
    )
    parser.add_argument(
        "--base-blur", default=3, type=int, help="base image blur"
    )
    parser.add_argument(
        "--base-brightness",
        default=80,
        type=int,
        help="base image brightness percentage",
    )
    parser.add_argument(
        "--base-noise",
        default=10,
        type=int,
        help="base image noise percentage",
    )
    parser.add_argument(
        "--base-color",
        default=50,
        type=int,
        help="base image color percentage",
    )

    parser.add_argument(
        "--cover-brightness",
        default=100,
        type=int,
        help="cover image brightness percentage",
    )
    parser.add_argument(
        "--cover-noise",
        default=5,
        type=int,
        help="cover image noise percentage",
    )
    parser.add_argument(
        "--cover-color",
        default=100,
        type=int,
        help="cover image color percentage",
    )
    parser.add_argument(
        "--cover-glow", default=40, type=int, help="cover glow amount"
    )

    parser.add_argument(
        "--random-seed",
        default=-1,
        type=int,
        help=(
            "seed number to initialize random number generator; "
            "random if negative"
        ),
    )

    parser.add_argument(
        "--layout",
        default=None,
        type=LayoutArgument,
        help=(
            "cover positions and layout"
            '; space separated list of "row,column" values'
        ),
    )

    args = parser.parse_args()
    config = parse_config(args.config, args.server)
    config = {
        key.lower().replace("-", "_"): value for key, value in config.items()
    }
    parser.set_defaults(**config)

    return parser.parse_args()


def background_image(path, width, height, blur_radius):
    background = Image.open(path, "r")
    background = background.convert("RGBA")
    extent = math.floor(max(width, height))
    x = (extent - width) // 2
    y = (extent - height) // 2
    background = background.resize((extent, extent), resample=Image.BICUBIC)
    background = background.crop((x, y, x + width, y + height))
    return blur(background, blur_radius)


def add_noise(img, noise_percentage):
    if noise_percentage <= 0:
        return img

    noise = numpy.random.randint(
        0, 255, size=(img.height, img.width, 3), dtype=numpy.uint8
    )
    noise_image = Image.fromarray(noise, mode="RGB").convert("RGBA")
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
    return img.rotate(
        angle, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0, 0)
    )


def blur(img, radius):
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def glow(img, amount):
    extent1 = img.width
    img = colorize(img, 200)
    img = brighter(img, 200)
    img = blur(img, amount)

    extent2 = int(img.width * 0.6)
    mask = img.convert("L")
    mask = mask.resize((extent2, extent2))
    d = (extent1 - extent2) // 2
    mask = ImageOps.expand(mask, d, "black")
    mask = mask.resize((img.width, extent1))
    mask = blur(mask, amount)
    img.putalpha(mask)

    return img


def auto_border_color(img):
    img = img.resize((1, 1), resample=Image.BILINEAR)
    img = colorize(img, 200)
    img = brighter(img, 30)
    return img.getpixel((0, 0))


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

    numpy.random.seed(seed)


def print_info(album_dir):
    path = image_path(album_dir, "wallpaper")
    if not os.path.isfile(path):
        raise SystemExit("No wallpaper found")

    img = Image.open(path, "r")

    info = img.info
    albums = info.pop("albums", None)

    info["image"] = path
    info["resolution"] = "{}x{}".format(img.width, img.height)

    for k, v in sorted(info.items()):
        print("# {}: {}".format(k, v))

    if albums:
        print("# albums:\n\n{}".format(albums))

    sys.exit(0)


def main():
    args = parse_args()

    album_dir = args.dir

    if args.info:
        print_info(album_dir)

    init_random_seed(args.random_seed)

    width, height = args.size.x, args.size.y
    max_count = args.count

    rows = args.rows
    columns = args.columns

    if args.layout:
        positions = args.layout.positions
        if len(positions) < max_count:
            raise SystemExit(
                "Expected %s positions but %s specified",
                max_count,
                len(positions),
            )
        if rows < 0:
            rows = args.layout.rows
        if columns < 0:
            columns = args.layout.columns
    else:
        positions = None

    user = lastfm_user(
        api_key=args.api_key, api_secret=args.api_secret, user=args.user
    )

    image_info_dict = {}

    search = [
        os.path.expanduser(path) for path in args.search.split(os.path.pathsep)
    ]

    if args.cached:
        count = max_count
    else:
        to_date = datetime.datetime.now() - datetime.timedelta(
            days=args.days_ago
        )
        from_date = (
            to_date
            - datetime.timedelta(days=args.days)
            - datetime.timedelta(hours=args.hours)
        )
        image_info_dict["dates"] = "{}..{}".format(
            from_date.date(), to_date.date()
        )
        tag_re = re.compile(args.tags, re.IGNORECASE) if args.tags else None
        count = download_covers(
            user=user,
            album_dir=album_dir,
            from_date=from_date,
            to_date=to_date,
            max_count=max_count,
            search=search,
            tag_re=tag_re,
            max_tags=args.max_tags,
        )

    if count <= 0:
        raise SystemExit("No albums in given time range")

    if rows < 0 and columns < 0:
        x = width / height
        _, _, rows = min(
            [count % rows, abs(x - (count / rows) / rows), rows]
            for rows in range(1, count + 1)
        )
    if rows < 0:
        rows = math.ceil(count / columns)
    if columns < 0:
        columns = math.ceil(count / rows)

    scale = max(1, int(width / DEFAULT_WIDTH))
    base_blur = args.base_blur * scale

    loader = CoverLoader(album_dir)

    if args.base == "random":
        i = numpy.random.randint(count)
        path = loader.cover_path(i)
    else:
        try:
            i = int(args.base) - 1
            if i < 0:
                i = count + i + 1
            path = loader.cover_path(i)
        except TypeError:
            path = args.base
    background = background_image(path, width, height, blur_radius=base_blur)

    background = add_noise(background, args.base_noise)
    background = brighter(background, args.base_brightness)
    background = colorize(background, args.base_color)

    if not columns:
        columns = math.ceil(count / rows)
    space = args.space * scale
    layout = Layout(
        positions,
        background,
        rows,
        columns,
        width,
        height,
        space,
        args.angle_range,
    )
    extent = layout.extent

    shadow_size = int(extent * 1.2)
    shadow = Image.new("RGBA", (shadow_size, shadow_size))
    shadow_pos = (shadow_size - extent) // 2
    shadow.paste(
        args.shadow_color,
        (shadow_pos, shadow_pos, extent + shadow_pos, extent + shadow_pos),
    )
    shadow_blur = args.shadow_blur * scale
    shadow = blur(shadow, shadow_blur)
    shadow_offset = (
        args.shadow_offset.x * scale,
        args.shadow_offset.y * scale,
    )

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
        artist = img.info.get("artist")
        album = img.info.get("album")
        if artist and album:
            albums.insert(0, "{} - {}".format(artist, album))

        border_color = args.border_color
        if border_color == "auto":
            border_color = auto_border_color(img)

        img = ImageOps.expand(img, border, border_color)

        img = add_noise(img, args.cover_noise)
        img = brighter(img, args.cover_brightness)
        img = colorize(img, args.cover_color)

        layout.paste(i, img)

    albums = "\n".join(albums)
    if args.cached:
        logger.info("Using cached covers for albums:\n%s", albums)

    image_info_dict["albums"] = albums
    image_info_dict["url"] = user.get_url()

    path = image_path(album_dir, "wallpaper")
    info = image_info(**image_info_dict)
    background.save(path, pnginfo=info)
    print("Wallpaper saved: {}".format(path))


if __name__ == "__main__":
    main()
