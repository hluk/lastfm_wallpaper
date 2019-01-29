Python script that creates wallpaper with covers from personal top chart on
Last.fm.

<p align="center"> 
  <img src="preview.png?raw=true" width="400"/>
</p>

## Dependencies

The script requires Python 3.

It also requires installing dependencies listed in `requirements.txt`. E.g.

```bash
pip install -r requirements.txt
```

## Configuration

The first run of the script should end up with an error. Just follow the
instructions in the error message to create new configuration file.

## Create Wallpaper

By default, the script creates wallpaper with resolution `1920x1080` pixels.

The wallpaper will contain cover images of the top albums from personal weekly
chart. If a cover image cannot be found, the album is skipped.

Change the wallpaper size:

```bash
./lastfm_wallpaper.py --size 3840x2160
```

Show albums (and other info) in the last generated wallpaper:

```bash
./lastfm_wallpaper.py --info
```

Use monthly album chart:

```bash
./lastfm_wallpaper.py --days 30
```

Show fewer covers (six in two rows):

```bash
./lastfm_wallpaper.py --rows 2 --count 6
```

Album chart from year ago:

```bash
./lastfm_wallpaper.py --days-ago 365
```

List available command line switches:

```bash
./lastfm_wallpaper.py --help
```
