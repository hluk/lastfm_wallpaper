Python script that creates wallpaper with covers from personal top chart on
Last.fm.

<p align="center">
  <img src="preview.png?raw=true" width="400"/>
</p>

## Dependencies

The script requires Python 3.8 or greater.

```bash
poetry install --no-root
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
poetry run lastfm_wallpaper --size 3840x2160
```

Show albums (and other info) in the last generated wallpaper:

```bash
poetry run lastfm_wallpaper --info
```

Use monthly album chart:

```bash
poetry run lastfm_wallpaper --days 30
```

Show fewer covers (six in two rows):

```bash
poetry run lastfm_wallpaper --rows 2 --count 6
```

Album chart from year ago:

```bash
poetry run lastfm_wallpaper --days-ago 365
```

List available command line switches:

```bash
poetry run lastfm_wallpaper --help
```

To avoid specifying the arguments repeatedly, you can set the options
in configuration file (the default configuration file path is printed with
`--help`). E.g. on Linux `~/.config/lastfm_wallpaper.ini` can contain:

```ini
[default]
api_key = xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
api_secret = xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
user = lastfm_login
size = 3840x2160
days = 90
days-ago = 1
search = ~/Music/{artist} - {album}/cover.*
```
