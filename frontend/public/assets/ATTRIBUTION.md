# Asset Attribution

All game art and audio in this directory are carried over **unchanged** (byte-for-byte
copies) from the original AIST1110 course project so the browser rebuild looks and sounds
exactly like the game it grew out of. The comments below are reproduced **verbatim** from
`TankWar/gym-tankwar/gym_tankwar/envs/assets.py`, which is where each asset's source and
license were recorded by the original authors.

Origin repo: https://github.com/shiyukfong/AIST1110_Project

---

## Tanks + bullets (craftpix freebies)

```python
class Player(_Tank):
    # Image source: https://craftpix.net/freebies/free-2d-battle-tank-game-assets/
    # License: https://craftpix.net/file-licenses/
    image_path = os.path.join(images_path, "tank_01/tank_01_A.png")
```

Applies to: `images/tank_01/tank_01_A.png`, `images/tank_01/tank_01_B.png`,
`images/tank_02/tank_02_A.png`, `images/tank_02/tank_02_B.png`,
`images/bullets/bullet_01.png`, `images/bullets/bullet_02.png`

```python
class Enemy(_Tank):
    # Image source: https://craftpix.net/freebies/free-2d-battle-tank-game-assets/
    # License: https://craftpix.net/file-licenses/
    image_path = os.path.join(images_path, "tank_02/tank_02_A.png")
```

```python
class _PlayerBullet(_Bullet):
    # Image source: https://craftpix.net/freebies/free-2d-battle-tank-game-assets/
    # License: https://craftpix.net/file-licenses/
    image_path = os.path.join(images_path, "bullets/bullet_01.png")
```

```python
class _EnemyBullet(_Bullet):
    # Image source: https://craftpix.net/freebies/free-2d-battle-tank-game-assets/
    # License: https://craftpix.net/file-licenses/
    image_path = os.path.join(images_path, "bullets/bullet_02.png")
```

## Heart (repair icon) — CC0

```python
class Heart(pygame.sprite.Sprite):
    # Image source: https://opengameart.org/content/heart-1
    # License: https://creativecommons.org/publicdomain/zero/1.0/
    image_path = os.path.join(images_path, "heart.png")
```

Applies to: `images/heart.png`

## Explosion sprites — gushh.net

```python
class Explosion(pygame.sprite.Sprite):
    # Image source: http://gushh.net/blog/free-game-sprites-explosion-4/
    # License: http://gushh.net/blog/free-game-sprites-explosion-1/
```

Applies to: `images/explosion/explosion.png`
(1024×1024 sprite sheet, 8×8 grid of 128 px frames — the browser rebuild samples it
directly via `drawImage` sub-rects instead of pre-cropping it into files. Note the file
is WEBP data carrying a `.png` extension; browsers content-sniff, so it is copied as-is.)

## Background — CC-BY 3.0

```python
class Background(pygame.sprite.Sprite):
    # Image source: https://opengameart.org/content/backgrounds-topdown-games
    # License: https://creativecommons.org/licenses/by/3.0/
    image_path = os.path.join(images_path, "background.png")
```

Applies to: `images/background.png`

## Audio

```python
class Audios:
    # Sound source: https://opengameart.org/content/war
    # License: https://creativecommons.org/publicdomain/zero/1.0/
    background_music = os.path.join(audios_path, "background_music.wav")

    # Sound source: https://opengameart.org/content/engine-loop-heavy-vehicletank
    # License 1: https://creativecommons.org/licenses/by/3.0/
    # License 2: https://www.gnu.org/licenses/gpl-3.0.html
    tank_engine_sound = os.path.join(audios_path, "tank_engine.ogg")

    # Sound source: https://opengameart.org/content/cannon-fire
    # License: https://creativecommons.org/publicdomain/zero/1.0/
    cannon_fire_sound = os.path.join(audios_path, "cannon_fire.ogg")

    # This work, made by Unnamed (Viktor.Hahn@web.de), is
    # licensed under the Creative Commons Attribution-ShareAlike 3.0 Unported License.
    # Sound source: https://opengameart.org/content/9-explosion-sounds
    # License: http://creativecommons.org/licenses/by-sa/3.0/
    explosion_sound = os.path.join(audios_path, "explosion.wav")
```

Applies to: `audios/background_music.wav` (CC0),
`audios/tank_engine.ogg` (dual CC-BY 3.0 / GPL-3.0),
`audios/cannon_fire.ogg` (CC0),
`audios/explosion.wav` (CC-BY-SA 3.0, by Viktor Hahn — Viktor.Hahn@web.de).
