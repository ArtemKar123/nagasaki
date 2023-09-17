# About

This project is focused on mahjong tile recognition on images and videos.
It uses two YOLO v8 models trained for detection and classification, as well as background subtraction for better
results.

# Roadmap

- [x] Tile detection
- [x] Tile recognition
- [x] Yaku list based on photo
- [ ] Melds detection
- [ ] Open/closed hand state detection
- [ ] Table state log
- [ ] Live game log

# Example application
For now it can recognize tiles in your hand (aka-doras included), as well as the winning tile and return list of yaku.  
Yaku and shanten are counted using [mahjong library](https://github.com/MahjongRepository/mahjong).  
![Example GIF](media/tg.gif)
