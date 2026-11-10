"""Draw the app's Home Screen icons from the logo (web/public/icons/*.png). iPhones can't use SVG icons."""
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[1] / "web" / "public" / "icons"
BG, KEY, AMBER, BLUE = "#1d2230", "#f5f5f2", "#e8a33a", "#3b82c4"
SS = 4                                     # drawn large, then shrunk for smooth edges


def logo(size: int, zone: float = 1.0, rounded: bool = False) -> Image.Image:
    """The 64-unit logo; zone < 1 keeps it inside the middle of the icon (for "maskable" icons)."""
    S = size * SS
    img = Image.new("RGBA", (S, S), BG if not rounded else (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if rounded:
        d.rounded_rectangle((0, 0, S - 1, S - 1), radius=S * 14 // 64, fill=BG)
    u = S * zone / 64
    off = (S - 64 * u) / 2

    def rect(x, y, w, h, r, fill):
        d.rounded_rectangle((off + x * u, off + y * u, off + (x + w) * u, off + (y + h) * u), radius=r * u, fill=fill)

    for x in (10, 20, 30, 40):
        rect(x, 14, 8, 36, 2, KEY)
    rect(16, 14, 5, 22, 1.5, AMBER)
    rect(36, 14, 5, 22, 1.5, AMBER)
    rect(26, 14, 5, 22, 1.5, BLUE)
    return img.resize((size, size), Image.LANCZOS)


OUT.mkdir(parents=True, exist_ok=True)
# Apple rounds the corners itself and shows transparency as black, so its icon is a full square.
logo(180).convert("RGB").save(OUT / "apple-touch-icon.png")
logo(192, rounded=True).save(OUT / "icon-192.png")
logo(512, rounded=True).save(OUT / "icon-512.png")
logo(512, zone=0.72).convert("RGB").save(OUT / "icon-maskable-512.png")
print(sorted(p.name for p in OUT.iterdir()))

# The iPhone app (web/ios): a 1024 icon with no transparency, and the start-up picture.
XC = OUT.parents[1] / "ios" / "App" / "App" / "Assets.xcassets"
if XC.is_dir():
    logo(1024).convert("RGB").save(XC / "AppIcon.appiconset" / "AppIcon-512@2x.png")
    splash = Image.new("RGB", (2732, 2732), "#14161c")
    mark = logo(420, rounded=True)
    splash.paste(mark, ((2732 - 420) // 2, (2732 - 420) // 2), mark)
    for name in ("splash-2732x2732.png", "splash-2732x2732-1.png", "splash-2732x2732-2.png"):
        splash.save(XC / "Splash.imageset" / name, optimize=True)
    print("iPhone app icon and start-up picture written")
