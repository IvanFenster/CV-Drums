# CV-Drums

Computer Vision Drums lets you play a simplified **virtual drum kit** using only your webcam and hand movements.  
Move your **index fingers** to strike four pads:

- **Tom** (bottom-left)  
- **Bass** (bottom-right)  
- **Crash** (left cymbal)  
- **Ride** (right cymbal)  

> If you don’t have real drums, CV-Drums gives you a fun way to feel like a musician!

---

## Features
- Real-time drum pad detection via webcam
- Four mapped drums (Tom, Bass, Crash, Ride)
- Playable with **only your index fingers**
- Instant audio playback from local samples
- Cross-platform (tested on Python 3.9+)

---

## Important notes
- Do **not hit too fast** – very rapid strikes may be missed.  
- One player only – avoid having other hands in the frame.  
- Turn on your sound before starting.

---

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/IvanFenster/CV-Drums.git
cd CV-Drums
pip install -r requirements.txt
