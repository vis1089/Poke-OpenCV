import cv2
import json
import os
from PIL import Image
from os import walk, remove

def match_images(img1, img2, sort=True):
    if img1 is None or img2 is None:
        raise ValueError("❌ Error: One or both input images are invalid (None). Check the image paths and formats.")
    
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        raise ValueError("❌ Error: Failed to compute descriptors. The images might not have enough features.")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if sort:
        matches = sorted(matches, key=lambda x: x.distance)
    return kp1, kp2, matches

def match_features_and_draw(img1, img2, name, result, sort=True):
    font = cv2.FONT_HERSHEY_DUPLEX
    org = (0, 25)
    fontScale = 1
    color = (255, 255, 5)
    thickness = 2

    kp1, kp2, matches = match_images(img1, img2, True)
    img2 = cv2.putText(img2, 'Source', org, font, fontScale, color, thickness, cv2.LINE_AA)
    img1 = cv2.putText(img1, 'Template', org, font, fontScale, color, thickness, cv2.LINE_AA)
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None)

    output_path = f"match_result_{name}.png"
    cv2.imwrite(output_path, match_img)
    print(f"✅ Match result saved: {output_path}")

class Cv2Image():
    def __init__(self, img_name):
        if not os.path.exists(img_name):
            raise FileNotFoundError(f"❌ Error: Image '{img_name}' does not exist! Check the file path.")
        
        self.template = cv2.imread(img_name)
        self.__name = img_name
        
        if self.template is None:
            raise ValueError(f"❌ Error: Image '{img_name}' could not be loaded. Make sure it's a valid image file.")

    def __str__(self):
        return self.__name

filenames = next(walk('./templates'), (None, None, []))[2]
test_poketwo_spawns = next(walk('./test_poketwo_spawns'), (None, None, []))[2]
template_list = []

poketwo_spawn = Cv2Image(os.path.abspath('./test_poketwo_spawns/palafin.png'))

def check_matching(src, templ): 
    templ_flipped = cv2.flip(templ, 1)
    kp1, kp2, matches = match_images(src, templ_flipped, False)
    return len(matches)

def path2pokename(path: str):
    return os.path.basename(path).replace(".png", "").capitalize()

def jpg2png(input: str):
    im1 = Image.open(input, 'r')
    im1.save(input.replace(".jpg", ".png"))
    im1.close()
    remove(input)

def check_and_convert_jpgs():
    for spawn in test_poketwo_spawns:
        if('.jpg' in spawn):
            jpg2png(f'./test_poketwo_spawns/{spawn}')

match_candidates = []

def main():
    check_and_convert_jpgs()

    with open('config.json') as f:
        config = json.load(f)

    print('[*] Initializing template pictures...')

    for filename in filenames:
        if('.jpg' in filename):
            jpg2png(f"./templates/{filename}")
        print(f'[*] Loaded "./templates/{filename}"!')
        template_list.append(Cv2Image(f"./templates/{filename}"))
    
    print(f"[*] Testing source: {str(poketwo_spawn)}")

    match_candidates = []

    for template in template_list:
        match_candidates.append({"name": str(template), "template": template.template, "result": check_matching(poketwo_spawn.template, template.template)})
    
    match_candidates.sort(key=lambda x: x['result'])
    
    if not match_candidates:
        print("❌ No matches found!")
        return

    best_match = match_candidates.pop()

    print(f"[*] Potential Match: '{path2pokename(best_match['name'])}' with score: {best_match['result']}")

    match_features_and_draw(best_match['template'], poketwo_spawn.template, path2pokename(best_match['name']), best_match['result'])

if __name__ == "__main__":
    main()
