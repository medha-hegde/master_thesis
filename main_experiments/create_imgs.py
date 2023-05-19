import os
from PIL import Image, ImageDraw, ImageFont
import glob
import json

# Create images
def create_imgs(config):

    resolution = config["img_size"]
    background_colour = 'black'
    font_ttf = 'master_thesis/FontsFree-Net-arial-bold.ttf'
    font_colour = 'white'
    font_size = 10
    DIR_PREP = 'word_imgs'

    # mc4 words
    wiki_wrds = []
    with open('master_thesis/data/datasets/mc4_data.txt') as wrd_list:
        for wrd in wrd_list:
            wiki_wrds.append(wrd.replace('\n', '').lower())

    # function to print text on image
    def create_image(size, bgColor, message, font, fontColor):
        image = Image.new('RGB', size, bgColor)
        draw = ImageDraw.Draw(image)
        W, H = size
        _, _, w, h = draw.textbbox((0, 0), message, font=font)
        draw.text(((W - w) / 2, (H - h) / 2), message, font=font, fill=fontColor)
        return image

    # create images
    count = 0
    metadata_list = []

    if not os.path.exists(os.path.join(DIR_PREP, 'words')):
        os.makedirs(os.path.join(DIR_PREP, 'words'))

    for wrd in wiki_wrds:

        if count % 500 == 0:
            print(count)

        caption = wrd.upper()
        myFont = ImageFont.truetype(font_ttf, font_size)
        myImage = create_image((resolution, resolution), background_colour, caption, myFont, font_colour)
        myImage.save("%s/%s/%s.jpg" % (DIR_PREP, 'words', wrd))

        metadata_text = {}
        metadata_text["file_name"] = "%s.jpg" % (wrd)
        metadata_text["text"] = config["text_prompt"] + " \"%s\"" % (caption)
        metadata_list.append(metadata_text)
        count += 1

    # save image-caption pair data in json to follow this format:
    # https://huggingface.co/docs/datasets/image_load#imagefolder
    with open(os.path.join(DIR_PREP, 'words', "metadata.jsonl"), 'w') as f:
        for item in metadata_list:
            f.write(json.dumps(item) + "\n")

    # check number of files created: images + 1 metadata file
    print("Number of files created: ", len(glob.glob(DIR_PREP+'/words/*')))
