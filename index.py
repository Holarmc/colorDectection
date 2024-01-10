from colorthief import ColorThief
from src.colorDetection import logger
from src.colorDetection import knn_handler
from src.colorDetection import dataCleaning

import argparse

class ParseCmd():
    def __init__(self) -> None:
        pass
            
    def process_image(self, image_path):
        # Code to process image
        logger.info(f"Processing image at path: {image_path}")
        color_thief = ColorThief(image_path)
        # get the dominant color
        dominant_color = color_thief.get_color(quality=1)
        [R, G, B] = dominant_color
        X_vec = dataCleaning.rgb_to_embedding([[R, G, B]])

        knn_handler.load_model()
        output = knn_handler.predict(X_vec)
        return output[0]

    def process_color(self, R, G, B):
        # Code to process color
        logger.info(f"Processing color: R={R}, G={G}, B={B}")
        [R, G, B] = R, G, B
        X_vec = dataCleaning.rgb_to_embedding([[R, G, B]])

        knn_handler.load_model()
        output = knn_handler.predict(X_vec)
        return output[0]

    def parse(self):
        parser = argparse.ArgumentParser(description='Command Line Argument Parser')

        # Add arguments for processing image
        parser.add_argument('-p', '--image_path', help='Path to the image file')

        # Add arguments for processing color
        parser.add_argument('-r', '--color', nargs=3, type=int, metavar=('R', 'G', 'B'),
                            help='RGB values of the color')

        args = parser.parse_args()
        return args

        


if __name__ == "__main__":
    cmd = ParseCmd()
    args = cmd.parse()
    if args.image_path:
        cmd.process_image(args.image_path)
    elif args.color:
        cmd.process_color(*args.color)
    else:
        logger.info("No valid arguments provided.")




# build a color palette
# palette = color_thief.get_palette(color_count=6)