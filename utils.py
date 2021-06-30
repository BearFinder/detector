from PIL import Image, ImageDraw


def draw_boxes(img_file: str, boxes: list, color: str="#1700FF") -> Image: # file_save_name="outFile.JPG") -> Image:
    source_file = Image.open(img_file).convert("RGB")
    draw = ImageDraw.Draw(source_file)
    for i in range(boxes.shape[0]):
        box = boxes[i]
        # box[0]: x0, box[1]: y0, box[2]: x1, box[3]: y1
        draw.rectangle([box[0].item(), box[1].item(), box[2].item(), box[3].item()], outline=color, fill=None,
                       width=2)
    # source_file.save(file_save_name)
    return source_file

