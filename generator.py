""" Created by MrBBS """
# 4/14/2021
# -*-encoding:utf-8-*-

from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter, ImageChops
import cv2
from tqdm import tqdm
import random
import numpy as np
from pathlib import Path
from termcolor import colored
import string
import os
import warnings
# from gen_text import gen_email, gen_phone, gen_address
from threading import Thread

warnings.filterwarnings("ignore", category=UserWarning)  # Tắt UserWarning
os.system('color')
vocab = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\t '

def print_error(name, mess=None):
    print(colored('\n[ERROR]', 'red'), name + ':', mess)


def print_info(name, isStart=True):
    if isStart:
        print(colored('\n[INFO]', 'cyan'), f'{name}: Start')
    else:
        print(colored('\n[INFO]', 'cyan'), f'{name}: End')

def get_font(path):
    try:
        print_info('get_font')
        fonts = []
        for p in Path(path).rglob('*.[ot][tt]*'):
            font = ImageFont.truetype(p.as_posix(), size=22, encoding='utf-8')
            fonts.append(font)
        print_info('get_font', False)
        return fonts
    except Exception as e:
        print_error('get_font', e)


def random_color(light=False, dark=False, hex_code=True):
    """
    Tạo màu ngẫu nhiên
    :param light: tạo màu sắc sáng
    :param hex_code: tạo mã màu hex
    :return: (light = False - hex_code = True) mã màu HEX của màu sắc sáng / (light = True) mã màu RGB
    """
    if dark:
        gamma = random.randint(0, 68)
        return gamma, gamma, gamma
    if light:
        return random.randint(128, 256), random.randint(128, 256), random.randint(128, 256)
    else:
        if hex_code:
            return "#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
        else:
            return random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)


def get_gradient_2d(start, stop, width, height, is_horizontal):
    """
    Tạo ảnh gradient 2D với thang màu trắng đen
    :param start: Màu sắc bắt đầu
    :param stop: Màu sắc kết thúc
    :param width: Chiều rộng
    :param height: Chiều cao
    :param is_horizontal: Bool - True: Gradient ngang - False: Gradient dọc
    :return: Ảnh gradient đen trắng
    """
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T


def get_gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
    """
    Tạo ảnh gradient 3D
    :param width: Chiều rộng
    :param height: Chiều cao
    :param start_list: Màu sắc bắt đầu
    :param stop_list: Màu sắc kết thúc
    :param is_horizontal_list: Danh sách bool > 3 item để xác định xem có bao nhiêu vùng gradient được tạo
    :return: Ảnh gradient
    """
    result = np.zeros((height, width, len(start_list)), dtype=np.float32)

    for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
        result[:, :, i] = get_gradient_2d(start, stop, width, height, is_horizontal)

    return result


def choice_background(use_color=True, width=300, heigth=30):
    """
    Tạo nền với màu sắc hoặc dùng ảnh từ thư mục chỉ định
    :param use_color: Bool - True ( Mặc định ): Tạo nền với màu - False: Sử dụng ảnh từ thư mục chỉ định
    :return: Ảnh nền <class 'PIL.Image.Image'>
    """
    try:
        if use_color:
            if bool(random.getrandbits(1)):
                L = Image.new('L', (width, heigth), random.randint(0, 255))
                RGB = Image.merge('RGB', (L, L, L))
                return RGB
            else:
                start_color = random_color(hex_code=False)
                end_color = random_color(hex_code=False)
                list_horizontal = sorted([bool(random.getrandbits(1)) for _ in range(random.randint(4, 7))],
                                         reverse=True)
                array_gradient = get_gradient_3d(width, heigth, start_color, end_color, list_horizontal)
                gradient = Image.fromarray(np.uint8(array_gradient))
                return gradient
        else:
            path = random.choice(list(Path('backgrounds').rglob('*.[jp][pn]*')))
            background_image = Image.open(path.as_posix())
            if background_image.size[0] > width and background_image.size[1] > heigth:
                left = random.randint(0, background_image.size[0] - width)
                right = left + width
                top = random.randint(0, background_image.size[1] - heigth)
                bottom = top + heigth
                return background_image.crop((left, top, right, bottom))
            return background_image.resize((width, heigth))
    except Exception as e:
        print_error('choice_background', e)


def drop_shadow(image, input_text, input_font, position, offset=(7, 7), shadow_color=None, border=8):
    """
    Đổ bóng cho chữ
    :param image: Hình ảnh đầu vào
    :param input_text: Chữ cần đổ bóng
    :param input_font: Font của chữ
    :param position: Vị trí của chữ trên ảnh đầu vào
    :param offset: Tọa độ ngang - dọc của bóng
    :param shadow_color: Màu bóng
    :param border: Độ dày bóng
    :return: Ảnh chứ chữ đã được đổ bóng
    """
    try:
        max_h = abs(offset[0]) + border
        max_v = abs(offset[1]) + border
        if shadow_color is None:
            shadow_color = random_color(dark=True)
        shadow = Image.new(image.mode, image.size, 0xffffff)
        for x in range(max_h):
            for y in range(max_v):
                shadowHorizontal = x + position[0] if offset[0] > 0 else position[0] - x
                shadowVertical = y + position[1] if offset[1] > 0 else position[1] - y
                ImageDraw.Draw(shadow).text((shadowHorizontal, shadowVertical), input_text, shadow_color, input_font)
        shadow = Image.alpha_composite(shadow, image)
        return shadow
    except Exception as e:
        print_error('drop_shadow', e)


def overlay(width=300, height=30):
    """ Sử dụng lớp phủ """
    try:
        overlay_image = random.choice(list(Path('overlay').rglob('*.[jp][pn]*')))
        original = Image.open(overlay_image.as_posix())
        L = original.convert('L')
        A = Image.new('L', original.size, random.randint(30, 95))  # Tạo ảnh tạm với độ trong suốt ngẫu nhiên
        original = Image.merge('RGBA', (L, L, L, A))
        original = original.filter(ImageFilter.BLUR)  # Làm sạch ảnh và làm mờ
        try:
            left = random.randint(0, original.size[0] - width)
            top = random.randint(0, original.size[1] - height)
        except:
            left = random.randint(0, original.size[0])
            top = random.randint(0, original.size[1])
        right = left + width
        bottom = top + height
        return original.crop((left, top, right, bottom))
    except Exception as e:
        print_error('overlay', e)


def change_view(img):
    img = np.array(img)
    rows, cols, ch = img.shape
    pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
    pts2 = np.float32([[0 + random.randint(-20, 20), 0], [cols + random.randint(-20, 20), 0], [0, rows], [cols, rows]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (cols, rows))
    return dst


def invert_color_image(image):
    if image.mode == 'RGBA':
        r, g, b, a = image.split()
        rgb_image = Image.merge('RGB', (r, g, b))
        inverted_image = ImageOps.invert(rgb_image)
        r2, g2, b2 = inverted_image.split()
        result = Image.merge('RGBA', (r2, g2, b2, a))
    else:
        result = ImageOps.invert(image)
    return result


count_train = 0
count_test = 0
no_folder = 0
count_all = 0
count_flag = 0
characters = set()
char = string.ascii_letters + string.digits

def generator(line, fonts, train_set=True):
    global count_train, count_test, count_all, count_flag, no_folder, characters
    line = line.strip()
    isPass = True
    for c in line:
        if c not in vocab:
            isPass = False
            break
        if c not in characters:
            characters.add(c)
    if isPass:
        for font in random.choices(fonts, k=5):
            if len(line) < 1:
                continue
            wT, hT = font.getsize(line)
            random_name = ''.join(random.choices(char, k=16))
            angle = random.randint(-5, 5)
            txt = Image.new('L', (wT + 25, hT + 25))
            ImageDraw.Draw(txt).text((12, 12), line, random_color(), font)
            txt = txt.rotate(angle, Image.BICUBIC, expand=1)
            background = choice_background(bool(random.getrandbits(1)), *txt.size)
            background = Image.fromarray(cv2.copyMakeBorder(np.array(background), 0, 0, 35, 35, cv2.BORDER_REFLECT))
            main_color = background.resize((1, 1)).getpixel((0, 0))
            txt = Image.new('RGBA', (wT + 25, hT + 25))
            ImageDraw.Draw(txt).text((12, 12), line, main_color, font)
            txt = Image.fromarray(change_view(txt))
            txt = txt.rotate(angle, Image.BICUBIC, expand=1)
            txt = invert_color_image(txt)
            pos = ((background.size[0] - txt.size[0]) // 2, (background.size[1] - txt.size[1]) // 2)
            background.paste(txt, pos, txt)
            if bool(random.getrandbits(1)):
                background = background.convert('RGBA')
                background = Image.alpha_composite(background, overlay(*background.size))
            Path(f'images/{no_folder}').mkdir(parents=True, exist_ok=True)
            background.save(f'images/{no_folder}/{random_name}.png', 'PNG')
            file_anno_name = 'train_line_annotation'
            if not train_set:
                file_anno_name = 'test_line_annotation'
            with open(f'{file_anno_name}.txt', 'a', encoding='utf-8') as f:
                f.write(f'data/images/{no_folder}/{random_name}.png\t{line}\n')
            if train_set:
                count_train += 1
            else:
                count_test += 1
            count_all += 1
            count_flag += 1
            print(count_all, end='\r')
            with open('log.txt', 'w') as f:
                f.write('Tong hinh da tao ' + str(count_all) + '\nHinh train ' + str(count_train) + '\nHinh test ' + str(count_test) + '\nSo luong folder ' + str(no_folder + 1))
            if count_flag > 19999:
                no_folder += 1
                count_flag = 0
        with open('character.txt', 'w', encoding='utf-8') as f:
            for c in characters:
                f.write(c + ' ')


def run_gen(lines=None, fonts=[], train_set=True):
    for line in lines:
        try:
            generator(line, fonts, train_set)
        except Exception as e:
            print_error('gen',e)
            pass


if __name__ == '__main__':
    Path('images').mkdir(parents=True, exist_ok=True)
    num_worker = 9
    thread_list = []
    fonts = get_font('fonts')
    print_info('generation')
    data = open('data.txt', 'r', encoding='utf-8').readlines()
    np.random.shuffle(data)
    split_rate = int(len(data) * 0.75)
    step_train_rate = split_rate // num_worker
    len_test = len(data) - split_rate
    step_test_rate = len_test // num_worker
    data_train_batch = [data[:split_rate][i:i + step_train_rate] for i in range(0, split_rate, step_train_rate)]
    data_test_batch = [data[split_rate:][i:i + step_test_rate] for i in range(0, len_test , step_test_rate)]
    for batch_train, batch_test in zip(data_train_batch, data_test_batch):
        th = Thread(target=run_gen, args=(batch_train, fonts), daemon=True)
        th.start()
        thread_list.append(th)
        th = Thread(target=run_gen, args=(batch_test, fonts, False), daemon=True)
        th.start()
        thread_list.append(th)
    for th in thread_list:
        th.join()
