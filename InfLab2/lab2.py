from datetime import datetime #текущее время 
from pathlib import Path #собирает пути к файлам
import numpy as np #хранение картинки как таблицы из чисел
from PIL import Image #для открытия и преобразования изображений

def svrt(channel, core): # свертка
    
    core = np.flipud(np.fliplr(core))  # в свёртке ядро переворачивается
    kh, kw = core.shape
    pad_y, pad_x = kh // 2, kw // 2
    h, w = channel.shape

    #края изображения дополняются копиями крайних пикселей
    padded = np.pad(channel.astype(np.float32), ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")

    # Сумма смещённых копий изображения, умноженных на элементы ядра
    result = np.zeros((h, w), dtype=np.float32)
    
    for i in range(kh):
        for j in range(kw):
            result += core[i, j] * padded[i:i + h, j:j + w]

    return np.clip(result, 0, 255).astype(np.uint8)


def gist(channel): #выравнивание гистограммы

    hist = np.bincount(channel.ravel(), minlength=256)  # гистограмма
    cdf = hist.cumsum() # накопленная гистограмма
    cdf_min = cdf[cdf > 0][0] # первое ненулевое значение
    total = channel.size

    if total == cdf_min:
        return channel

    #растягивание накопленной гистограмммы на 0 - 255
    lut = np.round((cdf - cdf_min) / (total - cdf_min) * 255)
    
    return lut.clip(0, 255).astype(np.uint8)[channel]


def image(img): # сначала свёртка, затем выравнивание гистограммы

    core = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]], dtype=np.float32)

    rgb = np.array(img.convert("RGB"))
    filtered = np.stack([svrt(rgb[:, :, c], core) for c in range(3)], axis=2)

    # Выравниваybt яркости Y
    y, cb, cr = Image.fromarray(filtered).convert("YCbCr").split()
    y = Image.fromarray(gist(np.array(y)))
    return Image.merge("YCbCr", (y, cb, cr)).convert("RGB")


# запись журнала
def write_log(log_path, file_name, filter_name, status):
    #Дописывает строку в history
    time_str = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    line = f"[{time_str}] Файл: {file_name} | Фильтр: {filter_name} | Статус: {status}\n"
    
    try:
        with open(log_path, "a") as log:
            log.write(line)
    except OSError:
        print("Предупреждение: не удалось записать журнал")


def main():
    # Пути к файловой системе
    base = Path(__file__).resolve().parent / "Lab2_Media"
    output = base / "data" / "output" / "result.jpg"
    log_path = base / "logs" / "history.txt"
    filter_name = "Свёртка + Выравнивание гистограммы"

    # потоковый ввод
    file_name = Path(input("Введите имя файла (например, image1.jpg): ").strip()).name
    if not file_name:
        print("Ошибка: Не указано имя файла")
        return

    def log(status):
        write_log(log_path, file_name, filter_name, status)

    # путь Lab2_Media/data/input/файл
    path = base / "data" / "input" / file_name
    try:
        # Режим чтение и запись
        with open(path, "r+b") as file:
            img = Image.open(file)
            img.load()  # читаем изображение целиком, пока файл открыт
    
    # обработка ошибок
    except FileNotFoundError:
        print("Ошибка: Файл не найден")
        log("Файл не найден")
        return
    except PermissionError:
        print(f"Ошибка: Нет прав на чтение файла {file_name}. Обратитесь к администратору")
        log("Ошибка доступа")
        return

    # обработка изображения
    result = image(img)

    # сохранение результата в output
    try:
        result.save(output, "JPEG")
    except PermissionError:
        print("Ошибка: Нет прав на запись в папку output. Обратитесь к администратору")
        log("Ошибка доступа")
        return

    # логирование
    log("Успешно")
    print("Готово! Результат сохранён: Lab2_Media/data/output/result.jpg")

if __name__ == "__main__":
    main()