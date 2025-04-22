import pandas as pd

def process_technology_file(file_path, operation_list):
    """
    Обрабатывает файл .xls и формирует словарь с временем операций.

    :param file_path: Путь к файлу .xls
    :param operation_list: Список операций, которые нужно обработать (например, ["Фрезерная", "Токарная"])
    :return: Словарь с временем операций
    """
    # Чтение файла
    df = pd.read_excel(file_path, header=0)  # Первая строка содержит заголовки

    # Нормализация имен столбцов
    df.columns = df.columns.str.strip()  # Удаляем лишние пробелы в названиях столбцов

    # Проверка наличия необходимых столбцов
    required_columns = ["Наименование операции", "Т ш.т.", "Т п.з.", "P"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Файл не содержит всех необходимых столбцов: {required_columns}")

    # Получение количества деталей из ячейки P1
    quantity = df.iloc[0]["P"]
    if pd.isna(quantity):
        raise ValueError("Значение в столбце 'P' не должно быть пустым.")
    quantity = int(quantity)  # Преобразуем в целое число

    # Получение имени работы из имени файла
    job_name = file_path.split("/")[-1].split(".")[0]  # Убираем путь и расширение

    # Фильтрация и обработка операций
    result = {}
    for index, row in df.iterrows():
        operation_name = str(row["Наименование операции"]).strip().rstrip(":")  # Наименование операции
        # if operation_name == "Маркирование":
        #     operation_name = "Слесарная"

        if operation_name in operation_list:  # Проверяем, есть ли операция в списке
            unit_time = row["Т ш.т."]  # Время на единицу
            setup_time = row["Т п.з."]  # Подготовительное время

            # Проверка на NaN или None
            if pd.isna(unit_time) or pd.isna(setup_time):
                print(f"Пропущена строка с некорректными данными: {row}")
                continue

            # Вычисление общего времени
            total_time = unit_time * quantity + setup_time

            # Отладочный вывод
            print(f"Обнаружена операция: {operation_name}, Время: {total_time:.2f}")

            # Формирование ключа для операции
            operation_key = f"{job_name}, O{len(result) + 1}"

            # Добавление в результат
            result[(job_name, f"O{len(result) + 1}")] = {operation_name: round(total_time, 2)}

    return result


# Пример использования
if __name__ == "__main__":
    # Путь к файлу
    file_path = "ГОЛОВКА.xls"

    # Список операций, которые нужно обработать
    operation_list = ["Сверлильная", "Фрезерная", "Фрезерная c ЧПУ", "Токарная", "Токарная с ЧПУ", "Слесарная", "Расточная", "Заготовительная", "Маркирование"]

    # Обработка файла
    output = process_technology_file(file_path, operation_list)

    # Вывод результата
    print(output)