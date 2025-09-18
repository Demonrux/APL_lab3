import logic
import numpy


def console_interface():
    commands = {
        '1': {
            'name': 'Кластеризация методом К-средних',
            'func': logic.kmeans_clustering
        },
        '2': {
            'name': 'Иерархическая кластеризация',
            'func': logic.hierarchical_clustering
        },
        '3': {
            'name': 'DBSCAN-кластеризация',
            'func': logic.dbscan_clustering
        },
        '4': {
            'name': 'Выход',
            'func': None
        }
    }

    while True:

        for key, value in commands.items():
            print(f"{key}. {value['name']}")

        choice = input("Выберите действие (1-4): ").strip()

        if choice == '4':
            print("Выход...")
            break

        if choice in commands:
            try:
                commands[choice]['func'](numpy.loadtxt('files/dataset.txt'))
            except Exception as error:
                print(f" Ошибка при выполнении: {error}")
        else:
            print(" Неизвестная команда.")
