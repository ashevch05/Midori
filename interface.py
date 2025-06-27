from midori import encrypt, decrypt  


def main():
    
    print("Це шифратор шифру Midori")
    
    mode = input("Бажаєте зашифрувати чи розшифрувати файл (enc чи dec)? ")
    if mode != "enc" and mode != "dec":
        print("Невірний режим")
        return

    input_file = input("Введіть ім’я файлу, для роботи: ")   
    output_file = input("Введіть м’я файлу, для результату: ")
    key_file = input("Введіть ім’я файла, в якому зберігається ключ шифрування: ")

    try:
        data = open(input_file, "r").read()
        key = open(key_file, "r").read()
    except FileNotFoundError as e:
        print(f"Помилка: {e}")
        return

    rounds = 20 

    try:
        if mode == "enc":
            result = encrypt(data, key, rounds)
        else:
            result = decrypt(data, key, rounds)
            
    except Exception as e:
        print(f"Сталася помилка : {e}")
        return

    with open(output_file, "w") as file:
        file.write(result)

    print(f"Результат записано у '{output_file}'")


if __name__ == "__main__":
    main()
