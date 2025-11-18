import os
from collections import Counter

def compare_markups(original_file, generated_file, output_file="comparison_results.txt"):
    """
    Сравнивает две разметки и выводит статистику
    """
    # Загружаем оригинальную разметку
    original_data = {}
    try:
        with open(original_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        frame_num = int(parts[0])
                        direction = parts[1]
                        original_data[frame_num] = direction
    except FileNotFoundError:
        print(f"❌ Ошибка: Файл {original_file} не найден")
        return
    except Exception as e:
        print(f"❌ Ошибка при чтении {original_file}: {e}")
        return
    
    # Загружаем сгенерированную разметку
    generated_data = {}
    try:
        with open(generated_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        frame_num = int(parts[0])
                        direction = parts[1]
                        generated_data[frame_num] = direction
    except FileNotFoundError:
        print(f"❌ Ошибка: Файл {generated_file} не найден")
        return
    except Exception as e:
        print(f"❌ Ошибка при чтении {generated_file}: {e}")
        return
    
    # Находим общие кадры
    common_frames = set(original_data.keys()) & set(generated_data.keys())
    
    if not common_frames:
        print("❌ Нет общих кадров для сравнения")
        return
    
    # Сравниваем направления
    matches = 0
    total_compared = 0
    discrepancies = []
    
    for frame in sorted(common_frames):
        original_dir = original_data[frame]
        generated_dir = generated_data[frame]
        
        if original_dir == generated_dir:
            matches += 1
        else:
            discrepancies.append((frame, original_dir, generated_dir))
        
        total_compared += 1
    
    # Статистика
    accuracy = (matches / total_compared) * 100 if total_compared > 0 else 0
    
    # Анализ по направлениям
    original_counter = Counter(original_data.values())
    generated_counter = Counter(generated_data.values())
    
    # Сохраняем результаты сравнения
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("СРАВНЕНИЕ РАЗМЕТОК\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Оригинальный файл: {original_file}\n")
        f.write(f"Сгенерированный файл: {generated_file}\n")
        f.write(f"Дата сравнения: {os.path.basename(output_file)}\n\n")
        
        f.write("ОБЩАЯ СТАТИСТИКА:\n")
        f.write(f"  Всего кадров в оригинале: {len(original_data)}\n")
        f.write(f"  Всего кадров в генерации: {len(generated_data)}\n")
        f.write(f"  Общих кадров для сравнения: {total_compared}\n")
        f.write(f"  Совпадений: {matches}\n")
        f.write(f"  Несовпадений: {len(discrepancies)}\n")
        f.write(f"  Точность: {accuracy:.2f}%\n\n")
        
        f.write("РАСПРЕДЕЛЕНИЕ НАПРАВЛЕНИЙ (оригинал):\n")
        for direction, count in original_counter.most_common():
            percentage = (count / len(original_data)) * 100
            f.write(f"  {direction}: {count} ({percentage:.1f}%)\n")
        
        f.write("\nРАСПРЕДЕЛЕНИЕ НАПРАВЛЕНИЙ (генерация):\n")
        for direction, count in generated_counter.most_common():
            percentage = (count / len(generated_data)) * 100
            f.write(f"  {direction}: {count} ({percentage:.1f}%)\n")
        
        if discrepancies:
            f.write(f"\nНЕСОВПАДЕНИЯ (первые 50):\n")
            f.write("Кадр | Оригинал | Генерация\n")
            f.write("-" * 35 + "\n")
            for frame, orig, gen in discrepancies[:50]:
                f.write(f"{frame:5d} | {orig:8s} | {gen:10s}\n")
            
            if len(discrepancies) > 50:
                f.write(f"... и еще {len(discrepancies) - 50} несовпадений\n")
        
        # Анализ точности по каждому направлению
        f.write("\nТОЧНОСТЬ ПО НАПРАВЛЕНИЯМ:\n")
        for direction in ['left', 'right', 'none']:
            direction_frames = [frame for frame in common_frames if original_data[frame] == direction]
            if direction_frames:
                correct = sum(1 for frame in direction_frames if generated_data[frame] == direction)
                dir_accuracy = (correct / len(direction_frames)) * 100
                f.write(f"  {direction}: {correct}/{len(direction_frames)} ({dir_accuracy:.1f}%)\n")
    
    # Вывод в консоль
    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
    print("=" * 50)
    print(f"📊 Точность: {accuracy:.2f}%")
    print(f"✅ Совпадений: {matches}/{total_compared}")
    print(f"❌ Несовпадений: {len(discrepancies)}")
    print(f"📁 Результаты сохранены в: {output_file}")
    
    if accuracy >= 80:
        print("🎉 Отлично! Точность выше 80%")
    elif accuracy >= 60:
        print("👍 Хорошо! Точность выше 60%")
    else:
        print("💡 Нужно улучшить алгоритм")
    
    return accuracy, matches, total_compared

def analyze_specific_frames(original_file, generated_file, frames_to_check):
    """
    Анализирует конкретные кадры
    """
    original_data = {}
    with open(original_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                original_data[int(parts[0])] = parts[1]
    
    generated_data = {}
    with open(generated_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                generated_data[int(parts[0])] = parts[1]
    
    print("\nАНАЛИЗ КОНКРЕТНЫХ КАДРОВ:")
    print("Кадр | Оригинал | Генерация | Статус")
    print("-" * 45)
    
    for frame in frames_to_check:
        orig = original_data.get(frame, "N/A")
        gen = generated_data.get(frame, "N/A")
        status = "✅" if orig == gen else "❌"
        print(f"{frame:4d} | {orig:8s} | {gen:9s} | {status}")

if __name__ == "__main__":
    # Сравниваем разметки
    original_markup = "main/check.txt"  # Ваш оригинальный файл
    generated_markup = "generated_check.txt"  # Сгенерированный файл
    results_file = "markup_comparison.txt"
    
    accuracy, matches, total = compare_markups(original_markup, generated_markup, results_file)
    
    # Дополнительный анализ конкретных кадров (опционально)
    # analyze_specific_frames(original_markup, generated_markup, [100, 200, 300, 400, 500])