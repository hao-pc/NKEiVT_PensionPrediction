import os
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from tkinter import ttk
import csv
import customtkinter as ctk
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

# Настройки интерфейса
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("green")

# Загрузка модели и скейлеров
pnsn_age_scaler = pickle.load(open('scalers/pnsn_age_scaler(WAA).sav', 'rb'))
cprtn_scaler = pickle.load(open('scalers/cprtn_scaler(WAA).sav', 'rb'))
rgn_encoder = pickle.load(open('scalers/rgn_encoder(WAA).sav', 'rb'))
rgn_scaler = pickle.load(open('scalers/rgn_scaler(WAA).sav', 'rb'))
model = load_model('models/Pensia_classifier(WAA1).keras', compile=False)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

# Функция для открытия файла
def open_file():
    file_path = filedialog.askopenfilename(title="Выберите CSV файл", filetypes=[("CSV файлы", "*.csv"), ("Все файлы", "*.*")])
    if file_path:
        process_file(file_path)

# Функция обработки данных и создания submission.csv
def process_file(file_path):
    df = pd.read_csv(file_path, sep=';', encoding='windows-1251')

    finally_df = df.iloc[:, 2:3].copy()

    # Предобработка данных для модели
    x = preprocess_data(df)
    # Получение предсказаний и сохранение в файл
    save_predictions(x, finally_df)

# Функция предобработки
def preprocess_data(df):
    # Удаление ненужных столбцов
    df = df.drop(columns=['slctn_nmbr',
                          'clnt_id',
                          'accnt_id',
                          'accnt_bgn_date',
                          'brth_yr',
                          'brth_plc',
                          'dstrct',
                          'city',
                          'sttlmnt',
                          'pstl_code',
                          'addrss_type',
                          'phn',
                          'email',
                          'lk',
                          'accnt_status',
                          'prsnt_age',
                          'okato'])  # Добавьте все ненужные столбцы
    # Заполнение пропусков
    df['rgn'] = df['rgn'].fillna('Нет данных')
    df = df.fillna(0.0)
    df.loc[df['prvs_npf'] != 0.0, 'prvs_npf'] = 1.0  # если человек впервые воспользовался нашими услугами 0 | 1 - если пришел от других компаний
    df['gndr'] = df['gndr'].replace(['ж', 'м'], [0.0, 1.0])  # 0 - жен | 1 - муж
    df['assgn_npo'] = df['assgn_npo'].replace(['нет', 'да'], [0.0, 1.0])  # 0 - не является правопреемником по договору НПО | 1 - является
    df['assgn_ops'] = df['assgn_ops'].replace(['нет', 'да'], [0.0, 1.0])  # 0 - не является правопреемником по договору ОПС | 1 - является
    # Масштабирование и кодирование
    df['pnsn_age'] = pnsn_age_scaler.transform(df[['pnsn_age']])
    df['cprtn_prd_d'] = cprtn_scaler.transform(df[['cprtn_prd_d']])
    df['rgn'] = rgn_encoder.transform(df['rgn'])
    df['rgn'] = rgn_scaler.transform(df[['rgn']])
    return df.drop(columns=['erly_pnsn_flg'])

# Функция для предсказания и сохранения в файл
def save_predictions(x, df):
    predictions = model.predict(x)
    erly_pnsn_flg = [1 if pred >= 0.9 else 0 for pred in predictions]
    df['erly_pnsn_flg'] = erly_pnsn_flg
    df.to_csv('submission.csv', index=False)
    display_csv_content('submission.csv')

# Функция для отображения содержимого CSV файла
def display_csv_content(file_path):
    for row in table.get_children():
        table.delete(row)
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None)
        if headers:
            table["columns"] = headers
            for header in headers:
                table.heading(header, text=header)
                table.column(header, anchor="w", width=100)
        for row in reader:
            table.insert("", tk.END, values=row)

root = tk.Tk()
root.title("Remote")
logo = ("logo.png")
root.geometry("1280x720")
root.resizable(width=False, height=False)

# bg
root.config(bg='#BDCCE9')

# logo
logoimg = PhotoImage(file=logo)
prlogo = tk.Label(root, image=logoimg, height=128, width=218, border=False)
prlogo.config(bg='#BDCCE9')
prlogo.pack(pady=10)

# Кнопка для выбора файла
open_button = ctk.CTkButton(root, text="Открыть CSV файл", command=open_file)
open_button.pack(pady=10)

# Для отображения содержимого CSV файла
table_frame = ctk.CTkFrame(root, height=300, width=680)
table_frame.pack_propagate(False)  # Запрещаем автоматическое изменение размеров
table_frame.pack(pady=10)

table = ttk.Treeview(table_frame, show="headings")
table.pack(side="left", fill="both", expand=True)
# Добавляем полосы прокрутки к таблице
scrollbar_y = ctk.CTkScrollbar(table_frame, orientation="vertical", command=table.yview)
scrollbar_y.pack(side="right", fill="y")
table.configure(yscrollcommand=scrollbar_y.set)

scrollbar_x = ctk.CTkScrollbar(root, orientation="horizontal", command=table.xview)
scrollbar_x.pack(side="bottom", fill="x")
table.configure(xscrollcommand=scrollbar_x.set)

root.mainloop()