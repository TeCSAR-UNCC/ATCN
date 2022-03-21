import os
from pathlib import Path
import re
import pandas as pd
from model.config import config

def gather(path, best_value_filename='best_value.txt', save_to_xlsx=True, results_file_name='Results.xlsx'):
    subfolders = [f.name for f in os.scandir(path) if f.is_dir()]
    values = {}
    for sub_folder in subfolders:
        file = os.path.join(path, sub_folder, best_value_filename)
        if os.path.isfile(file):
            with open(file, 'r') as file_handler:
                data = file_handler.readline()
                value = re.findall(r"\d+\.\d+", data)[0]
                values[sub_folder] = float(value)
    if save_to_xlsx:
        data = pd.DataFrame({'Dataset': values.keys(), 'Accuracy': values.values()})
        writer = pd.ExcelWriter(results_file_name, engine='xlsxwriter')

        data.to_excel(writer, sheet_name='UCR', index=False)

        workbook  = writer.book
        worksheet = writer.sheets['UCR']

        font_fmt = workbook.add_format({'font_name': 'Arial', 'font_size': 10})
        header_fmt = workbook.add_format({'font_name': 'Arial', 'font_size': 10, 'bold': True})

        worksheet.set_column('A:A', None, font_fmt)
        worksheet.set_row(0, None, header_fmt)

        writer.save()
    print('-> Done')
    return values

if __name__ == "__main__":
    file = Path(__file__).resolve()
    root_directory = file.parents[0]
    folder_prfx = 'TNHNK'
    gather(os.path.join(root_directory, 'checkpoints', folder_prfx), results_file_name='{}_best_results.xlsx'.format(config[folder_prfx]['Paper_name']))
                