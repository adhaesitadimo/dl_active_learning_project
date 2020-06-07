import fire
import shutil
from pathlib import Path
import os


def main(source_dir, output_dir):
    output_dir_conll = Path(output_dir) / 'conll2003' 
    os.makedirs(output_dir_conll, exist_ok=True)
    
    source_dir_conll = Path(source_dir) / 'conll2003'
    shutil.copy(source_dir_conll / 'eng.train.txt', output_dir_conll / 'train.txt')
    shutil.copy(source_dir_conll / 'eng.testa.txt', output_dir_conll / 'dev.txt')
    shutil.copy(source_dir_conll / 'eng.testb.txt', output_dir_conll / 'test.txt')

    
    output_dir_genia = Path(output_dir) / 'genia'
    os.makedirs(output_dir_genia, exist_ok=True)
    
    source_dir_genia = Path(source_dir) / 'genia'
    shutil.copy(source_dir_genia / 'Genia4ERtask1.iob2', output_dir_genia / 'train.txt')
    shutil.copy(source_dir_genia / 'Genia4EReval1.iob2', output_dir_genia / 'dev.txt')
    shutil.copy(source_dir_genia / 'Genia4EReval1.iob2', output_dir_genia / 'test.txt')


if __name__ == "__main__":
    fire.Fire(main)
