import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

DATA_FILE = os.path.join('data', 'all_data.json')
IMG_DIR = os.path.join('img')
GALLERY_MD = os.path.join('img_gallery.md')

def main():
    df = pd.read_json(DATA_FILE)
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    exclude = {'year', 'month', 'period', 'state'}
    numeric_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    os.makedirs(IMG_DIR, exist_ok=True)
    for col in sorted(numeric_cols):
        sub = df[['date', col]].dropna()
        if sub.empty:
            continue
        plt.figure()
        plt.plot(sub['date'], sub[col])
        plt.title(col.replace('_', ' '))
        plt.xlabel('Date')
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, f"{col}.png"), dpi=300)
        plt.close()
    with open(GALLERY_MD, 'w', encoding='utf-8') as md:
        md.write('## Appendix: All Series Plots\n\n')
        for col in sorted(numeric_cols):
            fname = f"{col}.png"
            caption = col.replace('_', ' ').title()
            md.write(f"![{caption}](img/{fname})  \n")
            md.write(f"*Figure: {caption}*\n\n")

if __name__ == '__main__':
    main()
