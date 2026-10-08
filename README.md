# uru-data-mining

**Note:** This repository is archived and read-only.

Projects and practices from the Data Mining college course (URU), in Python.

## Contents

- **`assignment001/`** — statistics exercises `2_2`, `2_3`, `2_4`, `2_6`, `2_8` (NumPy, SciPy, Matplotlib); `assignment001.bat` runs them all.
- **`stats/distance.py`** — Euclidean, Manhattan and Minkowski distances.
- **`dataset/road-accidents/`** — road accident dataset (original and transformed), `questions.md` and `selected_questions.txt`.
- **`project/`** — analysis of that dataset. `transform.py` drops duplicates, fills missing numeric values with the median and label-encodes categories; exploratory scripts produce heatmaps and pairplots; `question1.py` to `question6.py` answer six questions with random forest and OLS, ridge, lasso, elastic net and mutual-information regressions. Outputs and analyses are saved as `*_output.txt` and `*_analysis.txt`.

The six questions cover predicting accident severity from weather, visibility and time of day; the effect of speed limit and traffic volume; factors behind economic loss; severity by road type and urban/rural area; driver alcohol level; and emergency response time.

## Running

`requirements.txt` is UTF-16 encoded (convert it if pip complains) and omits `scikit-learn` and `statsmodels`, which the scripts import. Run from the repository root, since dataset paths depend on the working directory.

```bash
pip install numpy pandas scipy seaborn matplotlib scikit-learn statsmodels
python -m project.transform
python -m project.question1
python -m assignment001.2_2
```

## License

GNU General Public License v3.0 (see `LICENSE`).
