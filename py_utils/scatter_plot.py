from plotly.subplots import make_subplots
import plotly.express as px
from plotly.offline import plot
import pandas as pd
import argparse

from sklearn.decomposition import PCA


# print(px.colors.qualitative.Alphabet_r)
# exit(0)

argParser = argparse.ArgumentParser()
argParser.add_argument("-d", "--dimensions", help="The number of dimensions of the points", type=int,  required=True)
argParser.add_argument("-f", "--input-files", help="The CSV file(s) to read from", type=str, nargs='+', required=True)
args = argParser.parse_args()

print(args.input_files)
files_count = len(args.input_files)
figures = []
# pd.read_csv(sys.stdin if args.input_files == None else args.input_files)

for i in range(files_count):
  df = pd.read_csv(args.input_files[i])
  print(df)
  plt = None

  if args.dimensions == 2:
    plt = px.scatter(df, x=df.columns[1], y=df.columns[2], color=df.columns[0])
  elif args.dimensions == 3:
    plt = px.scatter_3d(df, x=df.columns[1], y=df.columns[2], z=df.columns[3], color=df.columns[0])
  else:
    # PCA time
    pca = PCA(n_components=2)
    X = df.iloc[:, 1:]
    pca.fit(X)
    X = pca.transform(X)
    plt = px.scatter(df, x=X[:, 0], y=X[:,1], color=df.columns[0])


  plt.update_layout(title_text=args.input_files[i])
  figures.append(plt)

fig = make_subplots(rows=len(figures), cols=1, subplot_titles=args.input_files)
fig.update_layout(title_text="K-means results comparison")

for i, figure in enumerate(figures):
    for trace in range(len(figure["data"])):
        fig.append_trace(figure["data"][trace], row=i+1, col=1)
        
plot(fig)
